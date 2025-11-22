"""
Hybrid Decision Engine with Adaptive Threshold Logic.

Combines ML model probability with velocity, ATO, and rule-based signals
to make final fraud detection decisions.

Uses base threshold of 0.0564 (F1-optimal from training), NOT 0.5.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
from enum import Enum

from .logging_utils import setup_logger

logger = setup_logger(__name__)


class Decision(str, Enum):
    """Final decision types."""
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    HOLD = "HOLD"
    BLOCK = "BLOCK"


class RiskLevel(str, Enum):
    """Risk level classifications."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class DecisionResult:
    """Result from decision engine."""
    decision: Decision
    risk_level: RiskLevel
    fraud_probability: float
    adjusted_probability: float
    risk_factors: List[str]
    ato_risk: float
    velocity_risk: float
    amount_risk: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'decision': self.decision.value,
            'risk_level': self.risk_level.value,
            'fraud_probability': self.fraud_probability,
            'adjusted_probability': self.adjusted_probability,
            'risk_factors': self.risk_factors,
            'ato_risk': self.ato_risk,
            'velocity_risk': self.velocity_risk,
            'amount_risk': self.amount_risk
        }


class DecisionEngine:
    """
    Hybrid decision engine combining ML model, velocity, ATO, and rules.

    Decision Logic:
    1. Compute adaptive threshold based on velocity/ATO/amount risks
    2. Adjust ML probability based on contextual risks
    3. Apply threshold bands:
       - prob >= 0.20 → BLOCK
       - prob >= 0.10 → HOLD
       - prob >= τ_hybrid OR rule_flags → REVIEW
       - else → APPROVE
    """

    def __init__(self, config):
        """
        Initialize decision engine.

        Args:
            config: Config object with threshold settings
        """
        self.config = config

        # Base threshold from training (F1-optimal = 0.0564)
        self.base_threshold = config.model.base_threshold

        # Decision thresholds
        self.review_threshold = config.model.review_threshold   # 0.10
        self.hold_threshold = config.model.hold_threshold       # 0.20
        self.block_threshold = config.model.block_threshold     # 0.20

        # Threshold weights for different risk scenarios
        self.weights = config.threshold_weights

        # Risky email domains (for rule-based checks)
        self.risky_email_domains = set(config.ato.risky_email_domains)

    def compute_adaptive_threshold(
        self,
        velocity_risk: float,
        amount_risk: float,
        ato_risk: float
    ) -> Tuple[float, List[float]]:
        """
        Compute adaptive threshold based on contextual risks.

        Formula:
            τ_velocity = τ_base - velocity_risk * 0.15
            τ_amount   = τ_base - amount_risk * 0.10
            τ_ato      = τ_base - ato_risk * 0.25

        Then select weights based on dominant risk:
            - High ATO: [0.1, 0.2, 0.1, 0.6]
            - High velocity: [0.2, 0.5, 0.2, 0.1]
            - High amount: [0.2, 0.2, 0.5, 0.1]
            - Normal: [0.6, 0.2, 0.1, 0.1]

        τ_hybrid = w0*τ_base + w1*τ_velocity + w2*τ_amount + w3*τ_ato

        Args:
            velocity_risk: [0, 1]
            amount_risk: [0, 1]
            ato_risk: [0, 1]

        Returns:
            (adaptive_threshold, weights_used)
        """
        # Compute risk-adjusted thresholds
        tau_base = self.base_threshold
        tau_velocity = tau_base - velocity_risk * 0.15
        tau_amount = tau_base - amount_risk * 0.10
        tau_ato = tau_base - ato_risk * 0.25

        # Select weights based on dominant risk
        if ato_risk >= 0.6:
            weights = self.weights.high_ato
            mode = "high_ato"
        elif velocity_risk >= 0.7:
            weights = self.weights.high_velocity
            mode = "high_velocity"
        elif amount_risk >= 0.7:
            weights = self.weights.high_amount
            mode = "high_amount"
        else:
            weights = self.weights.normal
            mode = "normal"

        # Compute weighted adaptive threshold
        tau_hybrid = (
            weights[0] * tau_base +
            weights[1] * tau_velocity +
            weights[2] * tau_amount +
            weights[3] * tau_ato
        )

        # Clamp to reasonable range [0.01, 0.20]
        tau_hybrid = max(0.01, min(0.20, tau_hybrid))

        logger.debug(
            f"Adaptive threshold: {tau_hybrid:.4f} (mode={mode}, "
            f"vel={velocity_risk:.2f}, amt={amount_risk:.2f}, ato={ato_risk:.2f})"
        )

        return tau_hybrid, weights

    def apply_rule_based_checks(
        self,
        transaction: Dict[str, Any]
    ) -> List[str]:
        """
        Apply rule-based fraud checks.

        Args:
            transaction: Transaction dictionary

        Returns:
            List of triggered rule names
        """
        triggered_rules = []

        # Check risky email domain
        p_email = transaction.get('P_emaildomain', '')
        if p_email in self.risky_email_domains:
            triggered_rules.append(f"risky_email:{p_email}")

        # Check extreme amount
        amount = transaction.get('TransactionAmt', 0.0)
        if amount > 5000:
            triggered_rules.append(f"extreme_amount:${amount:.0f}")

        # Check night transaction (if time features available)
        dt_hour = transaction.get('dt_hour')
        if dt_hour is not None and (dt_hour >= 22 or dt_hour <= 6):
            triggered_rules.append("night_transaction")

        # Check weekend transaction
        dt_is_weekend = transaction.get('dt_is_weekend')
        if dt_is_weekend == 1:
            triggered_rules.append("weekend_transaction")

        # Check email mismatch
        r_email = transaction.get('R_emaildomain', '')
        if p_email and r_email and p_email != r_email:
            triggered_rules.append("email_mismatch")

        # Check high-risk card types (discover, amex sometimes higher fraud)
        card4 = transaction.get('card4', '')
        if card4 in ['discover', 'american express']:
            triggered_rules.append(f"highrisk_card:{card4}")

        return triggered_rules

    def make_decision(
        self,
        fraud_probability: float,
        velocity_risk: float,
        amount_risk: float,
        ato_risk: float,
        ato_detected: bool,
        transaction: Dict[str, Any],
        velocity_factors: List[str] = None,
        ato_factors: List[str] = None
    ) -> DecisionResult:
        """
        Make final fraud detection decision.

        Decision logic:
        1. adjusted_prob = fraud_probability + 0.05 * velocity_risk + 0.10 * ato_risk
        2. if adjusted_prob >= 0.20 → BLOCK
        3. elif adjusted_prob >= 0.10 → HOLD
        4. elif adjusted_prob >= τ_hybrid OR rule_triggered OR ato_detected → REVIEW
        5. else → APPROVE

        Args:
            fraud_probability: ML model probability [0, 1]
            velocity_risk: Velocity risk score [0, 1]
            amount_risk: Amount risk score [0, 1]
            ato_risk: ATO risk score [0, 1]
            ato_detected: Whether ATO threshold was exceeded
            transaction: Transaction dictionary
            velocity_factors: List of velocity risk factors
            ato_factors: List of ATO risk factors

        Returns:
            DecisionResult
        """
        if velocity_factors is None:
            velocity_factors = []
        if ato_factors is None:
            ato_factors = []

        # Compute adaptive threshold
        tau_hybrid, weights = self.compute_adaptive_threshold(
            velocity_risk, amount_risk, ato_risk
        )

        # Adjust probability with contextual risks
        adjusted_prob = fraud_probability + 0.05 * velocity_risk + 0.10 * ato_risk

        # Apply rule-based checks
        rule_triggered = self.apply_rule_based_checks(transaction)

        # Collect all risk factors
        risk_factors = []
        if fraud_probability >= self.base_threshold:
            risk_factors.append(f"model_prob:{fraud_probability:.3f}")
        risk_factors.extend(velocity_factors)
        risk_factors.extend(ato_factors)
        risk_factors.extend(rule_triggered)

        # Decision logic
        if adjusted_prob >= self.block_threshold:
            decision = Decision.BLOCK
            risk_level = RiskLevel.HIGH
        elif adjusted_prob >= self.hold_threshold:
            decision = Decision.HOLD
            risk_level = RiskLevel.HIGH
        elif (
            adjusted_prob >= tau_hybrid or
            len(rule_triggered) >= 2 or
            ato_detected
        ):
            decision = Decision.REVIEW
            risk_level = RiskLevel.MEDIUM
        else:
            decision = Decision.APPROVE
            risk_level = RiskLevel.LOW

        result = DecisionResult(
            decision=decision,
            risk_level=risk_level,
            fraud_probability=fraud_probability,
            adjusted_probability=adjusted_prob,
            risk_factors=risk_factors,
            ato_risk=ato_risk,
            velocity_risk=velocity_risk,
            amount_risk=amount_risk
        )

        logger.info(
            f"Decision: {decision.value}, RiskLevel: {risk_level.value}, "
            f"Prob: {fraud_probability:.4f}, Adjusted: {adjusted_prob:.4f}, "
            f"Threshold: {tau_hybrid:.4f}, Factors: {len(risk_factors)}"
        )

        return result
