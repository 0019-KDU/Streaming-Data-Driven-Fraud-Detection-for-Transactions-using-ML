"""
Hybrid Decision Engine for fraud detection.

Combines ML probability with velocity, ATO, and rule-based signals
to make final fraud decisions using adaptive thresholds.

Key Features:
- Base threshold from F1-optimal training (0.0564, NOT 0.5)
- Hybrid adaptive threshold with velocity/amount/ATO adjustments
- Dynamic weight selection based on risk profile
- Four decision levels: APPROVE, REVIEW, HOLD, BLOCK
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .logging_utils import setup_logger

logger = setup_logger(__name__)


class Decision(str, Enum):
    """Fraud decision levels."""
    APPROVE = "APPROVE"   # Low risk, allow transaction
    REVIEW = "REVIEW"     # Medium risk, manual review
    HOLD = "HOLD"         # High risk, temporary hold
    BLOCK = "BLOCK"       # Very high risk, block immediately


class RiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class DecisionResult:
    """Final fraud detection decision."""
    decision: Decision
    risk_level: RiskLevel
    risk_factors: List[str]
    
    # Scores
    fraud_probability: float
    adjusted_probability: float
    hybrid_threshold: float
    
    # Component risks
    velocity_risk: float
    amount_risk: float
    ato_risk: float
    
    # Flags
    rule_triggered: bool = False
    ato_detected: bool = False


class HybridDecisionEngine:
    """
    Hybrid decision engine combining ML + heuristics.
    
    Uses adaptive thresholds that adjust based on:
    - Velocity risk (high frequency transactions)
    - Amount risk (unusual transaction amounts)
    - ATO risk (account takeover signals)
    
    IMPORTANT: Base threshold is loaded from model bundle (F1-optimal from training),
    NOT hardcoded or from config.
    """
    
    def __init__(self, config, model_metadata: Dict = None):
        """
        Initialize decision engine.
        
        Args:
            config: Config object with threshold settings
            model_metadata: Optional model metadata with threshold
        """
        self.config = config
        
        # ✅ FIX #1: Load threshold from model metadata if available
        if model_metadata and 'threshold' in model_metadata:
            self.base_threshold = model_metadata['threshold']
            logger.info(f"✅ Using threshold from model metadata: {self.base_threshold:.4f}")
        else:
            self.base_threshold = config.model.base_threshold
            logger.warning(f"⚠️ No model metadata, using config threshold: {self.base_threshold:.4f}")
        
        # Decision bands
        self.review_threshold = config.model.review_threshold  # 0.10
        self.hold_threshold = config.model.hold_threshold      # 0.20
        self.block_threshold = config.model.block_threshold    # 0.20
        
        # Adjustment weights
        self.velocity_adj = config.threshold_weights
        self.amount_adj = 0.10
        self.ato_adj = 0.25
        
        # Threshold weight profiles
        self.weights = {
            'high_ato': config.threshold_weights.high_ato,
            'high_velocity': config.threshold_weights.high_velocity,
            'high_amount': config.threshold_weights.high_amount,
            'normal': config.threshold_weights.normal
        }
        
        logger.info(
            f"DecisionEngine initialized with base_threshold={self.base_threshold}, "
            f"review={self.review_threshold}, hold={self.hold_threshold}"
        )
    
    def make_decision(
        self,
        fraud_probability: float,
        velocity_risk: float,
        amount_risk: float,
        ato_risk: float,
        ato_detected: bool,
        transaction_data: Dict,
        velocity_factors: List[str] = None,
        ato_factors: List[str] = None
    ) -> DecisionResult:
        """
        Make final fraud decision combining all signals.
        
        Args:
            fraud_probability: ML model fraud probability [0, 1]
            velocity_risk: Velocity risk score [0, 1]
            amount_risk: Amount risk score [0, 1]
            ato_risk: ATO risk score [0, 1]
            ato_detected: Boolean ATO flag
            transaction_data: Transaction fields (for rule checks)
            velocity_factors: Risk factors from velocity service
            ato_factors: Risk factors from ATO service
            
        Returns:
            DecisionResult with final decision and risk factors
        """
        if velocity_factors is None:
            velocity_factors = []
        if ato_factors is None:
            ato_factors = []
        
        # Calculate hybrid adaptive threshold
        hybrid_threshold = self._calculate_hybrid_threshold(
            velocity_risk, amount_risk, ato_risk
        )
        
        # Adjust probability based on risk signals
        adjusted_prob = self._adjust_probability(
            fraud_probability, velocity_risk, amount_risk, ato_risk
        )
        
        # Check rule-based triggers
        rule_triggered, rule_factors = self._check_rules(transaction_data)
        
        # Make final decision
        decision, risk_level = self._determine_decision(
            adjusted_prob,
            hybrid_threshold,
            ato_detected,
            rule_triggered
        )
        
        # Collect all risk factors
        all_factors = self._collect_risk_factors(
            fraud_probability,
            adjusted_prob,
            velocity_factors,
            ato_factors,
            rule_factors,
            transaction_data
        )
        
        return DecisionResult(
            decision=decision,
            risk_level=risk_level,
            risk_factors=all_factors,
            fraud_probability=fraud_probability,
            adjusted_probability=adjusted_prob,
            hybrid_threshold=hybrid_threshold,
            velocity_risk=velocity_risk,
            amount_risk=amount_risk,
            ato_risk=ato_risk,
            rule_triggered=rule_triggered,
            ato_detected=ato_detected
        )
    
    def _calculate_hybrid_threshold(
        self,
        velocity_risk: float,
        amount_risk: float,
        ato_risk: float
    ) -> float:
        """
        Calculate hybrid adaptive threshold.
        
        Formula:
            τ_velocity = τ_base - velocity_risk * 0.15
            τ_amount   = τ_base - amount_risk * 0.10
            τ_ato      = τ_base - ato_risk * 0.25
            
            τ_hybrid = w0*τ_base + w1*τ_velocity + w2*τ_amount + w3*τ_ato
        
        Weights depend on which risk dominates:
        - High ATO: [0.1, 0.2, 0.1, 0.6]
        - High velocity: [0.2, 0.5, 0.2, 0.1]
        - High amount: [0.2, 0.2, 0.5, 0.1]
        - Normal: [0.6, 0.2, 0.1, 0.1]
        """
        # Component thresholds
        tau_base = self.base_threshold
        tau_velocity = max(0.01, tau_base - velocity_risk * 0.15)
        tau_amount = max(0.01, tau_base - amount_risk * 0.10)
        tau_ato = max(0.01, tau_base - ato_risk * 0.25)
        
        # Select weights based on dominant risk
        if ato_risk > 0.6:
            weights = self.weights['high_ato']
        elif velocity_risk > 0.7:
            weights = self.weights['high_velocity']
        elif amount_risk > 0.7:
            weights = self.weights['high_amount']
        else:
            weights = self.weights['normal']
        
        # Calculate weighted threshold
        tau_hybrid = (
            weights[0] * tau_base +
            weights[1] * tau_velocity +
            weights[2] * tau_amount +
            weights[3] * tau_ato
        )
        
        # Bound threshold to reasonable range
        return np.clip(tau_hybrid, 0.01, 0.50)
    
    def _adjust_probability(
        self,
        fraud_prob: float,
        velocity_risk: float,
        amount_risk: float,
        ato_risk: float
    ) -> float:
        """
        Adjust ML probability based on additional risk signals.
        
        Boosts probability when velocity/amount/ATO signals are high.
        """
        adjusted = fraud_prob
        
        # Velocity boost
        if velocity_risk > 0.8:
            adjusted += 0.15 * (1 - adjusted)
        elif velocity_risk > 0.6:
            adjusted += 0.10 * (1 - adjusted)
        
        # Amount boost
        if amount_risk > 0.8:
            adjusted += 0.10 * (1 - adjusted)
        elif amount_risk > 0.6:
            adjusted += 0.05 * (1 - adjusted)
        
        # ATO boost (strongest signal)
        if ato_risk > 0.8:
            adjusted += 0.20 * (1 - adjusted)
        elif ato_risk > 0.6:
            adjusted += 0.15 * (1 - adjusted)
        
        return min(adjusted, 1.0)
    
    def _check_rules(self, transaction_data: Dict) -> Tuple[bool, List[str]]:
        """
        Check rule-based fraud triggers.
        
        Returns:
            (triggered, factors) tuple
        """
        triggered = False
        factors = []
        
        amount = transaction_data.get('TransactionAmt', 0)
        
        # Extreme high amount
        if amount >= 10000:
            triggered = True
            factors.append(f"extreme_amount_{amount:.0f}")
        
        # Night transaction with high amount
        hour = transaction_data.get('dt_hour')
        if hour and ((hour >= 22) or (hour <= 5)) and amount >= 1000:
            triggered = True
            factors.append(f"night_high_amount_{amount:.0f}")
        
        # Risky email domain
        email_domain = transaction_data.get('P_emaildomain', '')
        risky_domains = {
            'anonymous.com', 'mailinator.com', 'tempmail.com',
            '10minutemail.com', 'guerrillamail.com'
        }
        if email_domain in risky_domains:
            triggered = True
            factors.append(f"risky_email_{email_domain}")
        
        # Suspicious product + amount combination
        product = transaction_data.get('ProductCD', '')
        if product == 'W' and amount >= 5000:  # Wire transfer
            triggered = True
            factors.append(f"suspicious_wire_transfer_{amount:.0f}")
        
        return triggered, factors
    
    def _determine_decision(
        self,
        adjusted_prob: float,
        hybrid_threshold: float,
        ato_detected: bool,
        rule_triggered: bool
    ) -> Tuple[Decision, RiskLevel]:
        """
        Determine final decision and risk level.
        
        Decision logic:
        - prob >= 0.20 → BLOCK (HIGH)
        - prob >= 0.10 → HOLD (HIGH)
        - prob >= τ_hybrid OR ato_detected OR rule_triggered → REVIEW (MEDIUM)
        - else → APPROVE (LOW)
        """
        # BLOCK: Very high probability
        if adjusted_prob >= self.block_threshold:
            return Decision.BLOCK, RiskLevel.HIGH
        
        # HOLD: High probability
        if adjusted_prob >= self.hold_threshold:
            return Decision.HOLD, RiskLevel.HIGH
        
        # REVIEW: Above hybrid threshold OR ATO OR rule triggered
        if adjusted_prob >= hybrid_threshold or ato_detected or rule_triggered:
            return Decision.REVIEW, RiskLevel.MEDIUM
        
        # APPROVE: Low risk
        return Decision.APPROVE, RiskLevel.LOW
    
    def _collect_risk_factors(
        self,
        fraud_prob: float,
        adjusted_prob: float,
        velocity_factors: List[str],
        ato_factors: List[str],
        rule_factors: List[str],
        transaction_data: Dict
    ) -> List[str]:
        """Collect all risk factors into a single list."""
        factors = []
        
        # ML score
        if fraud_prob >= 0.20:
            factors.append(f"high_ml_score_{fraud_prob:.3f}")
        elif fraud_prob >= 0.10:
            factors.append(f"medium_ml_score_{fraud_prob:.3f}")
        
        # Velocity factors
        factors.extend(velocity_factors)
        
        # ATO factors
        factors.extend(ato_factors)
        
        # Rule factors
        factors.extend(rule_factors)
        
        # Time-based factors
        hour = transaction_data.get('dt_hour')
        if hour:
            if hour >= 22 or hour <= 5:
                factors.append("night_transaction")
            elif 9 <= hour <= 17:
                factors.append("business_hours")
        
        # Amount factors
        amount = transaction_data.get('TransactionAmt')
        if amount:
            if amount >= 5000:
                factors.append(f"very_high_amount_{amount:.0f}")
            elif amount >= 2000:
                factors.append(f"high_amount_{amount:.0f}")
            elif amount <= 1:
                factors.append(f"micro_transaction_{amount:.2f}")
        
        # Deduplicate
        return list(dict.fromkeys(factors))
