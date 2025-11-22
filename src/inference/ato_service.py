"""
Account Takeover (ATO) detection service using Redis.

Tracks user baselines (geo, device, email) and detects anomalies
that indicate potential account takeover.
"""

import json
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import redis

from .utils.redis_client import get_redis_client
from .logging_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class ATOResult:
    """Result from ATO analysis."""
    ato_risk: float         # [0, 1]
    ato_detected: bool      # True if risk > threshold
    factors: List[str]      # Risk factor descriptions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ATOService:
    """
    Tracks user behavior baselines and detects Account Takeover attempts.

    For each card1, maintains:
    - Usual geo locations (distances)
    - Usual IP addresses / devices
    - Usual email domains
    - Usual transaction patterns
    """

    def __init__(self, config):
        """
        Initialize ATO service.

        Args:
            config: Config object with ATO and Redis settings
        """
        self.config = config
        self.redis_client = get_redis_client(config)
        self.ttl = config.redis.ato_baseline_ttl
        self.risky_email_domains = set(config.ato.risky_email_domains)

    def _get_baseline_key(self, card1: str) -> str:
        """Generate Redis key for baseline data."""
        return f"ato:baseline:{card1}"

    def _haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate great circle distance between two points in kilometers.

        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates

        Returns:
            Distance in kilometers
        """
        R = 6371  # Earth's radius in kilometers

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def update_baseline(
        self,
        card1: str,
        transaction: Dict[str, Any]
    ) -> None:
        """
        Update baseline profile for a card.

        Args:
            card1: Card identifier
            transaction: Transaction dictionary with fields:
                - addr1, addr2 (addresses)
                - dist1, dist2 (distances)
                - P_emaildomain, R_emaildomain
                - DeviceInfo, DeviceType
                - card4, card6 (card types)
                - TransactionAmt
        """
        baseline_key = self._get_baseline_key(card1)

        # Get current baseline
        baseline_data = self.redis_client.get(baseline_key)
        if baseline_data:
            baseline = json.loads(baseline_data)
        else:
            baseline = {
                'addresses': [],
                'email_domains': [],
                'devices': [],
                'card_types': [],
                'avg_amount': 0.0,
                'transaction_count': 0
            }

        # Update addresses
        addr1 = transaction.get('addr1')
        if addr1 and addr1 not in baseline['addresses']:
            baseline['addresses'].append(addr1)
            # Keep only last 5 addresses
            baseline['addresses'] = baseline['addresses'][-5:]

        # Update email domains
        for email_field in ['P_emaildomain', 'R_emaildomain']:
            email = transaction.get(email_field)
            if email and email not in baseline['email_domains']:
                baseline['email_domains'].append(email)
                baseline['email_domains'] = baseline['email_domains'][-5:]

        # Update devices
        device_info = transaction.get('DeviceInfo')
        if device_info and device_info not in baseline['devices']:
            baseline['devices'].append(device_info)
            baseline['devices'] = baseline['devices'][-5:]

        # Update card types
        card4 = transaction.get('card4')
        if card4 and card4 not in baseline['card_types']:
            baseline['card_types'].append(card4)
            baseline['card_types'] = baseline['card_types'][-3:]

        # Update average amount (running average)
        count = baseline['transaction_count']
        avg = baseline['avg_amount']
        current_amt = transaction.get('TransactionAmt', 0.0)

        baseline['avg_amount'] = (avg * count + current_amt) / (count + 1)
        baseline['transaction_count'] = count + 1

        # Save updated baseline
        self.redis_client.setex(
            baseline_key,
            self.ttl,
            json.dumps(baseline)
        )

    def analyze_ato(
        self,
        card1: str,
        transaction: Dict[str, Any]
    ) -> ATOResult:
        """
        Analyze transaction for Account Takeover signals.

        Args:
            card1: Card identifier
            transaction: Transaction dictionary

        Returns:
            ATOResult with risk score and factors
        """
        baseline_key = self._get_baseline_key(card1)
        baseline_data = self.redis_client.get(baseline_key)

        ato_risk = 0.0
        factors = []

        # If no baseline, this is first transaction (low risk)
        if not baseline_data:
            return ATOResult(ato_risk=0.0, ato_detected=False, factors=['first_transaction'])

        baseline = json.loads(baseline_data)

        # 1. Check geo anomaly (distance)
        dist1 = transaction.get('dist1', 0.0)
        if dist1 and dist1 > self.config.ato.geo_anomaly_distance:
            ato_risk += self.config.ato.geo_anomaly_weight
            factors.append(f"geo_anomaly:{dist1:.0f}km")

        # 2. Check new address
        addr1 = transaction.get('addr1')
        if addr1 and addr1 not in baseline['addresses']:
            ato_risk += self.config.ato.new_address_ip_weight
            factors.append("new_address")

        # 3. Check email mismatch or risky domain
        p_email = transaction.get('P_emaildomain', '')
        r_email = transaction.get('R_emaildomain', '')

        # Email mismatch
        if p_email and r_email and p_email != r_email:
            if p_email not in baseline['email_domains'] and r_email not in baseline['email_domains']:
                ato_risk += self.config.ato.email_mismatch_weight
                factors.append("email_mismatch_new")

        # Risky email domain
        if p_email in self.risky_email_domains:
            ato_risk += self.config.ato.email_mismatch_weight * 0.5
            factors.append(f"risky_email:{p_email}")

        # 4. Check new device
        device_info = transaction.get('DeviceInfo')
        if device_info and device_info not in baseline['devices'] and len(baseline['devices']) > 0:
            ato_risk += self.config.ato.new_address_ip_weight * 0.5
            factors.append("new_device")

        # 5. Check high amount (vs baseline)
        current_amt = transaction.get('TransactionAmt', 0.0)
        avg_amt = baseline.get('avg_amount', 0.0)

        if current_amt > self.config.ato.high_amount_threshold:
            ato_risk += self.config.ato.high_amount_weight
            factors.append(f"high_amount:${current_amt:.0f}")
        elif avg_amt > 0 and current_amt > 3 * avg_amt:
            ato_risk += self.config.ato.high_amount_weight * 0.7
            factors.append(f"amount_3x_baseline:{current_amt/avg_amt:.1f}x")

        # 6. Check unusual card type
        card4 = transaction.get('card4')
        if card4 and card4 not in baseline['card_types'] and len(baseline['card_types']) > 0:
            ato_risk += self.config.ato.unusual_card_weight
            factors.append(f"unusual_card_type:{card4}")

        # Cap at 1.0
        ato_risk = min(1.0, ato_risk)

        # Detect ATO
        ato_detected = ato_risk >= self.config.ato.ato_detection_threshold

        if ato_detected:
            logger.warning(f"ATO detected for card1={card1}, risk={ato_risk:.2f}, factors={factors}")

        return ATOResult(
            ato_risk=ato_risk,
            ato_detected=ato_detected,
            factors=factors
        )

    def process_transaction(
        self,
        card1: str,
        transaction: Dict[str, Any]
    ) -> ATOResult:
        """
        Analyze transaction and update baseline.

        Args:
            card1: Card identifier
            transaction: Transaction dictionary

        Returns:
            ATOResult
        """
        # First analyze (before updating baseline)
        result = self.analyze_ato(card1, transaction)

        # Then update baseline (for next transaction)
        self.update_baseline(card1, transaction)

        return result
