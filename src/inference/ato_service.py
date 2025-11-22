"""
Account Takeover (ATO) detection service using Redis for baseline tracking.

Detects anomalous behavior patterns that indicate account takeover:
- Geographic anomalies (unusual locations)
- Device changes (new devices/browsers)
- Email mismatches
- Unusual transaction patterns

Maintains baseline behavior profiles in Redis and compares current
transactions against historical patterns.
"""

import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import redis
import numpy as np

from .logging_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class ATOResult:
    """Result from ATO check."""
    ato_risk: float       # 0-1 risk score
    ato_detected: bool    # True if risk > threshold
    factors: List[str]    # Human-readable risk factors
    
    # Signal breakdown
    geo_anomaly: float = 0.0
    device_anomaly: float = 0.0
    email_anomaly: float = 0.0
    amount_anomaly: float = 0.0


class ATOService:
    """
    Redis-backed Account Takeover detection service.
    
    Tracks baseline behavior for each card and detects deviations
    that indicate potential account takeover attacks.
    """
    
    def __init__(self, redis_client: redis.Redis, config):
        """
        Initialize ATO service.
        
        Args:
            redis_client: Redis client instance
            config: Config object with ATO settings
        """
        self.redis = redis_client
        self.config = config
        
        # ✅ FIX #3: Add environment/version namespacing
        import os
        self.env = os.getenv('ENVIRONMENT', 'prod')
        self.model_version = os.getenv('MODEL_VERSION', 'v1')
        self.key_prefix = f"{self.env}:{self.model_version}"
        
        # Risk weights from config
        self.geo_weight = config.ato.geo_anomaly_weight
        self.device_weight = config.ato.new_address_ip_weight
        self.email_weight = config.ato.email_mismatch_weight
        self.amount_weight = config.ato.high_amount_weight
        self.card_weight = config.ato.unusual_card_weight
        
        # Thresholds
        self.geo_distance_threshold = config.ato.geo_anomaly_distance
        self.high_amount_threshold = config.ato.high_amount_threshold
        self.ato_threshold = config.ato.ato_detection_threshold
        
        # Risky domains
        self.risky_domains = set(config.ato.risky_email_domains)
        
        # TTL for baseline (30 days)
        self.baseline_ttl = config.redis.ato_baseline_ttl
        
        logger.info(f"ATOService initialized with namespace: {self.key_prefix}")
        logger.info(f"ATOService threshold: {self.ato_threshold}")
    
    def check_ato(
        self,
        card_id: str,
        transaction_data: Dict
    ) -> ATOResult:
        """
        Check ATO risk for a transaction.
        
        Args:
            card_id: Card identifier (card1)
            transaction_data: Dictionary with transaction fields:
                - addr1, addr2: Address fields
                - dist1, dist2: Distance fields
                - device_info: Device information
                - device_type: Device type
                - email_domain: Email domain
                - amount: Transaction amount
                - card_type: Card type (card4, card6)
                
        Returns:
            ATOResult with risk scores and factors
        """
        # Get baseline for this card
        baseline = self._get_baseline(card_id)
        
        # Calculate individual risk signals
        geo_anomaly = self._check_geo_anomaly(transaction_data, baseline)
        device_anomaly = self._check_device_anomaly(transaction_data, baseline)
        email_anomaly = self._check_email_anomaly(transaction_data, baseline)
        amount_anomaly = self._check_amount_anomaly(transaction_data, baseline)
        card_anomaly = self._check_card_type_anomaly(transaction_data, baseline)
        
        # Calculate composite ATO risk
        ato_risk = (
            geo_anomaly * self.geo_weight +
            device_anomaly * self.device_weight +
            email_anomaly * self.email_weight +
            amount_anomaly * self.amount_weight +
            card_anomaly * self.card_weight
        )
        
        # Cap at 1.0
        ato_risk = min(ato_risk, 1.0)
        
        # Determine if ATO detected
        ato_detected = ato_risk >= self.ato_threshold
        
        # Identify risk factors
        factors = self._identify_risk_factors(
            geo_anomaly, device_anomaly, email_anomaly,
            amount_anomaly, card_anomaly, transaction_data
        )
        
        # Update baseline with this transaction
        self._update_baseline(card_id, transaction_data)
        
        return ATOResult(
            ato_risk=ato_risk,
            ato_detected=ato_detected,
            factors=factors,
            geo_anomaly=geo_anomaly,
            device_anomaly=device_anomaly,
            email_anomaly=email_anomaly,
            amount_anomaly=amount_anomaly
        )
    
    def _get_baseline(self, card_id: str) -> Dict:
        """Get baseline behavior profile for card from Redis."""
        key = f"ato:baseline:{card_id}"
        
        try:
            baseline_json = self.redis.get(key)
            if baseline_json:
                return json.loads(baseline_json)
            else:
                return {}
        except redis.RedisError as e:
            logger.warning(f"Redis error getting baseline: {e}")
            return {}
    
    def _update_baseline(self, card_id: str, transaction_data: Dict) -> None:
        """Update baseline profile with new transaction."""
        key = f"ato:baseline:{card_id}"
        
        try:
            # Get current baseline
            baseline = self._get_baseline(card_id)
            
            # Update with new transaction data
            # Use sets to track unique values
            if 'addr1' in transaction_data:
                baseline.setdefault('addresses', []).append(
                    str(transaction_data['addr1'])
                )
                # Keep only last 10 addresses
                baseline['addresses'] = baseline['addresses'][-10:]
            
            if 'device_info' in transaction_data:
                baseline.setdefault('devices', []).append(
                    str(transaction_data['device_info'])
                )
                baseline['devices'] = baseline['devices'][-10:]
            
            if 'email_domain' in transaction_data:
                baseline.setdefault('email_domains', []).append(
                    str(transaction_data['email_domain'])
                )
                baseline['email_domains'] = baseline['email_domains'][-5:]
            
            if 'amount' in transaction_data:
                baseline.setdefault('amounts', []).append(
                    float(transaction_data['amount'])
                )
                baseline['amounts'] = baseline['amounts'][-50:]
            
            if 'card_type' in transaction_data:
                baseline.setdefault('card_types', []).append(
                    str(transaction_data['card_type'])
                )
                baseline['card_types'] = baseline['card_types'][-10:]
            
            # Store updated baseline
            self.redis.setex(
                key,
                self.baseline_ttl,
                json.dumps(baseline)
            )
            
        except redis.RedisError as e:
            logger.error(f"Redis error updating baseline: {e}")
    
    def _check_geo_anomaly(
        self,
        transaction_data: Dict,
        baseline: Dict
    ) -> float:
        """
        Check for geographic anomalies.
        
        Returns risk score [0, 1] based on:
        - Address changes
        - Distance from usual location
        """
        risk = 0.0
        
        addr1 = str(transaction_data.get('addr1', ''))
        addr2 = str(transaction_data.get('addr2', ''))
        
        # Check if address is new
        baseline_addrs = baseline.get('addresses', [])
        
        if baseline_addrs:
            if addr1 and addr1 not in baseline_addrs:
                risk += 0.5  # New address
            
            if addr2 and addr2 not in baseline_addrs:
                risk += 0.3  # New secondary address
        
        # Check distance (if provided)
        dist1 = transaction_data.get('dist1')
        if dist1 and dist1 > self.geo_distance_threshold:
            risk += 0.7  # Very far from usual location
        elif dist1 and dist1 > 500:
            risk += 0.4  # Moderately far
        
        return min(risk, 1.0)
    
    def _check_device_anomaly(
        self,
        transaction_data: Dict,
        baseline: Dict
    ) -> float:
        """Check for device/browser anomalies."""
        risk = 0.0
        
        device_info = str(transaction_data.get('device_info', ''))
        device_type = str(transaction_data.get('device_type', ''))
        
        baseline_devices = baseline.get('devices', [])
        
        if baseline_devices and device_info:
            if device_info not in baseline_devices:
                risk += 0.6  # New device
        
        # Check for suspicious device patterns
        if device_info:
            device_lower = device_info.lower()
            # Emulators, bots, unusual devices
            if any(keyword in device_lower for keyword in [
                'emulator', 'bot', 'crawler', 'scraper', 'unknown'
            ]):
                risk += 0.8
        
        return min(risk, 1.0)
    
    def _check_email_anomaly(
        self,
        transaction_data: Dict,
        baseline: Dict
    ) -> float:
        """Check for email domain anomalies."""
        risk = 0.0
        
        email_domain = str(transaction_data.get('email_domain', ''))
        
        if not email_domain:
            return 0.0
        
        # Check if email domain is new
        baseline_emails = baseline.get('email_domains', [])
        if baseline_emails and email_domain not in baseline_emails:
            risk += 0.5  # New email domain
        
        # Check if risky domain
        if email_domain in self.risky_domains:
            risk += 0.7  # Risky disposable email
        
        # Check for suspicious patterns
        email_lower = email_domain.lower()
        if any(keyword in email_lower for keyword in [
            'temp', 'disposable', 'fake', 'trash', 'spam', '10minute'
        ]):
            risk += 0.6
        
        return min(risk, 1.0)
    
    def _check_amount_anomaly(
        self,
        transaction_data: Dict,
        baseline: Dict
    ) -> float:
        """Check for unusual transaction amounts."""
        risk = 0.0
        
        amount = transaction_data.get('amount')
        if not amount:
            return 0.0
        
        # Check if amount is very high
        if amount >= self.high_amount_threshold:
            risk += 0.7
        elif amount >= 1000:
            risk += 0.4
        
        # Check against baseline amounts
        baseline_amounts = baseline.get('amounts', [])
        if len(baseline_amounts) >= 5:
            mean_amt = np.mean(baseline_amounts)
            std_amt = np.std(baseline_amounts)
            
            # Z-score anomaly detection
            if std_amt > 0:
                z_score = abs((amount - mean_amt) / std_amt)
                if z_score > 3.0:  # 3 sigma outlier
                    risk += 0.6
                elif z_score > 2.0:  # 2 sigma outlier
                    risk += 0.3
            
            # Spike detection
            if mean_amt > 0:
                spike_ratio = amount / mean_amt
                if spike_ratio >= 5.0:
                    risk += 0.5
                elif spike_ratio >= 3.0:
                    risk += 0.3
        
        return min(risk, 1.0)
    
    def _check_card_type_anomaly(
        self,
        transaction_data: Dict,
        baseline: Dict
    ) -> float:
        """Check for unusual card type usage."""
        risk = 0.0
        
        card_type = str(transaction_data.get('card_type', ''))
        
        if not card_type:
            return 0.0
        
        baseline_card_types = baseline.get('card_types', [])
        
        if baseline_card_types and card_type not in baseline_card_types:
            risk += 0.4  # Different card type than usual
        
        return min(risk, 1.0)
    
    def _identify_risk_factors(
        self,
        geo_anomaly: float,
        device_anomaly: float,
        email_anomaly: float,
        amount_anomaly: float,
        card_anomaly: float,
        transaction_data: Dict
    ) -> List[str]:
        """Identify human-readable ATO risk factors."""
        factors = []
        
        if geo_anomaly > 0.5:
            dist = transaction_data.get('dist1', 0)
            if dist > 0:
                factors.append(f"geo_anomaly_distance_{dist:.0f}km")
            else:
                factors.append("geo_anomaly_new_address")
        
        if device_anomaly > 0.5:
            factors.append("device_anomaly_new_device")
        
        if email_anomaly > 0.5:
            email = transaction_data.get('email_domain', '')
            if email in self.risky_domains:
                factors.append(f"email_risky_domain_{email}")
            else:
                factors.append("email_anomaly_new_domain")
        
        if amount_anomaly > 0.5:
            amount = transaction_data.get('amount', 0)
            factors.append(f"amount_anomaly_{amount:.0f}")
        
        if card_anomaly > 0.3:
            factors.append("card_type_anomaly")
        
        return factors
    
    def get_baseline_stats(self, card_id: str) -> Dict:
        """
        Get current baseline statistics for a card.
        
        Returns:
            Dictionary with baseline behavior profile
        """
        baseline = self._get_baseline(card_id)
        
        if not baseline:
            return {'card_id': card_id, 'has_baseline': False}
        
        # Calculate statistics
        amounts = baseline.get('amounts', [])
        stats = {
            'card_id': card_id,
            'has_baseline': True,
            'n_addresses': len(set(baseline.get('addresses', []))),
            'n_devices': len(set(baseline.get('devices', []))),
            'n_email_domains': len(set(baseline.get('email_domains', []))),
            'n_transactions': len(amounts),
        }
        
        if amounts:
            stats['mean_amount'] = float(np.mean(amounts))
            stats['std_amount'] = float(np.std(amounts))
            stats['min_amount'] = float(min(amounts))
            stats['max_amount'] = float(max(amounts))
        
        return stats
