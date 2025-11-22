"""
Velocity detection service using Redis for real-time transaction tracking.

Tracks transaction patterns across multiple time windows (1h, 6h, 24h, 7d)
to detect high-velocity fraud attacks.

Key Features:
- Multi-window tracking (1h, 6h, 24h, 7d)
- Transaction count and amount aggregations
- Velocity risk scoring
- Amount spike detection
"""

import json
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import redis

from .logging_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class VelocityResult:
    """Result from velocity check."""
    velocity_risk: float  # 0-1 risk score based on transaction frequency
    amount_risk: float    # 0-1 risk score based on amount patterns
    factors: List[str]    # Human-readable risk factors
    
    # Detailed metrics
    txn_count_1h: int = 0
    txn_count_6h: int = 0
    txn_count_24h: int = 0
    txn_count_7d: int = 0
    
    amt_sum_1h: float = 0.0
    amt_sum_24h: float = 0.0
    amt_mean_24h: float = 0.0


class VelocityService:
    """
    Redis-backed velocity tracking service.
    
    Tracks transaction patterns for each card/user combination across
    multiple time windows to detect velocity-based fraud attacks.
    """
    
    def __init__(self, redis_client: redis.Redis, config):
        """
        Initialize velocity service.
        
        Args:
            redis_client: Redis client instance
            config: Config object with velocity settings
        """
        self.redis = redis_client
        self.config = config
        
        # ✅ FIX #3: Add environment/version namespacing
        import os
        self.env = os.getenv('ENVIRONMENT', 'prod')
        self.model_version = os.getenv('MODEL_VERSION', 'v1')
        self.key_prefix = f"{self.env}:{self.model_version}"
        
        # Time windows (in seconds)
        self.windows = {
            '1h': 3600,
            '6h': 6 * 3600,
            '24h': 24 * 3600,
            '7d': 7 * 24 * 3600
        }
        
        # Thresholds from config
        self.high_1h_count = config.velocity.high_1h_count
        self.high_6h_count = config.velocity.high_6h_count
        self.high_24h_count = config.velocity.high_24h_count
        self.spike_3x = config.velocity.amount_spike_3x
        self.spike_5x = config.velocity.amount_spike_5x
        
        # TTL for Redis keys (7 days)
        self.ttl = config.redis.velocity_ttl
        
        logger.info(f"VelocityService initialized with namespace: {self.key_prefix}")
        logger.info(f"VelocityService windows: {list(self.windows.keys())}")
    
    def check_velocity(
        self,
        card_id: str,
        user_id: str,
        amount: float,
        timestamp: float = None
    ) -> VelocityResult:
        """
        Check velocity risk for a transaction.
        
        Args:
            card_id: Card identifier (card1)
            user_id: User identifier (UID)
            amount: Transaction amount
            timestamp: Unix timestamp (default: current time)
            
        Returns:
            VelocityResult with risk scores and factors
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Get transaction counts for all windows
        counts = self._get_txn_counts(card_id, user_id, timestamp)
        
        # Get amount statistics
        amt_sum_1h, amt_sum_24h, amt_mean_24h = self._get_amount_stats(
            card_id, user_id, timestamp
        )
        
        # Calculate velocity risk
        velocity_risk = self._calculate_velocity_risk(counts)
        
        # Calculate amount risk
        amount_risk = self._calculate_amount_risk(
            amount, amt_mean_24h, amt_sum_24h
        )
        
        # Identify risk factors
        factors = self._identify_risk_factors(
            counts, amount, amt_mean_24h
        )
        
        # Record this transaction
        self._record_transaction(card_id, user_id, amount, timestamp)
        
        return VelocityResult(
            velocity_risk=velocity_risk,
            amount_risk=amount_risk,
            factors=factors,
            txn_count_1h=counts['1h'],
            txn_count_6h=counts['6h'],
            txn_count_24h=counts['24h'],
            txn_count_7d=counts['7d'],
            amt_sum_1h=amt_sum_1h,
            amt_sum_24h=amt_sum_24h,
            amt_mean_24h=amt_mean_24h
        )
    
    def _get_txn_counts(
        self,
        card_id: str,
        user_id: str,
        timestamp: float
    ) -> Dict[str, int]:
        """Get transaction counts for all time windows."""
        counts = {}
        
        for window_name, window_sec in self.windows.items():
            key = f"velocity:{card_id}:{user_id}:txns"
            
            # Get timestamps in window
            min_time = timestamp - window_sec
            
            try:
                # Count transactions in time range using sorted set
                count = self.redis.zcount(key, min_time, timestamp)
                counts[window_name] = count
            except redis.RedisError as e:
                logger.warning(f"Redis error counting transactions: {e}")
                counts[window_name] = 0
        
        return counts
    
    def _get_amount_stats(
        self,
        card_id: str,
        user_id: str,
        timestamp: float
    ) -> Tuple[float, float, float]:
        """Get amount statistics for time windows."""
        # 1h window
        key_1h = f"velocity:{card_id}:{user_id}:amounts:1h"
        min_time_1h = timestamp - self.windows['1h']
        
        try:
            amounts_1h = self.redis.zrangebyscore(
                key_1h, min_time_1h, timestamp, withscores=False
            )
            if amounts_1h:
                amounts_1h = [float(amt) for amt in amounts_1h]
                amt_sum_1h = sum(amounts_1h)
            else:
                amt_sum_1h = 0.0
        except redis.RedisError as e:
            logger.warning(f"Redis error getting 1h amounts: {e}")
            amt_sum_1h = 0.0
        
        # 24h window
        key_24h = f"velocity:{card_id}:{user_id}:amounts:24h"
        min_time_24h = timestamp - self.windows['24h']
        
        try:
            amounts_24h = self.redis.zrangebyscore(
                key_24h, min_time_24h, timestamp, withscores=False
            )
            if amounts_24h:
                amounts_24h = [float(amt) for amt in amounts_24h]
                amt_sum_24h = sum(amounts_24h)
                amt_mean_24h = amt_sum_24h / len(amounts_24h)
            else:
                amt_sum_24h = 0.0
                amt_mean_24h = 0.0
        except redis.RedisError as e:
            logger.warning(f"Redis error getting 24h amounts: {e}")
            amt_sum_24h = 0.0
            amt_mean_24h = 0.0
        
        return amt_sum_1h, amt_sum_24h, amt_mean_24h
    
    def _calculate_velocity_risk(self, counts: Dict[str, int]) -> float:
        """
        Calculate velocity risk score [0, 1].
        
        High transaction frequency indicates potential card testing or fraud attack.
        """
        risk = 0.0
        
        # 1h window (highest weight)
        if counts['1h'] >= 10:
            risk += 0.9
        elif counts['1h'] >= self.high_1h_count:
            risk += 0.7
        elif counts['1h'] >= 3:
            risk += 0.3
        
        # 6h window
        if counts['6h'] >= 30:
            risk += 0.5
        elif counts['6h'] >= self.high_6h_count:
            risk += 0.3
        
        # 24h window
        if counts['24h'] >= 50:
            risk += 0.4
        elif counts['24h'] >= self.high_24h_count:
            risk += 0.2
        
        # Cap at 1.0
        return min(risk, 1.0)
    
    def _calculate_amount_risk(
        self,
        current_amount: float,
        mean_amount: float,
        sum_24h: float
    ) -> float:
        """
        Calculate amount-based risk score [0, 1].
        
        Detects amount spikes and high cumulative spending.
        """
        risk = 0.0
        
        # Amount spike detection
        if mean_amount > 0:
            spike_ratio = current_amount / mean_amount
            
            if spike_ratio >= self.spike_5x:
                risk += 0.8
            elif spike_ratio >= self.spike_3x:
                risk += 0.5
            elif spike_ratio >= 2.0:
                risk += 0.3
        
        # High cumulative spending in 24h
        if sum_24h >= 10000:
            risk += 0.7
        elif sum_24h >= 5000:
            risk += 0.4
        elif sum_24h >= 2000:
            risk += 0.2
        
        # High single transaction
        if current_amount >= 5000:
            risk += 0.5
        elif current_amount >= 2000:
            risk += 0.3
        
        return min(risk, 1.0)
    
    def _identify_risk_factors(
        self,
        counts: Dict[str, int],
        amount: float,
        mean_amount: float
    ) -> List[str]:
        """Identify human-readable risk factors."""
        factors = []
        
        # High velocity
        if counts['1h'] >= self.high_1h_count:
            factors.append(f"high_1h_txn_count_{counts['1h']}")
        
        if counts['6h'] >= self.high_6h_count:
            factors.append(f"high_6h_txn_count_{counts['6h']}")
        
        if counts['24h'] >= self.high_24h_count:
            factors.append(f"high_24h_txn_count_{counts['24h']}")
        
        # Amount spikes
        if mean_amount > 0:
            spike_ratio = amount / mean_amount
            if spike_ratio >= self.spike_5x:
                factors.append(f"amount_spike_5x_{spike_ratio:.1f}x")
            elif spike_ratio >= self.spike_3x:
                factors.append(f"amount_spike_3x_{spike_ratio:.1f}x")
        
        # High amount
        if amount >= 5000:
            factors.append(f"high_amount_{amount:.0f}")
        
        return factors
    
    def _record_transaction(
        self,
        card_id: str,
        user_id: str,
        amount: float,
        timestamp: float
    ) -> None:
        """Record transaction in Redis for future velocity checks."""
        try:
            # Record transaction timestamp
            txn_key = f"velocity:{card_id}:{user_id}:txns"
            self.redis.zadd(txn_key, {str(timestamp): timestamp})
            self.redis.expire(txn_key, self.ttl)
            
            # Record amounts for 1h and 24h windows
            amt_key_1h = f"velocity:{card_id}:{user_id}:amounts:1h"
            self.redis.zadd(amt_key_1h, {str(amount): timestamp})
            self.redis.expire(amt_key_1h, 7200)  # 2 hours TTL
            
            amt_key_24h = f"velocity:{card_id}:{user_id}:amounts:24h"
            self.redis.zadd(amt_key_24h, {str(amount): timestamp})
            self.redis.expire(amt_key_24h, 86400 * 2)  # 2 days TTL
            
            # Clean old entries (beyond 7d window)
            min_time_7d = timestamp - self.windows['7d']
            self.redis.zremrangebyscore(txn_key, 0, min_time_7d)
            
        except redis.RedisError as e:
            logger.error(f"Redis error recording transaction: {e}")
    
    def get_velocity_stats(
        self,
        card_id: str,
        user_id: str,
        timestamp: float = None
    ) -> Dict:
        """
        Get current velocity statistics for a card/user.
        
        Returns:
            Dictionary with current counts and amounts
        """
        if timestamp is None:
            timestamp = time.time()
        
        counts = self._get_txn_counts(card_id, user_id, timestamp)
        amt_sum_1h, amt_sum_24h, amt_mean_24h = self._get_amount_stats(
            card_id, user_id, timestamp
        )
        
        return {
            'card_id': card_id,
            'user_id': user_id,
            'txn_counts': counts,
            'amt_sum_1h': amt_sum_1h,
            'amt_sum_24h': amt_sum_24h,
            'amt_mean_24h': amt_mean_24h,
            'timestamp': timestamp
        }
