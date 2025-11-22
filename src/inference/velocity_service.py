"""
Velocity tracking service using Redis.

Tracks transaction patterns over time windows (1h, 6h, 24h, 7d) and
detects abnormal velocity and amount spikes.
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import redis

from .utils.redis_client import get_redis_client
from .logging_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class VelocityResult:
    """Result from velocity analysis."""
    velocity_risk: float  # [0, 1]
    amount_risk: float    # [0, 1]
    factors: List[str]    # Risk factor descriptions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class VelocityService:
    """
    Tracks transaction velocity and amount patterns using Redis.

    For each (card1, uid) combination, maintains:
    - Transaction counts in time windows: 1h, 6h, 24h, 7d
    - Amount totals and statistics
    - Historical baselines for anomaly detection
    """

    def __init__(self, config):
        """
        Initialize velocity service.

        Args:
            config: Config object with velocity and Redis settings
        """
        self.config = config
        self.redis_client = get_redis_client(config)
        self.ttl = config.redis.velocity_ttl

    def _get_key(self, entity_id: str, window: str) -> str:
        """Generate Redis key for velocity tracking."""
        return f"velocity:{entity_id}:{window}"

    def _get_baseline_key(self, entity_id: str) -> str:
        """Generate Redis key for baseline statistics."""
        return f"velocity:baseline:{entity_id}"

    def record_transaction(
        self,
        card1: str,
        uid: str,
        amount: float,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Record a transaction for velocity tracking.

        Args:
            card1: Card identifier
            uid: User identifier (card1_addr1_email)
            amount: Transaction amount
            timestamp: Unix timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = time.time()

        # Track for both card1 and uid
        for entity_id in [f"card1:{card1}", f"uid:{uid}"]:
            transaction_data = {
                'timestamp': timestamp,
                'amount': amount
            }

            # Add to time-series lists for each window
            for window, seconds in [('1h', 3600), ('6h', 21600), ('24h', 86400), ('7d', 604800)]:
                key = self._get_key(entity_id, window)

                # Add transaction with score = timestamp
                self.redis_client.zadd(
                    key,
                    {json.dumps(transaction_data): timestamp}
                )

                # Remove old transactions outside window
                cutoff = timestamp - seconds
                self.redis_client.zremrangebyscore(key, '-inf', cutoff)

                # Set TTL
                self.redis_client.expire(key, self.ttl)

            # Update baseline statistics
            self._update_baseline(entity_id, amount)

    def _update_baseline(self, entity_id: str, amount: float) -> None:
        """Update running baseline statistics using Welford's algorithm."""
        baseline_key = self._get_baseline_key(entity_id)

        # Get current baseline
        baseline_data = self.redis_client.get(baseline_key)
        if baseline_data:
            baseline = json.loads(baseline_data)
            count = baseline['count']
            mean = baseline['mean']
            m2 = baseline['m2']
        else:
            count = 0
            mean = 0.0
            m2 = 0.0

        # Update using Welford's online algorithm
        count += 1
        delta = amount - mean
        mean += delta / count
        delta2 = amount - mean
        m2 += delta * delta2

        # Store updated baseline
        baseline = {
            'count': count,
            'mean': mean,
            'm2': m2,
            'std': (m2 / count) ** 0.5 if count > 1 else 0.0
        }
        self.redis_client.setex(
            baseline_key,
            self.ttl,
            json.dumps(baseline)
        )

    def get_velocity_stats(
        self,
        card1: str,
        uid: str,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get velocity statistics for entity."""
        if timestamp is None:
            timestamp = time.time()

        stats = {}

        for entity_id in [f"card1:{card1}", f"uid:{uid}"]:
            entity_stats = {}

            for window in ['1h', '6h', '24h', '7d']:
                key = self._get_key(entity_id, window)

                # Get all transactions in window
                transactions = self.redis_client.zrangebyscore(
                    key, '-inf', timestamp, withscores=False
                )

                count = len(transactions)
                total_amount = 0.0
                amounts = []

                for txn_json in transactions:
                    txn = json.loads(txn_json)
                    amounts.append(txn['amount'])
                    total_amount += txn['amount']

                mean_amount = total_amount / count if count > 0 else 0.0
                max_amount = max(amounts) if amounts else 0.0

                entity_stats[window] = {
                    'count': count,
                    'total_amount': total_amount,
                    'mean_amount': mean_amount,
                    'max_amount': max_amount
                }

            # Get baseline
            baseline_key = self._get_baseline_key(entity_id)
            baseline_data = self.redis_client.get(baseline_key)
            if baseline_data:
                entity_stats['baseline'] = json.loads(baseline_data)
            else:
                entity_stats['baseline'] = {'count': 0, 'mean': 0.0, 'std': 0.0}

            stats[entity_id] = entity_stats

        return stats

    def analyze_velocity(
        self,
        card1: str,
        uid: str,
        current_amount: float,
        timestamp: Optional[float] = None
    ) -> VelocityResult:
        """
        Analyze velocity and detect anomalies.

        Args:
            card1: Card identifier
            uid: User identifier
            current_amount: Current transaction amount
            timestamp: Current timestamp

        Returns:
            VelocityResult with risk scores and factors
        """
        stats = self.get_velocity_stats(card1, uid, timestamp)

        velocity_risk = 0.0
        amount_risk = 0.0
        factors = []

        # Analyze card1 velocity
        card_stats = stats.get(f"card1:{card1}", {})

        # Check 1-hour transaction count
        count_1h = card_stats.get('1h', {}).get('count', 0)
        if count_1h >= self.config.velocity.high_1h_count:
            velocity_risk = max(velocity_risk, 0.8)
            factors.append(f"high_1h_txn_count:{count_1h}")

        # Check 6-hour transaction count
        count_6h = card_stats.get('6h', {}).get('count', 0)
        if count_6h >= self.config.velocity.high_6h_count:
            velocity_risk = max(velocity_risk, 0.6)
            factors.append(f"high_6h_txn_count:{count_6h}")

        # Check 24-hour transaction count
        count_24h = card_stats.get('24h', {}).get('count', 0)
        if count_24h >= self.config.velocity.high_24h_count:
            velocity_risk = max(velocity_risk, 0.5)
            factors.append(f"high_24h_txn_count:{count_24h}")

        # Check amount spike vs baseline
        baseline = card_stats.get('baseline', {})
        baseline_mean = baseline.get('mean', 0.0)
        baseline_std = baseline.get('std', 0.0)

        if baseline_mean > 0:
            amount_ratio = current_amount / baseline_mean

            if amount_ratio >= self.config.velocity.amount_spike_5x:
                amount_risk = max(amount_risk, 0.9)
                factors.append(f"amount_spike_5x:{amount_ratio:.1f}")
            elif amount_ratio >= self.config.velocity.amount_spike_3x:
                amount_risk = max(amount_risk, 0.6)
                factors.append(f"amount_spike_3x:{amount_ratio:.1f}")

            # Check if amount is outside 3 standard deviations
            if baseline_std > 0 and current_amount > (baseline_mean + 3 * baseline_std):
                amount_risk = max(amount_risk, 0.7)
                factors.append("amount_outlier_3std")

        # UID-based velocity (more granular)
        uid_stats = stats.get(f"uid:{uid}", {})
        uid_count_1h = uid_stats.get('1h', {}).get('count', 0)

        if uid_count_1h >= 3:
            velocity_risk = max(velocity_risk, 0.7)
            factors.append(f"uid_burst:{uid_count_1h}")

        return VelocityResult(
            velocity_risk=min(1.0, velocity_risk),
            amount_risk=min(1.0, amount_risk),
            factors=factors
        )
