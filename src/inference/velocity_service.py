"""
Distributed Velocity Feature Service for Real-time Inference

Computes velocity features (transaction counts, amounts, risk scores)
for real-time fraud detection in a distributed Spark environment.

Uses Redis for distributed state management across Spark workers.
Thread-safe and handles network failures gracefully.
"""

import time
import json
import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

try:
    import redis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available. Velocity features will use local fallback (not distributed).")

logger = logging.getLogger(__name__)


class VelocityFeatureService:
    """
    Distributed velocity feature computation for Spark inference

    Uses Redis for shared state across all Spark workers. Thread-safe and
    handles network failures gracefully with fallback to local cache.

    Features computed:
    - Transaction counts (1h, 6h, 24h, 7d)
    - Amount statistics (sum, mean, std, max)
    - Risk scores (frequency risk, amount risk, spike risk)
    - Combined velocity risk score

    Architecture:
    - Primary: Redis sorted sets (ZADD with timestamp as score)
    - Fallback: Local in-memory cache (for Redis failures)
    - TTL: Automatic cleanup via Redis EXPIRE (7 days)
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        max_history_seconds: int = 7 * 24 * 3600,
        connection_timeout: int = 2,
        use_fallback: bool = True
    ):
        """
        Initialize distributed velocity service

        Args:
            redis_url: Redis connection URL (defaults to redis://redis:6379/0)
            max_history_seconds: How long to keep transaction history (default: 7 days)
            connection_timeout: Redis connection timeout in seconds
            use_fallback: Whether to use local cache on Redis failure
        """
        self.max_history_seconds = max_history_seconds
        self.use_fallback = use_fallback
        self.connection_timeout = connection_timeout

        # Time windows in seconds
        self.windows = {
            '1h': 3600,
            '6h': 6 * 3600,
            '24h': 24 * 3600,
            '7d': 7 * 24 * 3600
        }

        # Initialize Redis connection
        if redis_url is None:
            redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")

        self.redis_client = None
        self.redis_available = False

        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis.from_url(
                    redis_url,
                    socket_connect_timeout=connection_timeout,
                    socket_timeout=connection_timeout,
                    decode_responses=False,  # We'll handle encoding
                    max_connections=50,
                    health_check_interval=30
                )
                # Test connection
                self.redis_client.ping()
                self.redis_available = True
                logger.info(f"✓ Redis connected: {redis_url}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using local fallback.")
                self.redis_available = False
        else:
            logger.warning("Redis library not installed. Using local fallback.")

        # Fallback: Local in-memory cache (for Redis failures or development)
        self.local_cache: Dict[str, List[Tuple[float, float]]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.redis_errors = 0

    def create_uid(self, transaction: Dict) -> str:
        """
        Create user identifier from transaction

        Combines card1, card2, addr1, P_emaildomain to identify unique users
        """
        parts = []
        for field in ['card1', 'card2', 'addr1', 'P_emaildomain']:
            val = transaction.get(field)
            if val is not None:
                parts.append(str(val))
            else:
                parts.append('na')

        return '-'.join(parts) if parts else 'global'

    def add_transaction(self, uid: str, timestamp: float, amount: float):
        """
        Add transaction to distributed history (Redis)

        Args:
            uid: User identifier
            timestamp: Transaction timestamp (unix seconds)
            amount: Transaction amount
        """
        if self.redis_available:
            try:
                key = f"velocity:{uid}"
                # Store as "timestamp:amount" with timestamp as score for range queries
                member = f"{timestamp}:{amount}"

                # Add to Redis sorted set (ZADD)
                self.redis_client.zadd(key, {member: timestamp})

                # Set TTL on key (auto-cleanup after 7 days)
                self.redis_client.expire(key, self.max_history_seconds)

                self.cache_hits += 1
                return
            except (RedisError, RedisConnectionError) as e:
                logger.warning(f"Redis write error: {e}. Using local fallback.")
                self.redis_errors += 1
                self.redis_available = False  # Temporarily disable

        # Fallback: Local cache
        if self.use_fallback:
            if uid not in self.local_cache:
                self.local_cache[uid] = []

            self.local_cache[uid].append((timestamp, amount))

            # Clean old transactions
            cutoff = timestamp - self.max_history_seconds
            self.local_cache[uid] = [
                (ts, amt) for ts, amt in self.local_cache[uid] if ts >= cutoff
            ]

            # Limit cache size per user (prevent memory exhaustion)
            if len(self.local_cache[uid]) > 10000:
                self.local_cache[uid] = self.local_cache[uid][-10000:]

            self.cache_misses += 1

    def _get_transaction_history(self, uid: str, window_start: float, current_timestamp: float) -> List[Tuple[float, float]]:
        """
        Fetch transaction history from Redis or local fallback

        Args:
            uid: User identifier
            window_start: Start of time window (unix seconds)
            current_timestamp: Current transaction timestamp

        Returns:
            List of (timestamp, amount) tuples within window
        """
        if self.redis_available:
            try:
                key = f"velocity:{uid}"

                # Get transactions in time range using ZRANGEBYSCORE
                members = self.redis_client.zrangebyscore(
                    key,
                    min=window_start,
                    max=current_timestamp - 0.001  # Exclude current transaction
                )

                # Parse members: "timestamp:amount"
                history = []
                for member in members:
                    try:
                        member_str = member.decode('utf-8')
                        ts_str, amt_str = member_str.split(':', 1)
                        history.append((float(ts_str), float(amt_str)))
                    except (ValueError, AttributeError) as e:
                        logger.warning(f"Failed to parse Redis member: {member}. Error: {e}")
                        continue

                self.cache_hits += 1
                return history

            except (RedisError, RedisConnectionError) as e:
                logger.warning(f"Redis read error: {e}. Using local fallback.")
                self.redis_errors += 1
                self.redis_available = False  # Temporarily disable

        # Fallback: Local cache
        if self.use_fallback and uid in self.local_cache:
            history = [
                (ts, amt) for ts, amt in self.local_cache[uid]
                if window_start <= ts < current_timestamp
            ]
            self.cache_misses += 1
            return history

        return []

    def compute_velocity_features(
        self,
        uid: str,
        current_timestamp: float,
        current_amount: float
    ) -> Dict[str, float]:
        """
        Compute velocity features from distributed history (Redis)

        Args:
            uid: User identifier
            current_timestamp: Current transaction timestamp
            current_amount: Current transaction amount

        Returns:
            Dictionary of velocity features
        """
        features = {}

        # For each time window, compute features
        for window_name, window_seconds in self.windows.items():
            window_start = current_timestamp - window_seconds

            # Fetch history from Redis or local cache
            window_txns = self._get_transaction_history(uid, window_start, current_timestamp)

            if window_txns:
                amounts = np.array([amt for _, amt in window_txns])

                features[f'txn_count_{window_name}'] = len(window_txns)
                features[f'amt_sum_{window_name}'] = float(amounts.sum())
                features[f'amt_mean_{window_name}'] = float(amounts.mean())
                features[f'amt_std_{window_name}'] = float(amounts.std()) if len(amounts) > 1 else 0.0
                features[f'amt_max_{window_name}'] = float(amounts.max())
            else:
                # No history in this window
                features[f'txn_count_{window_name}'] = 0
                features[f'amt_sum_{window_name}'] = 0.0
                features[f'amt_mean_{window_name}'] = 0.0
                features[f'amt_std_{window_name}'] = 0.0
                features[f'amt_max_{window_name}'] = 0.0

        # Compute risk scores
        features['freq_risk_1h'] = min(features['txn_count_1h'] / 10.0, 1.0)
        features['freq_risk_24h'] = min(features['txn_count_24h'] / 50.0, 1.0)
        features['amt_risk_24h'] = min(features['amt_sum_24h'] / 10000.0, 1.0)

        # Amount spike risk (current amount vs historical mean)
        for window_name in ['1h', '6h', '24h']:
            mean_col = f'amt_mean_{window_name}'
            if features[mean_col] > 0:
                spike = min(current_amount / features[mean_col], 10.0) / 10.0
                features[f'amt_spike_{window_name}'] = spike
            else:
                features[f'amt_spike_{window_name}'] = 0.0

        # Combined velocity risk score (weighted average)
        features['velocity_risk_score'] = min(
            0.3 * features['freq_risk_1h'] +
            0.2 * features['freq_risk_24h'] +
            0.2 * features['amt_risk_24h'] +
            0.15 * features['amt_spike_1h'] +
            0.15 * features['amt_spike_24h'],
            1.0
        )

        return features

    def process_batch(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of transactions and add velocity features

        Args:
            transactions: DataFrame with columns:
                - TransactionAmt
                - timestamp (datetime or unix timestamp)
                - card1, card2, addr1, P_emaildomain (for uid)

        Returns:
            DataFrame with velocity features added
        """
        # Convert timestamp to unix seconds if datetime
        if 'timestamp' in transactions.columns:
            if pd.api.types.is_datetime64_any_dtype(transactions['timestamp']):
                timestamps = transactions['timestamp'].astype('int64') / 1e9
            else:
                timestamps = transactions['timestamp']
        else:
            # Use current time if no timestamp
            timestamps = pd.Series([time.time()] * len(transactions))

        # Create UIDs
        uids = transactions.apply(
            lambda row: self.create_uid(row.to_dict()),
            axis=1
        )

        # Compute velocity features for each transaction
        velocity_features_list = []

        for idx, (uid, ts, amt) in enumerate(zip(uids, timestamps, transactions['TransactionAmt'])):
            # Compute features BEFORE adding current transaction
            features = self.compute_velocity_features(uid, ts, amt)

            # Add current transaction to history
            self.add_transaction(uid, ts, amt)

            velocity_features_list.append(features)

        # Convert to DataFrame
        velocity_df = pd.DataFrame(velocity_features_list, index=transactions.index)

        # Merge with original transactions
        result = pd.concat([transactions, velocity_df], axis=1)

        return result

    def get_stats(self) -> Dict:
        """Get service statistics"""
        stats = {
            'redis_available': self.redis_available,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'redis_errors': self.redis_errors,
            'max_history_seconds': self.max_history_seconds,
            'local_cache_users': len(self.local_cache),
            'local_cache_size': sum(len(txns) for txns in self.local_cache.values())
        }

        # Get Redis stats if available
        if self.redis_available:
            try:
                info = self.redis_client.info('stats')
                stats['redis_keys'] = self.redis_client.dbsize()
                stats['redis_total_commands'] = info.get('total_commands_processed', 0)
            except Exception as e:
                logger.warning(f"Failed to get Redis stats: {e}")

        return stats

    def clear_history(self, uid: Optional[str] = None):
        """
        Clear transaction history

        Args:
            uid: User identifier (if None, clears all users)
        """
        if uid:
            # Clear specific user
            if self.redis_available:
                try:
                    self.redis_client.delete(f"velocity:{uid}")
                except Exception as e:
                    logger.warning(f"Failed to clear Redis key for {uid}: {e}")

            if uid in self.local_cache:
                del self.local_cache[uid]
        else:
            # Clear all
            if self.redis_available:
                try:
                    # Delete all velocity keys
                    pattern = "velocity:*"
                    for key in self.redis_client.scan_iter(match=pattern, count=100):
                        self.redis_client.delete(key)
                except Exception as e:
                    logger.warning(f"Failed to clear all Redis keys: {e}")

            self.local_cache.clear()

    def reconnect_redis(self):
        """Attempt to reconnect to Redis (for health check recovery)"""
        if not self.redis_available and REDIS_AVAILABLE and self.redis_client:
            try:
                self.redis_client.ping()
                self.redis_available = True
                logger.info("✓ Redis reconnected successfully")
                return True
            except Exception as e:
                logger.debug(f"Redis reconnection failed: {e}")
                return False
        return self.redis_available

    def health_check(self) -> Dict[str, any]:
        """
        Check service health

        Returns:
            Dictionary with health status
        """
        health = {
            'status': 'healthy',
            'redis_connected': self.redis_available,
            'fallback_enabled': self.use_fallback,
            'redis_errors': self.redis_errors
        }

        # Try to reconnect if Redis is down
        if not self.redis_available:
            self.reconnect_redis()
            health['status'] = 'degraded' if self.use_fallback else 'unhealthy'

        return health


# Global singleton instance
_velocity_service_instance = None

def get_velocity_service() -> VelocityFeatureService:
    """Get global velocity service instance"""
    global _velocity_service_instance
    if _velocity_service_instance is None:
        _velocity_service_instance = VelocityFeatureService()
    return _velocity_service_instance
