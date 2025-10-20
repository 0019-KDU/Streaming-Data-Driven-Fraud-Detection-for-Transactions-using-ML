"""
Lightweight Velocity Feature Service for Real-time Inference

Computes velocity features (transaction counts, amounts, risk scores)
for real-time fraud detection without needing full historical data.

Uses Redis/in-memory cache to track recent transactions per user.
"""

import time
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import threading


class VelocityFeatureService:
    """
    In-memory velocity feature computation for inference

    Maintains sliding windows of recent transactions per user (uid)
    and computes velocity features on-the-fly.

    Features computed:
    - Transaction counts (1h, 6h, 24h, 7d)
    - Amount statistics (sum, mean, std, max)
    - Risk scores (frequency risk, amount risk, spike risk)
    - Combined velocity risk score

    Note: This is NOT thread-safe for Spark broadcasting.
    Each worker will have its own instance.
    """

    def __init__(self, max_history_seconds: int = 7 * 24 * 3600):
        """
        Initialize velocity service

        Args:
            max_history_seconds: How long to keep transaction history (default: 7 days)
        """
        self.max_history_seconds = max_history_seconds

        # Storage: uid -> [(timestamp, amount), ...]
        self.user_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))

        # Time windows in seconds
        self.windows = {
            '1h': 3600,
            '6h': 6 * 3600,
            '24h': 24 * 3600,
            '7d': 7 * 24 * 3600
        }

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
        Add transaction to history

        Args:
            uid: User identifier
            timestamp: Transaction timestamp (unix seconds)
            amount: Transaction amount
        """
        # No lock needed - each Spark worker has its own instance
        self.user_history[uid].append((timestamp, amount))

        # Clean old transactions (beyond max_history_seconds)
        cutoff = timestamp - self.max_history_seconds
        while self.user_history[uid] and self.user_history[uid][0][0] < cutoff:
            self.user_history[uid].popleft()

    def compute_velocity_features(
        self,
        uid: str,
        current_timestamp: float,
        current_amount: float
    ) -> Dict[str, float]:
        """
        Compute velocity features for a single transaction

        Args:
            uid: User identifier
            current_timestamp: Current transaction timestamp
            current_amount: Current transaction amount

        Returns:
            Dictionary of velocity features
        """
        features = {}

        # No lock needed - each Spark worker has its own instance
        history = list(self.user_history.get(uid, []))

        # For each time window, compute features
        for window_name, window_seconds in self.windows.items():
            window_start = current_timestamp - window_seconds

            # Filter transactions within window (BEFORE current transaction)
            window_txns = [(ts, amt) for ts, amt in history if window_start <= ts < current_timestamp]

            if window_txns:
                amounts = np.array([amt for _, amt in window_txns])

                features[f'txn_count_{window_name}'] = len(window_txns)
                features[f'amt_sum_{window_name}'] = amounts.sum()
                features[f'amt_mean_{window_name}'] = amounts.mean()
                features[f'amt_std_{window_name}'] = amounts.std() if len(amounts) > 1 else 0.0
                features[f'amt_max_{window_name}'] = amounts.max()
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
        total_users = len(self.user_history)
        total_transactions = sum(len(hist) for hist in self.user_history.values())

        return {
            'total_users': total_users,
            'total_transactions': total_transactions,
            'max_history_seconds': self.max_history_seconds
        }

    def clear_history(self):
        """Clear all history (for testing)"""
        self.user_history.clear()


# Global singleton instance
_velocity_service_instance = None

def get_velocity_service() -> VelocityFeatureService:
    """Get global velocity service instance"""
    global _velocity_service_instance
    if _velocity_service_instance is None:
        _velocity_service_instance = VelocityFeatureService()
    return _velocity_service_instance
