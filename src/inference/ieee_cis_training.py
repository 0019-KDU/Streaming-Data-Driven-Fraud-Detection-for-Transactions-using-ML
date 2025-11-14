"""
IEEE-CIS Fraud Detection Training Module - FULL ML TRAINING

- This module includes advanced features from research notebook (VAE removed):
    - Velocity Features (optimized time-windows: 1h, 6h, 24h, 7d)
    - Adaptive Threshold System with dynamic adjustment
    - Frequency Encoding for categorical variables
    - 60+ Advanced Features for maximum performance
    - GPU support for training acceleration
    - MLflow experiment tracking and model registry

"""

import os
import gc
import logging
import pickle
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from collections import deque

import yaml
import joblib
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

# Import feature pipeline
import sys
sys.path.append(os.path.dirname(__file__))
from feature_pipeline import IEEECISFeaturePipeline

# Boosting models
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

# VAE ensemble removed: TensorFlow/VAE code is no longer used in the pipeline
TF_AVAILABLE = False
logging.info("VAE ensemble removed from training pipeline.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
RNG = 42
RISKY_DOMAINS = {
    'anonymous.com', 'mailinator.com', 'tempmail.com', 'dispostable.com',
    'yopmail.com', '10minutemail.com', 'guerrillamail.com'
}


# VAE ensemble has been removed from the pipeline. No VAE architecture is defined.


# ==================== ADAPTIVE THRESHOLD SYSTEM ====================
class AdaptiveThresholdSystem:
    """
    Adaptive threshold that adjusts based on recent fraud rates
    and velocity risk patterns
    """
    def __init__(self, base_threshold=0.5, window_size=1000,
                 min_threshold=0.1, max_threshold=0.9):
        self.base_threshold = base_threshold
        self.window_size = window_size
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.recent_predictions = deque(maxlen=window_size)
        self.recent_true_labels = deque(maxlen=window_size)
        self.current_threshold = base_threshold
        self.threshold_history = []

    def update(self, y_true, y_pred_proba, velocity_risk):
        """Update threshold based on recent performance"""
        self.recent_predictions.append(y_pred_proba)
        self.recent_true_labels.append(y_true)

        if len(self.recent_predictions) < 100:
            return self.current_threshold

        # Calculate recent fraud rate
        recent_fraud_rate = np.mean(self.recent_true_labels)

        # Calculate prediction accuracy
        recent_preds_binary = np.array(self.recent_predictions) >= self.current_threshold
        recent_accuracy = np.mean(recent_preds_binary == np.array(self.recent_true_labels))

        # Adjust threshold
        if recent_fraud_rate > 0.05:  # High fraud period
            adjustment = -0.02
        elif recent_fraud_rate < 0.02:  # Low fraud period
            adjustment = 0.02
        else:
            adjustment = 0

        # Additional adjustment based on velocity risk
        if velocity_risk > 0.7:
            adjustment -= 0.01

        # Update threshold with constraints
        self.current_threshold = np.clip(
            self.current_threshold + adjustment,
            self.min_threshold,
            self.max_threshold
        )

        self.threshold_history.append({
            'threshold': self.current_threshold,
            'fraud_rate': recent_fraud_rate,
            'accuracy': recent_accuracy
        })

        return self.current_threshold

    def get_threshold(self, velocity_risk=0.0):
        """Get current threshold, adjusted for velocity risk"""
        adjusted = self.current_threshold

        # Lower threshold for high-risk velocity patterns
        if velocity_risk > 0.8:
            adjusted *= 0.9
        elif velocity_risk > 0.6:
            adjusted *= 0.95

        return np.clip(adjusted, self.min_threshold, self.max_threshold)

    def get_hybrid_threshold(self, velocity_risk, amount_risk=0.0, 
                            method='weighted', w1=0.6, w2=0.3, w3=0.1):
        """
        Hybrid threshold combining F1-optimal with risk-based adjustments
        
        Industry best practice: τ_hybrid = w₁·τ_A + w₂·τ_V + w₃·τ_Amount
        
        Args:
            velocity_risk: 0-1 velocity risk score (high velocity = lower threshold)
            amount_risk: 0-1 amount risk score (high amount = lower threshold)
            method: Combination strategy
                - 'weighted': Fixed weighted average (default)
                - 'dynamic': Adaptive weights based on risk confidence
                - 'max': Most conservative (highest threshold)
                - 'min': Most aggressive (lowest threshold)
            w1, w2, w3: Weights for adaptive, velocity, amount (sum to 1.0)
        
        Returns:
            float: Hybrid threshold bounded by min/max thresholds
        
        Examples:
            Low risk:  τ_hybrid = 0.42 (near F1-optimal)
            High velocity: τ_hybrid = 0.32 (stricter detection)
            High amount: τ_hybrid = 0.35 (stricter detection)
        """
        # Component thresholds
        tau_A = self.current_threshold  # F1-optimal base (statistical)
        tau_V = self.base_threshold - (velocity_risk * 0.15)  # Velocity-adjusted
        tau_Amount = self.base_threshold - (amount_risk * 0.10)  # Amount-adjusted
        
        if method == 'max':
            # Most conservative: Take highest threshold (hardest to flag fraud)
            # Use when: False positives are very costly
            tau = max(tau_A, tau_V, tau_Amount)
            
        elif method == 'min':
            # Most aggressive: Take lowest threshold (easiest to flag fraud)
            # Use when: False negatives are very costly
            tau = min(tau_A, tau_V, tau_Amount)
            
        elif method == 'dynamic':
            # Dynamic weights based on signal confidence
            # High risk signals get more weight
            if velocity_risk > 0.7:
                # Trust velocity signal more
                w1, w2, w3 = 0.3, 0.5, 0.2
            elif amount_risk > 0.7:
                # Trust amount signal more
                w1, w2, w3 = 0.3, 0.2, 0.5
            else:
                # Trust F1-optimal statistical threshold
                w1, w2, w3 = 0.6, 0.2, 0.2
            
            tau = w1 * tau_A + w2 * tau_V + w3 * tau_Amount
            
        else:  # 'weighted' (default)
            # Fixed weighted average
            tau = w1 * tau_A + w2 * tau_V + w3 * tau_Amount
        
        # Apply safety bounds
        return np.clip(tau, self.min_threshold, self.max_threshold)


# ==================== MAIN TRAINING CLASS ====================
class IEEECISFraudTraining:
    """
    IEEE-CIS Fraud Detection Training Pipeline - FULL VERSION

    Includes ALL advanced features:
    - VAE Ensemble (3 models)
    - Velocity Features (optimized time-windows)
    - Adaptive Threshold System
    - Frequency Encoding
    - 60+ Advanced Features
    """

    def __init__(self, config_path: str = "/app/config.yaml"):
        """Initialize training pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.model = None
        self.vae_models = []
        self.scaler = None
        self.freq_maps = {}
        self.feature_pipeline = IEEECISFeaturePipeline()
        self.adaptive_threshold_system = None
        self.best_threshold = 0.5
        self.all_features = []

        # Set random seeds
        np.random.seed(RNG)
        if TF_AVAILABLE:
            tf.random.set_seed(RNG)

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def load_and_merge_data(self) -> pd.DataFrame:
        """Load and merge IEEE-CIS datasets"""
        logger.info("Loading IEEE-CIS datasets...")

        trans_path = self.config["data"]["ieee_cis"]["train_transaction_path"]
        df_trans = pd.read_csv(trans_path)
        logger.info(f"Loaded {len(df_trans):,} transactions")

        identity_path = self.config["data"]["ieee_cis"]["train_identity_path"]
        if os.path.exists(identity_path):
            df_identity = pd.read_csv(identity_path)
            logger.info(f"Loaded {len(df_identity):,} identity records")
            df = df_trans.merge(df_identity, on='TransactionID', how='left')
            logger.info(f"Merged dataset: {len(df):,} rows")
        else:
            logger.warning(f"Identity file not found: {identity_path}")
            df = df_trans

        # Sort by time
        df = df.sort_values('TransactionDT').reset_index(drop=True)

        logger.info(f"Fraud rate: {df['isFraud'].mean():.4%}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        return df

    def basic_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features and normalize data"""
        logger.info("Creating basic features...")
        df = df.copy()

        # Normalize text columns
        for col in ['ProductCD', 'DeviceInfo', 'id_31', 'P_emaildomain', 'R_emaildomain']:
            if col in df.columns:
                df[col] = df[col].astype('string').str.strip().str.lower()

        # Ensure numeric types
        numeric_cols = ['TransactionAmt', 'TransactionDT', 'addr1', 'addr2',
                        'card1', 'card2', 'card3', 'card5']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle email unknowns
        for col in ['P_emaildomain', 'R_emaildomain']:
            if col in df.columns:
                df[col] = df[col].replace({'email_not_provided': pd.NA, 'unknown': pd.NA})

        # Target
        df['isFraud'] = df['isFraud'].astype('int8')

        return df

    def create_uid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create user identifier from card and address"""
        logger.info("Creating user identifiers...")
        df = df.copy()

        uid_parts = []
        for col in ['card1', 'card2', 'addr1', 'P_emaildomain']:
            if col in df.columns:
                uid_parts.append(df[col].astype('string').fillna('na'))

        if uid_parts:
            df['uid'] = uid_parts[0]
            for p in uid_parts[1:]:
                df['uid'] = df['uid'].astype('string') + '-' + p.astype('string')
        else:
            df['uid'] = 'global'

        return df

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from TransactionDT"""
        logger.info("Creating time features...")
        df = df.copy()

        sec = df['TransactionDT'].astype('float64')
        df['dt_day'] = (sec // (24*60*60)).astype('int32')
        df['dt_hour'] = (sec // 3600 % 24).astype('int16')
        df['dt_wday'] = (df['dt_day'] % 7).astype('int8')
        df['dt_is_weekend'] = (df['dt_wday'] >= 5).astype('int8')
        df['dt_is_night'] = ((df['dt_hour'] >= 22) | (df['dt_hour'] <= 6)).astype('int8')

        return df

    def create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create transaction amount features"""
        logger.info("Creating amount features...")
        df = df.copy()

        if 'TransactionAmt' in df.columns:
            df['TransactionAmt'] = df['TransactionAmt'].astype('float32')
            df['log_TransactionAmt'] = np.log1p(df['TransactionAmt'].fillna(0)).astype('float32')
            df['sqrt_TransactionAmt'] = np.sqrt(df['TransactionAmt'].fillna(0)).astype('float32')

        return df

    def create_email_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create email-based features"""
        logger.info("Creating email features...")
        df = df.copy()

        # Email match
        if 'P_emaildomain' in df.columns and 'R_emaildomain' in df.columns:
            df['email_match'] = (df['P_emaildomain'] == df['R_emaildomain']).fillna(False).astype('int8')

        # Risky domains
        if 'P_emaildomain' in df.columns:
            df['email_risky'] = df['P_emaildomain'].isin(RISKY_DOMAINS).astype('int8')
            df['email_is_generic'] = df['P_emaildomain'].isin(
                ['gmail.com', 'yahoo.com', 'hotmail.com']
            ).astype('int8')

        return df

    def calculate_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate enhanced velocity features with optimized time-windows

        Uses binary search for 100x+ speedup vs nested loops
        """
        logger.info("Calculating velocity features (optimized)...")
        logger.info(f"Processing {len(df):,} transactions across {df['uid'].nunique():,} users...")

        start_time = time.time()

        df = df.sort_values(['uid', 'TransactionDT']).reset_index(drop=True).copy()

        # Windows: (seconds, suffix)
        windows = [(3600, '1h'), (6*3600, '6h'), (24*3600, '24h'), (7*24*3600, '7d')]

        # Initialize all columns
        for window, name in windows:
            df[f'txn_count_{name}'] = 0
            df[f'amt_sum_{name}'] = 0.0
            df[f'amt_mean_{name}'] = 0.0
            df[f'amt_std_{name}'] = 0.0
            df[f'amt_max_{name}'] = 0.0

        # Process each window
        for window_sec, window_name in windows:
            logger.info(f"  Processing {window_name} window...")

            grouped = df.groupby('uid')

            results = []
            for uid, group in grouped:
                if len(group) < 2:
                    result = pd.DataFrame({
                        'idx': group.index,
                        f'txn_count_{window_name}': 0,
                        f'amt_sum_{window_name}': 0.0,
                        f'amt_mean_{window_name}': 0.0,
                        f'amt_std_{window_name}': 0.0,
                        f'amt_max_{window_name}': 0.0
                    })
                    results.append(result)
                    continue

                times = group['TransactionDT'].values
                amts = group['TransactionAmt'].fillna(0).values

                counts = np.zeros(len(times), dtype=np.int32)
                sums = np.zeros(len(times), dtype=np.float32)
                means = np.zeros(len(times), dtype=np.float32)
                stds = np.zeros(len(times), dtype=np.float32)
                maxs = np.zeros(len(times), dtype=np.float32)

                # Vectorized calculation using searchsorted
                for i in range(len(times)):
                    current_time = times[i]
                    window_start = current_time - window_sec

                    start_idx = np.searchsorted(times[:i], window_start, side='left')

                    if start_idx < i:
                        window_amts = amts[start_idx:i]
                        counts[i] = len(window_amts)
                        sums[i] = window_amts.sum()
                        means[i] = window_amts.mean()
                        stds[i] = window_amts.std() if len(window_amts) > 1 else 0.0
                        maxs[i] = window_amts.max()

                result = pd.DataFrame({
                    'idx': group.index,
                    f'txn_count_{window_name}': counts,
                    f'amt_sum_{window_name}': sums,
                    f'amt_mean_{window_name}': means,
                    f'amt_std_{window_name}': stds,
                    f'amt_max_{window_name}': maxs
                })
                results.append(result)

            if results:
                all_results = pd.concat(results, ignore_index=False)
                all_results = all_results.set_index('idx').sort_index()

                for col in all_results.columns:
                    df.loc[all_results.index, col] = all_results[col].values

        # Calculate risk scores (vectorized)
        logger.info("  Calculating risk scores...")
        df['freq_risk_1h'] = np.clip(df['txn_count_1h'] / 10.0, 0, 1)
        df['freq_risk_24h'] = np.clip(df['txn_count_24h'] / 50.0, 0, 1)
        df['amt_risk_24h'] = np.clip(df['amt_sum_24h'] / 10000.0, 0, 1)

        # Amount spike risk
        for name in ['1h', '6h', '24h']:
            mean_col = f'amt_mean_{name}'
            df[f'amt_spike_{name}'] = 0.0
            mask = df[mean_col] > 0
            if mask.any():
                df.loc[mask, f'amt_spike_{name}'] = (
                    df.loc[mask, 'TransactionAmt'] / df.loc[mask, mean_col]
                ).clip(0, 10) / 10.0

        # Combined velocity risk score
        df['velocity_risk_score'] = (
            0.3 * df['freq_risk_1h'] +
            0.2 * df['freq_risk_24h'] +
            0.2 * df['amt_risk_24h'] +
            0.15 * df['amt_spike_1h'] +
            0.15 * df['amt_spike_24h']
        ).clip(0, 1)

        elapsed = time.time() - start_time
        logger.info(f"  Velocity features complete in {elapsed:.1f}s ({elapsed/60:.2f} min)")
        logger.info(f"  Velocity risk score range: [{df['velocity_risk_score'].min():.4f}, {df['velocity_risk_score'].max():.4f}]")

        return df

    def apply_frequency_encoding(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply frequency encoding to categorical variables

        Fit on train, apply to both train and valid
        """
        logger.info("Applying frequency encoding...")

        freq_cols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                     'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain']
        freq_cols = [c for c in freq_cols if c in train_df.columns]

        # Fit on train
        self.freq_maps = {}
        for col in freq_cols:
            vc = train_df[col].value_counts(dropna=False)
            self.freq_maps[col] = (vc / vc.sum()).to_dict()

            # Apply to train
            train_df[col + '_freq'] = train_df[col].map(self.freq_maps[col]).fillna(0.0).astype('float32')

            # Apply to valid
            valid_df[col + '_freq'] = valid_df[col].map(self.freq_maps[col]).fillna(0.0).astype('float32')

        logger.info(f"  Frequency encoded {len(freq_cols)} columns")

        # Drop high-cardinality string columns
        drop_cols = []
        from pandas.api.types import is_string_dtype, is_object_dtype
        for col in freq_cols:
            if col in train_df.columns and (is_string_dtype(train_df[col]) or is_object_dtype(train_df[col])):
                drop_cols.append(col)

        train_df = train_df.drop(columns=drop_cols, errors='ignore')
        valid_df = valid_df.drop(columns=drop_cols, errors='ignore')

        logger.info(f"  Dropped {len(drop_cols)} high-cardinality columns")

        return train_df, valid_df

    def select_all_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Select comprehensive feature set (60+ features)
        """
        # Base features
        base_features = [
            'TransactionAmt', 'log_TransactionAmt', 'sqrt_TransactionAmt',
            'dt_day', 'dt_hour', 'dt_wday', 'dt_is_weekend', 'dt_is_night',
            'email_match', 'email_risky', 'email_is_generic'
        ]

        # Velocity features
        velocity_features = [c for c in df.columns if any(x in c for x in
            ['txn_count_', 'amt_sum_', 'amt_mean_', 'amt_std_', 'amt_max_',
             'freq_risk_', 'amt_risk_', 'amt_spike_', 'velocity_risk_score'])]

        # Frequency encoded features
        freq_features = [c for c in df.columns if c.endswith('_freq')]

        # Combine all
        self.all_features = base_features + velocity_features + freq_features
        self.all_features = [f for f in self.all_features if f in df.columns]

        logger.info(f"Total features: {len(self.all_features)}")
        logger.info(f"  Base: {len(base_features)}")
        logger.info(f"  Velocity: {len(velocity_features)}")
        logger.info(f"  Frequency: {len(freq_features)}")

        X = df[self.all_features].fillna(0).copy()
        y = df['isFraud'].copy() if 'isFraud' in df.columns else None

        return X, y

    # VAE ensemble was removed; no train_vae_ensemble implementation

    def train_boosting_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series
    ) -> Any:
        """Train gradient boosting model (XGBoost/LightGBM/CatBoost)"""
        logger.info("Training gradient boosting models...")

        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos_weight = float(neg) / float(pos)
        logger.info(f"  Class imbalance ratio: {scale_pos_weight:.2f}")

        # Check if CPU-only mode is enabled
        force_cpu = self.config.get("training", {}).get("force_cpu_only", False)
        if force_cpu:
            logger.info("  CPU-only mode enabled (skipping GPU attempts)")

        models_to_try = []

        # XGBoost with GPU support
        if XGBClassifier is not None:
            if not force_cpu:
                try:
                    logger.info("  Attempting XGBoost with GPU...")
                    xgb_model = XGBClassifier(
                        n_estimators=2000,
                        max_depth=6,
                        learning_rate=0.03,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.1,
                        reg_lambda=1.5,
                        gamma=0.1,
                        scale_pos_weight=scale_pos_weight,
                        eval_metric="aucpr",
                        tree_method="hist",
                        device="cuda",
                        random_state=RNG
                    )
                    models_to_try.append(('XGBoost-GPU', xgb_model))
                except:
                    logger.info("  XGBoost GPU failed, using CPU...")
                    xgb_model = XGBClassifier(
                        n_estimators=2000,
                        max_depth=6,
                        learning_rate=0.03,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.1,
                        reg_lambda=1.5,
                        gamma=0.1,
                        scale_pos_weight=scale_pos_weight,
                        eval_metric="aucpr",
                        tree_method="hist",
                        random_state=RNG,
                        n_jobs=-1
                    )
                    models_to_try.append(('XGBoost', xgb_model))
            else:
                logger.info("  Using XGBoost with CPU...")
                xgb_model = XGBClassifier(
                    n_estimators=2000,
                    max_depth=6,
                    learning_rate=0.03,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.1,
                    reg_lambda=1.5,
                    gamma=0.1,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="aucpr",
                    tree_method="hist",
                    random_state=RNG,
                    n_jobs=-1
                )
                models_to_try.append(('XGBoost', xgb_model))

        # LightGBM with GPU support
        if LGBMClassifier is not None:
            if not force_cpu:
                try:
                    logger.info("  Attempting LightGBM with GPU...")
                    lgbm_model = LGBMClassifier(
                        n_estimators=5000,
                        num_leaves=63,
                        learning_rate=0.03,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.1,
                        reg_lambda=1.5,
                        min_child_samples=50,
                        objective="binary",
                        device="gpu",
                        gpu_platform_id=0,
                        gpu_device_id=0,
                        random_state=RNG,
                        scale_pos_weight=scale_pos_weight,
                        verbose=-1
                    )
                    models_to_try.append(('LightGBM-GPU', lgbm_model))
                except:
                    logger.info("  LightGBM GPU failed, using CPU...")
                    lgbm_model = LGBMClassifier(
                        n_estimators=5000,
                        num_leaves=63,
                        learning_rate=0.03,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.1,
                        reg_lambda=1.5,
                        min_child_samples=50,
                        objective="binary",
                        random_state=RNG,
                        n_jobs=-1,
                        scale_pos_weight=scale_pos_weight,
                        verbose=-1
                    )
                    models_to_try.append(('LightGBM', lgbm_model))
            else:
                logger.info("  Using LightGBM with CPU...")
                lgbm_model = LGBMClassifier(
                    n_estimators=5000,
                    num_leaves=63,
                    learning_rate=0.03,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.1,
                    reg_lambda=1.5,
                    min_child_samples=50,
                    objective="binary",
                    random_state=RNG,
                    n_jobs=-1,
                    scale_pos_weight=scale_pos_weight,
                    verbose=-1
                )
                models_to_try.append(('LightGBM', lgbm_model))

        # CatBoost with GPU support
        if CatBoostClassifier is not None:
            if not force_cpu:
                try:
                    logger.info("  Attempting CatBoost with GPU...")
                    cat_model = CatBoostClassifier(
                        iterations=3000,
                        depth=6,
                        learning_rate=0.03,
                        l2_leaf_reg=3.0,
                        random_state=RNG,
                        class_weights=[1.0, scale_pos_weight],
                        loss_function="Logloss",
                        task_type="GPU",
                        devices="0",
                        verbose=False
                    )
                    models_to_try.append(('CatBoost-GPU', cat_model))
                except:
                    logger.info("  CatBoost GPU failed, using CPU...")
                    cat_model = CatBoostClassifier(
                        iterations=3000,
                        depth=6,
                        learning_rate=0.03,
                        l2_leaf_reg=3.0,
                        random_state=RNG,
                        class_weights=[1.0, scale_pos_weight],
                        loss_function="Logloss",
                        verbose=False
                    )
                    models_to_try.append(('CatBoost', cat_model))
            else:
                logger.info("  Using CatBoost with CPU...")
                cat_model = CatBoostClassifier(
                    iterations=3000,
                    depth=6,
                    learning_rate=0.03,
                    l2_leaf_reg=3.0,
                    random_state=RNG,
                    class_weights=[1.0, scale_pos_weight],
                    loss_function="Logloss",
                    verbose=False
                )
                models_to_try.append(('CatBoost', cat_model))

        # Train and evaluate
        best_model = None
        best_score = 0.0
        best_name = None

        for name, model in models_to_try:
            try:
                logger.info(f"  Training {name}...")

                if 'XGBoost' in name or 'LightGBM' in name:
                    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
                else:
                    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)

                y_valid_proba = model.predict_proba(X_valid)[:, 1]
                auc_pr = average_precision_score(y_valid, y_valid_proba)
                auc_roc = roc_auc_score(y_valid, y_valid_proba)

                logger.info(f"    {name} - AUC-PR: {auc_pr:.4f}, AUC-ROC: {auc_roc:.4f}")

                if auc_pr > best_score:
                    best_score = auc_pr
                    best_model = model
                    best_name = name

            except Exception as e:
                logger.warning(f"  Failed to train {name}: {str(e)}")
                continue

        if best_model is None:
            raise RuntimeError("All model training attempts failed")

        logger.info(f"  Best model: {best_name} (AUC-PR: {best_score:.4f})")
        return best_model

    def calibrate_model(
        self,
        model: Any,
        X_valid: pd.DataFrame,
        y_valid: pd.Series
    ) -> CalibratedClassifierCV:
        """Calibrate model probabilities"""
        logger.info("Calibrating model probabilities...")

        calibrated_model = CalibratedClassifierCV(
            model,
            method="sigmoid",
            cv="prefit"
        )

        calibrated_model.fit(X_valid, y_valid)

        y_valid_proba_cal = calibrated_model.predict_proba(X_valid)[:, 1]
        auc_pr_cal = average_precision_score(y_valid, y_valid_proba_cal)
        auc_roc_cal = roc_auc_score(y_valid, y_valid_proba_cal)

        logger.info(f"  Calibrated - AUC-PR: {auc_pr_cal:.4f}, AUC-ROC: {auc_roc_cal:.4f}")

        return calibrated_model

    def initialize_adaptive_threshold(
        self,
        y_valid: pd.Series,
        y_valid_proba: np.ndarray
    ) -> float:
        """Initialize adaptive threshold system with F1-optimal threshold"""
        logger.info("Initializing adaptive threshold system...")

        prec, rec, thr = precision_recall_curve(y_valid, y_valid_proba)
        f1 = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
        best_idx = np.nanargmax(f1)
        f1_optimal_threshold = float(thr[best_idx])

        self.adaptive_threshold_system = AdaptiveThresholdSystem(
            base_threshold=f1_optimal_threshold,
            window_size=1000,
            min_threshold=0.05,
            max_threshold=0.95
        )

        logger.info(f"  Base threshold (F1-optimal): {f1_optimal_threshold:.4f}")
        logger.info(f"  Precision: {prec[best_idx]:.4f}")
        logger.info(f"  Recall: {rec[best_idx]:.4f}")
        logger.info(f"  F1-Score: {f1[best_idx]:.4f}")

        return f1_optimal_threshold

    def save_artifacts(self, model: Any):
        """Save all model artifacts including VAE and adaptive threshold"""
        logger.info("Saving model artifacts...")

        model_path = self.config["training"]["model_path"]
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)

        # Save main artifact bundle (VAE removed)
        artifact_bundle = {
            'calibrated_model': model,
            'n_vae_models': 0,
            'scaler': self.scaler,
            'freq_maps': self.freq_maps,
            'feature_names': self.all_features,
            'adaptive_threshold_system': self.adaptive_threshold_system
        }

        joblib.dump(artifact_bundle, model_path)
        logger.info(f"  Model bundle saved to: {model_path}")

        # Update the feature pipeline and save it as a proper object with transform() method
        self.feature_pipeline.freq_maps = self.freq_maps
        self.feature_pipeline.scaler = self.scaler
        self.feature_pipeline.feature_names = self.all_features

        # Save feature pipeline separately for inference compatibility
        pipeline_path = self.config["training"]["feature_pipeline_path"]
        joblib.dump(self.feature_pipeline, pipeline_path)
        logger.info(f"  Feature pipeline (with transform() method) saved to: {pipeline_path}")

    def log_to_mlflow(
        self,
        model: Any,
        y_valid: pd.Series,
        y_valid_proba: np.ndarray,
        threshold: float
    ):
        """Log experiment to MLflow"""
        experiment_name = self.config.get("training", {}).get(
            "experiment_name", "ieee_cis_fraud_detection"
        )

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("model_type", "EnhancedEnsemble")
            mlflow.log_param("n_features", len(self.all_features))
            mlflow.log_param("n_vae_models", 0)
            mlflow.log_param("base_threshold", threshold)
            mlflow.log_param("velocity_features", "enabled")
            mlflow.log_param("adaptive_threshold", "enabled")
            mlflow.log_param("frequency_encoding", "enabled")

            # Log metrics
            y_valid_pred = (y_valid_proba >= threshold).astype(int)

            mlflow.log_metric("auc_pr", average_precision_score(y_valid, y_valid_proba))
            mlflow.log_metric("auc_roc", roc_auc_score(y_valid, y_valid_proba))
            mlflow.log_metric("precision", precision_score(y_valid, y_valid_pred))
            mlflow.log_metric("recall", recall_score(y_valid, y_valid_pred))
            mlflow.log_metric("f1_score", f1_score(y_valid, y_valid_pred))

            # Log confusion matrix
            cm = confusion_matrix(y_valid, y_valid_pred)
            mlflow.log_metric("true_positives", int(cm[1, 1]))
            mlflow.log_metric("false_positives", int(cm[0, 1]))
            mlflow.log_metric("true_negatives", int(cm[0, 0]))
            mlflow.log_metric("false_negatives", int(cm[1, 0]))

            # Register model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=self.config["mlflow"]["registered_model_name"]
            )

            logger.info(f"  Experiment logged to MLflow: {experiment_name}")

    def train_pipeline(self) -> Dict[str, Any]:
        """Execute full training pipeline with all advanced features"""
        logger.info("="*80)
        logger.info("IEEE-CIS Fraud Detection Training Pipeline - FULL ML TRAINING")
        logger.info("="*80)

        # Load data
        df = self.load_and_merge_data()

        # Basic cleaning
        df = self.basic_feature_engineering(df)

        # Create UID
        df = self.create_uid(df)

        # Create all feature groups
        df = self.create_time_features(df)
        df = self.create_amount_features(df)
        df = self.create_email_features(df)

        # Velocity features (time-intensive)
        df = self.calculate_velocity_features(df)

        # Chronological split BEFORE frequency encoding to prevent leakage
        split_ratio = self.config["data"]["ieee_cis"]["chronological_split_ratio"]
        split_idx = int(len(df) * split_ratio)
        train_df = df.iloc[:split_idx].copy()
        valid_df = df.iloc[split_idx:].copy()

        logger.info(f"Chronological split:")
        logger.info(f"  Train: {len(train_df):,} ({train_df['isFraud'].mean():.4%} fraud)")
        logger.info(f"  Valid: {len(valid_df):,} ({valid_df['isFraud'].mean():.4%} fraud)")

        # Frequency encoding
        train_df, valid_df = self.apply_frequency_encoding(train_df, valid_df)

        # Select features
        X_train, y_train = self.select_all_features(train_df)
        X_valid, y_valid = self.select_all_features(valid_df)

        logger.info(f"Final feature set:")
        logger.info(f"  X_train: {X_train.shape}")
        logger.info(f"  X_valid: {X_valid.shape}")

        # VAE ensemble removed: no anomaly score added

        # Train gradient boosting
        model = self.train_boosting_model(X_train, y_train, X_valid, y_valid)

        # Calibrate
        calibrated_model = self.calibrate_model(model, X_valid, y_valid)
        self.model = calibrated_model

        # Get predictions
        y_valid_proba = calibrated_model.predict_proba(X_valid)[:, 1]

        # Initialize adaptive threshold
        self.best_threshold = self.initialize_adaptive_threshold(y_valid, y_valid_proba)

        # Save artifacts
        self.save_artifacts(calibrated_model)

        # Log to MLflow (optional - don't fail if MLflow unavailable)
        try:
            self.log_to_mlflow(calibrated_model, y_valid, y_valid_proba, self.best_threshold)
        except Exception as e:
            logger.warning(f"  MLflow logging failed (non-critical): {str(e)}")
            logger.warning("  Continuing with training... Model artifacts already saved.")

        # Calculate final metrics
        y_valid_pred = (y_valid_proba >= self.best_threshold).astype(int)

        metrics = {
            "status": "success",
            "model_type": "enhanced_ensemble",
            "auc_pr": average_precision_score(y_valid, y_valid_proba),
            "auc_roc": roc_auc_score(y_valid, y_valid_proba),
            "precision": precision_score(y_valid, y_valid_pred),
            "recall": recall_score(y_valid, y_valid_pred),
            "f1_score": f1_score(y_valid, y_valid_pred),
            "threshold": self.best_threshold,
            "n_features": len(self.all_features),
            "n_vae_models": 0,
            "train_samples": len(y_train),
            "valid_samples": len(y_valid)
        }

        logger.info("="*80)
        logger.info("Training Complete!")
        logger.info(f"  AUC-PR: {metrics['auc_pr']:.4f}")
        logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"  Total Features: {metrics['n_features']}")
        logger.info(f"  VAE Models: {metrics['n_vae_models']}")
        logger.info("="*80)

        # Cleanup
        gc.collect()

        return metrics


def train_ieee_cis_model(config_path: str = "/app/config.yaml") -> Dict[str, Any]:
    """Main entry point for IEEE-CIS training"""
    trainer = IEEECISFraudTraining(config_path)
    return trainer.train_pipeline()


if __name__ == "__main__":
    metrics = train_ieee_cis_model()
    print(f"\nTraining completed with status: {metrics['status']}")
    print(f"Final AUC-PR: {metrics['auc_pr']:.4f}")
    print(f"Final F1-Score: {metrics['f1_score']:.4f}")
    print(f"Total Features: {metrics['n_features']}")
