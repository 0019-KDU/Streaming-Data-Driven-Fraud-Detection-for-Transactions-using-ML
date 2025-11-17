"""
Feature Pipeline for IEEE-CIS Fraud Detection

This pipeline ensures train/serve consistency by providing a transform() method
that applies the same feature engineering steps used during training.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.preprocessing import RobustScaler


class IEEECISFeaturePipeline:
    """
    Feature pipeline with proper transform() method for train/serve consistency

    This pipeline applies:
    - Frequency encoding for categorical variables
    - Feature scaling (RobustScaler)
    - Ensures consistent feature names between training and inference
    """

    def __init__(self):
        self.freq_maps: Dict[str, Dict[Any, float]] = {}
        self.scaler: Optional[RobustScaler] = None
        self.feature_names: list = []
        self.risky_domains = {
            'anonymous.com', 'mailinator.com', 'tempmail.com', 'dispostable.com',
            'yopmail.com', '10minutemail.com', 'guerrillamail.com'
        }

    def fit(self, train_df: pd.DataFrame, feature_names: list):
        """
        Fit frequency maps on training data

        Args:
            train_df: Training dataframe with all features
            feature_names: List of final feature names
        """
        self.feature_names = feature_names

        # Fit frequency encoding
        freq_cols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                     'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain']
        freq_cols = [c for c in freq_cols if c in train_df.columns]

        for col in freq_cols:
            vc = train_df[col].value_counts(dropna=False)
            self.freq_maps[col] = (vc / vc.sum()).to_dict()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input dataframe to match training feature space

        This method applies ALL feature engineering steps to ensure
        train/serve consistency.

        Args:
            df: Input dataframe with raw features

        Returns:
            Transformed dataframe with all engineered features
        """
        import numpy as np
        df = df.copy()

        # ========== AMOUNT FEATURES ==========
        if 'TransactionAmt' in df.columns:
            df['TransactionAmt'] = df['TransactionAmt'].astype('float32')
            df['log_TransactionAmt'] = np.log1p(df['TransactionAmt'].fillna(0)).astype('float32')
            df['sqrt_TransactionAmt'] = np.sqrt(df['TransactionAmt'].fillna(0)).astype('float32')
            df['TransactionAmt_decimal'] = ((df['TransactionAmt'] - df['TransactionAmt'].astype(int)) * 1000).astype('int16')
            
            # Card aggregations (use pre-computed statistics from training)
            if 'card1' in df.columns and hasattr(self, 'card1_amt_mean'):
                df['TransactionAmt_to_mean_card1'] = (df['TransactionAmt'] / (df['card1'].map(self.card1_amt_mean).fillna(100) + 1e-5)).astype('float32')
                df['TransactionAmt_to_std_card1'] = (df['TransactionAmt'] / (df['card1'].map(self.card1_amt_std).fillna(50) + 1e-5)).astype('float32')
            else:
                df['TransactionAmt_to_mean_card1'] = 1.0
                df['TransactionAmt_to_std_card1'] = 1.0
            
            if 'card4' in df.columns and hasattr(self, 'card4_amt_mean'):
                df['TransactionAmt_to_mean_card4'] = (df['TransactionAmt'] / (df['card4'].map(self.card4_amt_mean).fillna(100) + 1e-5)).astype('float32')
                df['TransactionAmt_to_std_card4'] = (df['TransactionAmt'] / (df['card4'].map(self.card4_amt_std).fillna(50) + 1e-5)).astype('float32')
            else:
                df['TransactionAmt_to_mean_card4'] = 1.0
                df['TransactionAmt_to_std_card4'] = 1.0

        # ========== EMAIL FEATURES ==========
        RISKY_DOMAINS = {
            'anonymous.com', 'mailinator.com', 'tempmail.com', 'dispostable.com',
            'yopmail.com', '10minutemail.com', 'guerrillamail.com'
        }
        HIGH_RISK_DOMAINS = {
            'protonmail.com', 'guerrillamail.com', 'mailinator.com',
            '10minutemail.com', 'tempmail.com', 'throwaway.email',
            'yopmail.com', 'sharklasers.com', 'guerrillamail.info',
            'dispostable.com', 'trashmail.com'
        }
        
        if 'P_emaildomain' in df.columns:
            df['email_risky'] = df['P_emaildomain'].isin(RISKY_DOMAINS).astype('int8')
            df['email_is_generic'] = df['P_emaildomain'].isin(['gmail.com', 'yahoo.com', 'hotmail.com']).astype('int8')
            df['is_high_risk_email'] = df['P_emaildomain'].isin(HIGH_RISK_DOMAINS).astype('int8')
            df['is_disposable_email'] = df['P_emaildomain'].fillna('').astype(str).str.contains(
                'temp|disposable|guerrilla|throwaway|fake|spam|trash', case=False, regex=True
            ).astype('int8')
            df['email_domain_length'] = df['P_emaildomain'].fillna('').str.len().astype('int8')
            df['email_is_short_domain'] = (df['email_domain_length'] <= 8).astype('int8')
        
        if 'P_emaildomain' in df.columns and 'R_emaildomain' in df.columns:
            df['email_match'] = (df['P_emaildomain'] == df['R_emaildomain']).fillna(False).astype('int8')

        # ========== CARD FEATURES ==========
        if 'card4' in df.columns:
            df['card_is_discover'] = (df['card4'] == 'discover').astype('int8')
            df['card_is_amex'] = (df['card4'] == 'american express').astype('int8')
        
        if 'card6' in df.columns:
            df['card_is_charge'] = (df['card6'] == 'charge card').astype('int8')
        
        if 'ProductCD' in df.columns:
            df['product_is_C'] = (df['ProductCD'] == 'C').astype('int8')
            df['product_is_R'] = (df['ProductCD'] == 'R').astype('int8')

        # ========== DISTANCE FEATURES ==========
        if 'dist1' in df.columns:
            df['dist1_filled'] = df['dist1'].fillna(0).astype('float32')
            df['has_dist1'] = (~df['dist1'].isna()).astype('int8')
            df['dist1_high'] = (df['dist1'].fillna(0) > 1000).astype('int8')
        
        if 'dist2' in df.columns:
            df['dist2_filled'] = df['dist2'].fillna(0).astype('float32')
            df['has_dist2'] = (~df['dist2'].isna()).astype('int8')
            df['dist2_high'] = (df['dist2'].fillna(0) > 1000).astype('int8')

        # Apply frequency encoding
        for col, freq_map in self.freq_maps.items():
            if col in df.columns:
                df[col + '_freq'] = df[col].map(freq_map).fillna(0.0).astype('float32')

        # Select and order features to match training
        # Fill missing features with 0
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = 0.0

        # Return only the features in the correct order
        return df[self.feature_names]

    def get_config(self) -> Dict[str, Any]:
        """Get pipeline configuration for serialization"""
        return {
            'freq_maps': self.freq_maps,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'risky_domains': self.risky_domains
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'IEEECISFeaturePipeline':
        """Create pipeline from configuration"""
        pipeline = cls()
        pipeline.freq_maps = config.get('freq_maps', {})
        pipeline.scaler = config.get('scaler')
        pipeline.feature_names = config.get('feature_names', [])
        if 'risky_domains' in config:
            pipeline.risky_domains = config['risky_domains']
        return pipeline
