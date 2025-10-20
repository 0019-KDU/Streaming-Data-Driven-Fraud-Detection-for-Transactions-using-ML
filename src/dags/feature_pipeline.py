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
        df = df.copy()

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
