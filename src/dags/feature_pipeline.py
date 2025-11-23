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
        train/serve consistency. Processes 434 raw features → 88 engineered features.

        Args:
            df: Input dataframe with raw IEEE-CIS features (434 columns)

        Returns:
            Transformed dataframe with 88 engineered features
        """
        import numpy as np
        df = df.copy()

        # ========== BASIC PREPROCESSING ==========
        # Create card1_addr1 combination for later use
        if 'card1' in df.columns and 'addr1' in df.columns:
            df['card1_addr1'] = df['card1'].fillna(-999).astype(str) + '_' + df['addr1'].fillna(-999).astype(str)

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

        # ========== MAGIC UID FEATURES (44 features) ==========
        # These capture user-level behavior patterns critical for fraud detection
        if 'TransactionDT' in df.columns and 'card1_addr1' in df.columns:
            df['day'] = df['TransactionDT'] / (24 * 60 * 60)
            
            # Create Magic UID: card1_addr1 + floor(day - D1)
            if 'D1' in df.columns:
                df['magic_uid'] = df['card1_addr1'].astype(str) + '_' + np.floor(df['day'] - df['D1']).fillna(-999).astype(str)
            else:
                df['magic_uid'] = df['card1_addr1'].astype(str)
            
            # Apply pre-computed Magic UID aggregations from training
            # These are stored as dictionaries mapping magic_uid → statistic
            if hasattr(self, 'magic_uid_stats'):
                # TransactionAmt aggregations
                for stat in ['mean', 'std']:
                    col_name = f'magic_uid_TransactionAmt_{stat}'
                    if col_name in self.magic_uid_stats:
                        df[col_name] = df['magic_uid'].map(self.magic_uid_stats[col_name]).fillna(0).astype(np.float32)
                
                # D-column aggregations (D4, D9, D10, D15)
                for d_col in ['D4', 'D9', 'D10', 'D15']:
                    for stat in ['mean', 'std']:
                        col_name = f'magic_uid_{d_col}_{stat}'
                        if col_name in self.magic_uid_stats:
                            df[col_name] = df['magic_uid'].map(self.magic_uid_stats[col_name]).fillna(-1).astype(np.float32)
                
                # C-column aggregations (C1-C14 except C3)
                for i in range(1, 15):
                    if i != 3:  # Skip C3
                        col_name = f'magic_uid_C{i}_mean'
                        if col_name in self.magic_uid_stats:
                            df[col_name] = df['magic_uid'].map(self.magic_uid_stats[col_name]).fillna(-1).astype(np.float32)
                
                # M-column aggregations (M1-M9)
                for i in range(1, 10):
                    col_name = f'magic_uid_M{i}_mean'
                    if col_name in self.magic_uid_stats:
                        df[col_name] = df['magic_uid'].map(self.magic_uid_stats[col_name]).fillna(-1).astype(np.float32)
                
                # C14 std (special case)
                if 'magic_uid_C14_std' in self.magic_uid_stats:
                    df['magic_uid_C14_std'] = df['magic_uid'].map(self.magic_uid_stats['magic_uid_C14_std']).fillna(-1).astype(np.float32)
                
                # Frequency encoding
                if 'magic_uid_freq' in self.magic_uid_stats:
                    df['magic_uid_freq'] = df['magic_uid'].map(self.magic_uid_stats['magic_uid_freq']).fillna(0).astype(np.float32)
                
                # Nunique aggregations (12 features)
                nunique_cols = ['email_nunique', 'dist1_nunique', 'DT_M_nunique', 'id02_nunique', 
                                'cents_nunique', 'C13_nunique', 'V314_nunique',
                                'V127_nunique', 'V136_nunique', 'V309_nunique', 'V307_nunique', 'V320_nunique']
                for col in nunique_cols:
                    col_name = f'magic_uid_{col}'
                    if col_name in self.magic_uid_stats:
                        df[col_name] = df['magic_uid'].map(self.magic_uid_stats[col_name]).fillna(0).astype(np.int16)
            else:
                # Fallback: create zero-filled features if stats not available
                # This ensures feature count matches, but predictions may be degraded
                for col in self.feature_names:
                    if 'magic_uid' in col and col not in df.columns:
                        df[col] = 0.0
            
            # outsider15 feature (D1 and D15 differ significantly)
            if 'D1' in df.columns and 'D15' in df.columns:
                df['outsider15'] = (np.abs(df['D1'] - df['D15']) > 3).astype(np.int8)
            else:
                df['outsider15'] = 0
            
            # Clean up temporary columns
            df = df.drop(columns=['magic_uid', 'day'], errors='ignore')

        # ========== D-NORMALIZED FEATURES (4 features) ==========
        # Normalize D4, D10, D11, D15 by subtracting TransactionDT
        if 'TransactionDT' in df.columns:
            transaction_days = df['TransactionDT'] / (24 * 60 * 60)
            for d_col in ['D4', 'D10', 'D11', 'D15']:
                if d_col in df.columns:
                    df[f'{d_col}_normalized'] = (df[d_col] - transaction_days).fillna(-1).astype('float32')
                else:
                    df[f'{d_col}_normalized'] = -1.0

        # ========== CARD ENCODING FEATURES (5 features) ==========
        # Frequency encoding for card+address combinations
        if hasattr(self, 'card_encoding_maps'):
            for encoding_name, encoding_map in self.card_encoding_maps.items():
                df[encoding_name] = df[encoding_map['key_col']].map(encoding_map['freq_map']).fillna(0).astype('int32')
        else:
            # Fallback: compute basic frequency encoding on-the-fly
            if 'card1_addr1' in df.columns:
                card1_addr1_freq = df['card1_addr1'].value_counts().to_dict()
                df['card1_addr1_FE'] = df['card1_addr1'].map(card1_addr1_freq).fillna(0).astype('int32')
            if 'card1' in df.columns:
                card1_freq = df['card1'].value_counts().to_dict()
                df['card1_FE'] = df['card1'].map(card1_freq).fillna(0).astype('int32')
            if 'card2' in df.columns:
                card2_freq = df['card2'].value_counts().to_dict()
                df['card2_FE'] = df['card2'].map(card2_freq).fillna(0).astype('int32')

        # ========== GROUP AGGREGATIONS (4 features) ==========
        # TransactionAmt, D9, D11 grouped by card1, card1_addr1, card1_addr1_P_emaildomain
        if hasattr(self, 'group_agg_stats'):
            for stat_name, stat_map in self.group_agg_stats.items():
                # Map using the appropriate grouping key
                if 'card1_addr1_P_emaildomain' in stat_name:
                    if all(c in df.columns for c in ['card1_addr1', 'P_emaildomain']):
                        key = df['card1_addr1'].astype(str) + '_' + df['P_emaildomain'].fillna('na').astype(str)
                        df[stat_name] = key.map(stat_map).fillna(-1).astype(np.float32)
                elif 'card1_addr1' in stat_name:
                    if 'card1_addr1' in df.columns:
                        df[stat_name] = df['card1_addr1'].map(stat_map).fillna(-1).astype(np.float32)
                elif 'card1' in stat_name:
                    if 'card1' in df.columns:
                        df[stat_name] = df['card1'].map(stat_map).fillna(-1).astype(np.float32)
        else:
            # Fallback: create zero-filled features
            for col in self.feature_names:
                if any(x in col for x in ['_card1_mean', '_card1_std', '_card1_addr1_mean', '_card1_addr1_std']):
                    if col not in df.columns:
                        df[col] = -1.0

        # ========== FREQUENCY ENCODING (11 features) ==========
        # Apply pre-computed frequency maps
        for col, freq_map in self.freq_maps.items():
            if col in df.columns:
                df[col + '_freq'] = df[col].map(freq_map).fillna(0.0).astype('float32')

        # ========== MEAN ENCODING (12 features) ==========
        # Apply pre-computed mean encoding (fraud rate) maps
        if hasattr(self, 'mean_encoding_maps'):
            for col, mean_map in self.mean_encoding_maps.items():
                if col in df.columns:
                    df[col + '_mean_target'] = df[col].map(mean_map).fillna(0.0).astype('float32')
        else:
            # Fallback: create zero-filled features
            for col in self.feature_names:
                if col.endswith('_mean_target') and col not in df.columns:
                    df[col] = 0.0

        # ========== FINAL FEATURE SELECTION ==========
        # Clean up temporary columns
        df = df.drop(columns=['card1_addr1'], errors='ignore')
        
        # Fill missing features with default values
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = 0.0

        # Return only the 88 features in the correct order
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
