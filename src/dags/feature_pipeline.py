"""
Feature pipeline for the IEEE-CIS benchmark model: ONE definition used by
training and by both serving paths (REST API and Spark).

    pipeline = IEEECISFeaturePipeline().fit(train_period_df)   # learns every map
    X_train  = pipeline.fit_transform(train_period_df)          # out-of-fold target encoding
    X_test   = pipeline.transform(later_df)                     # same code as serving

`fit()` learns all statistics from the *training period only*, so validation,
test and live traffic are transformed exactly like a production system would:
with history that existed before them. The module and class names must stay
`feature_pipeline.IEEECISFeaturePipeline` because pickled artifacts reference them.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

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

FREQ_COLS = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
             'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain']
MEAN_ENCODE_COLS = ['P_emaildomain', 'R_emaildomain', 'DeviceInfo', 'DeviceType',
                    'card4', 'card6', 'ProductCD']
NUMERIC_RAW = ['TransactionAmt', 'TransactionDT', 'card1', 'card2', 'card3', 'card5',
               'addr1', 'addr2', 'dist1', 'dist2', 'id_02']
STRING_RAW = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
              'DeviceInfo', 'DeviceType']
C_COLS = [f'C{i}' for i in range(1, 15)]
D_COLS = [f'D{i}' for i in range(1, 16)]
M_COLS = [f'M{i}' for i in range(1, 10)]
MAGIC_NUNIQUE = {  # feature suffix -> source column
    'email_nunique': 'P_emaildomain', 'dist1_nunique': 'dist1', 'DT_M_nunique': 'DT_M',
    'id02_nunique': 'id_02', 'cents_nunique': 'TransactionAmt_decimal', 'C13_nunique': 'C13',
    'V314_nunique': 'V314', 'V127_nunique': 'V127', 'V136_nunique': 'V136',
    'V309_nunique': 'V309', 'V307_nunique': 'V307', 'V320_nunique': 'V320',
}
# Raw Vesta counters/timedeltas passed straight to the model (NaN = missing; XGBoost handles it)
RAW_PASSTHROUGH = C_COLS + ['D1', 'D2', 'D3', 'D4', 'D10', 'D11', 'D15'] + M_COLS
M_MAP = {'T': 1, 'F': 0, 't': 1, 'f': 0, 'M0': 0, 'M1': 1, 'M2': 2}
START_DATE = pd.Timestamp('2017-11-30')


class IEEECISFeaturePipeline:
    """Fit on the training period; transform anything (training, test, live)."""

    def __init__(self):
        self.freq_maps: Dict[str, Dict[Any, float]] = {}
        self.scaler: Optional[RobustScaler] = None
        self.feature_names: list = []
        self.risky_domains = set(RISKY_DOMAINS)

    # ------------------------------------------------------------------ prepare
    @staticmethod
    def _prepare(df: pd.DataFrame, extra_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normalise raw input identically in fit and transform.

        - an absent field is created as NaN (same as present-but-empty), so a row's
          features never depend on which other rows share its batch
        - JSON null / None becomes NaN (frequency maps have a NaN key, not a None key)
        - emails and device strings are lower-cased; M flags are encoded to numbers
        """
        df = df.copy()
        needed = NUMERIC_RAW + STRING_RAW + D_COLS + C_COLS + M_COLS + list(extra_cols or [])
        for col in needed:
            if col not in df.columns:
                df[col] = np.nan

        for col in NUMERIC_RAW + C_COLS + D_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in STRING_RAW:
            s = df[col].where(df[col].notna(), np.nan)
            if col in ('P_emaildomain', 'R_emaildomain', 'DeviceInfo'):
                s = s.map(lambda v: v.strip().lower() if isinstance(v, str) else v)
            df[col] = s
        for col in M_COLS:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].map(M_MAP)
            df[col] = df[col].astype('float32')

        df['card1_addr1'] = df['card1'].fillna(-999).astype(str) + '_' + df['addr1'].fillna(-999).astype(str)
        df['day'] = df['TransactionDT'] / (24 * 60 * 60)
        df['magic_uid'] = df['card1_addr1'] + '_' + np.floor(df['day'] - df['D1']).fillna(-999).astype(str)
        amt = df['TransactionAmt']
        df['TransactionAmt_decimal'] = ((amt - np.floor(amt.fillna(0))) * 1000).fillna(0).astype('int16')
        return df

    # ---------------------------------------------------------------------- fit
    def fit(self, train_df: pd.DataFrame, target: str = 'isFraud', feature_names: Optional[list] = None):
        """Learn every statistic used by transform() from the training period only."""
        df = self._prepare(train_df, extra_cols=list(set(MAGIC_NUNIQUE.values()) - {'DT_M', 'TransactionAmt_decimal'}))
        t = df['TransactionDT']
        months = START_DATE + pd.to_timedelta(t.fillna(0), unit='s')
        df['DT_M'] = (months.dt.year - 2017) * 12 + months.dt.month

        # Frequency encoding (normalised counts, NaN kept as its own category)
        self.freq_maps = {}
        for col in FREQ_COLS:
            vc = df[col].value_counts(dropna=False)
            self.freq_maps[col] = (vc / vc.sum()).to_dict()

        # Amount relative to the card's normal spending
        self.card1_amt_mean = df.groupby('card1')['TransactionAmt'].mean().to_dict()
        self.card1_amt_std = df.groupby('card1')['TransactionAmt'].std().fillna(50).to_dict()
        self.card4_amt_mean = df.groupby('card4')['TransactionAmt'].mean().to_dict()
        self.card4_amt_std = df.groupby('card4')['TransactionAmt'].std().fillna(50).to_dict()

        # Magic-UID (pseudo customer) aggregates
        g = df.groupby('magic_uid')
        stats: Dict[str, Dict[str, float]] = {}
        for stat in ('mean', 'std'):
            stats[f'magic_uid_TransactionAmt_{stat}'] = g['TransactionAmt'].agg(stat).to_dict()
        for col in ('D4', 'D9', 'D10', 'D15'):
            for stat in ('mean', 'std'):
                stats[f'magic_uid_{col}_{stat}'] = g[col].agg(stat).to_dict()
        for i in range(1, 15):
            if i != 3:
                stats[f'magic_uid_C{i}_mean'] = g[f'C{i}'].mean().to_dict()
        for col in M_COLS:
            stats[f'magic_uid_{col}_mean'] = g[col].mean().to_dict()
        stats['magic_uid_C14_std'] = g['C14'].std().to_dict()
        vc = df['magic_uid'].value_counts(normalize=True)
        stats['magic_uid_freq'] = vc.to_dict()
        for name, src in MAGIC_NUNIQUE.items():
            stats[f'magic_uid_{name}'] = g[src].nunique().to_dict()
        self.magic_uid_stats = stats

        # Group aggregations at three granularities
        df['card1_addr1_P_emaildomain'] = df['card1_addr1'] + '_' + df['P_emaildomain'].fillna('na').astype(str)
        self.group_agg_stats = {}
        for col in ('TransactionAmt', 'D9', 'D11'):
            for group in ('card1', 'card1_addr1', 'card1_addr1_P_emaildomain'):
                for stat in ('mean', 'std'):
                    self.group_agg_stats[f'{col}_{group}_{stat}'] = df.groupby(group)[col].agg(stat).to_dict()

        # Count encodings of card / card+address
        self.card_encoding_maps = {
            name: {'key_col': key, 'freq_map': df[key].value_counts().to_dict()}
            for name, key in (('card1_addr1_FE', 'card1_addr1'), ('card1_FE', 'card1'), ('card2_FE', 'card2'))
        }

        # Smoothed target (fraud-rate) encoding; train rows are re-encoded out-of-fold in fit_transform
        self.global_fraud_rate = float(train_df[target].mean())
        self.mean_encoding_smoothing = 10.0
        self.mean_encoding_maps = {
            col: self._target_rate_map(df[col].fillna('missing'), train_df[target])
            for col in MEAN_ENCODE_COLS
        }

        self.feature_names = list(feature_names) if feature_names else self.default_feature_names()
        self.fitted_period = (float(t.min()), float(t.max()))
        return self

    def _target_rate_map(self, keys: pd.Series, y: pd.Series) -> Dict[Any, float]:
        agg = pd.DataFrame({'k': keys.values, 'y': y.values}).groupby('k')['y'].agg(['sum', 'count'])
        m, prior = self.mean_encoding_smoothing, self.global_fraud_rate
        return ((agg['sum'] + prior * m) / (agg['count'] + m)).to_dict()

    def fit_transform(self, train_df: pd.DataFrame, target: str = 'isFraud', n_folds: int = 5) -> pd.DataFrame:
        """
        Fit, then transform the training rows with OUT-OF-FOLD target encoding
        (contiguous time folds), so a row's own label never leaks into its features.
        """
        self.fit(train_df, target)
        X = self.transform(train_df)
        y = train_df[target].reset_index(drop=True)
        prepared = self._prepare(train_df).reset_index(drop=True)
        folds = np.array_split(np.arange(len(prepared)), n_folds)
        for col in MEAN_ENCODE_COLS:
            name = f'{col}_mean_target'
            if name not in X.columns:
                continue
            keys = prepared[col].fillna('missing')
            oof = np.full(len(prepared), self.global_fraud_rate, dtype='float32')
            for idx in folds:
                rest = np.setdiff1d(np.arange(len(prepared)), idx)
                rate = self._target_rate_map(keys.iloc[rest], y.iloc[rest])
                oof[idx] = keys.iloc[idx].map(rate).fillna(self.global_fraud_rate).values
            X[name] = oof
        return X

    def default_feature_names(self) -> List[str]:
        amount = ['TransactionAmt', 'log_TransactionAmt', 'sqrt_TransactionAmt', 'TransactionAmt_decimal',
                  'TransactionAmt_to_mean_card1', 'TransactionAmt_to_std_card1',
                  'TransactionAmt_to_mean_card4', 'TransactionAmt_to_std_card4']
        email = ['email_risky', 'email_is_generic', 'is_high_risk_email', 'is_disposable_email',
                 'email_domain_length', 'email_is_short_domain', 'email_match']
        card = ['card_is_discover', 'card_is_amex', 'card_is_charge', 'product_is_C', 'product_is_R']
        dist = ['dist1_filled', 'has_dist1', 'dist1_high', 'dist2_filled', 'has_dist2', 'dist2_high']
        magic = list(getattr(self, 'magic_uid_stats', {}).keys()) + ['outsider15']
        d_norm = [f'{c}_normalized' for c in ('D4', 'D10', 'D11', 'D15')]
        enc = list(getattr(self, 'card_encoding_maps', {}).keys())
        group = list(getattr(self, 'group_agg_stats', {}).keys())
        freq = [f'{c}_freq' for c in self.freq_maps]
        mean = [f'{c}_mean_target' for c in getattr(self, 'mean_encoding_maps', {})]
        names = amount + email + card + dist + magic + d_norm + enc + group + freq + mean + RAW_PASSTHROUGH
        return list(dict.fromkeys(names))

    # ---------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Raw IEEE-CIS rows (any subset of columns) -> model features, in training order."""
        df = self._prepare(df)
        out: Dict[str, Any] = {}
        amt = df['TransactionAmt'].astype('float32')

        # Amount
        out['TransactionAmt'] = amt
        out['log_TransactionAmt'] = np.log1p(amt.fillna(0)).astype('float32')
        out['sqrt_TransactionAmt'] = np.sqrt(amt.fillna(0).clip(lower=0)).astype('float32')
        out['TransactionAmt_decimal'] = df['TransactionAmt_decimal']
        for card in ('card1', 'card4'):
            mean_map = getattr(self, f'{card}_amt_mean', None)
            std_map = getattr(self, f'{card}_amt_std', None)
            if mean_map is not None:
                out[f'TransactionAmt_to_mean_{card}'] = (amt / (df[card].map(mean_map).fillna(100) + 1e-5)).astype('float32')
                out[f'TransactionAmt_to_std_{card}'] = (amt / (df[card].map(std_map).fillna(50) + 1e-5)).astype('float32')
            else:
                out[f'TransactionAmt_to_mean_{card}'] = 1.0
                out[f'TransactionAmt_to_std_{card}'] = 1.0

        # Email
        email = df['P_emaildomain']
        out['email_risky'] = email.isin(list(RISKY_DOMAINS)).astype('int8')
        out['email_is_generic'] = email.isin(['gmail.com', 'yahoo.com', 'hotmail.com']).astype('int8')
        out['is_high_risk_email'] = email.isin(list(HIGH_RISK_DOMAINS)).astype('int8')
        out['is_disposable_email'] = email.fillna('').astype(str).str.contains(
            'temp|disposable|guerrilla|throwaway|fake|spam|trash', case=False, regex=True).astype('int8')
        out['email_domain_length'] = email.fillna('').astype(str).str.len().clip(upper=127).astype('int8')
        out['email_is_short_domain'] = (out['email_domain_length'] <= 8).astype('int8')
        out['email_match'] = (email == df['R_emaildomain']).fillna(False).astype('int8')

        # Card / product
        out['card_is_discover'] = (df['card4'] == 'discover').astype('int8')
        out['card_is_amex'] = (df['card4'] == 'american express').astype('int8')
        out['card_is_charge'] = (df['card6'] == 'charge card').astype('int8')
        out['product_is_C'] = (df['ProductCD'] == 'C').astype('int8')
        out['product_is_R'] = (df['ProductCD'] == 'R').astype('int8')

        # Distance
        for d in ('dist1', 'dist2'):
            out[f'{d}_filled'] = df[d].fillna(0).astype('float32')
            out[f'has_{d}'] = df[d].notna().astype('int8')
            out[f'{d}_high'] = (df[d].fillna(0) > 1000).astype('int8')

        # Magic-UID aggregates (history from the training period)
        stats = getattr(self, 'magic_uid_stats', None)
        if stats:
            for name, mapping in stats.items():
                default = 0 if name.endswith(('_freq', '_nunique')) or name.endswith('TransactionAmt_mean') \
                    or name.endswith('TransactionAmt_std') else -1
                out[name] = df['magic_uid'].map(mapping).fillna(default).astype('float32')
        out['outsider15'] = ((df['D1'] - df['D15']).abs() > 3).astype('int8')

        # Normalised D-columns
        for d in ('D4', 'D10', 'D11', 'D15'):
            out[f'{d}_normalized'] = (df[d] - df['day']).fillna(-1).astype('float32')

        # Card count encodings
        enc = getattr(self, 'card_encoding_maps', None)
        if enc:
            for name, spec in enc.items():
                out[name] = df[spec['key_col']].map(spec['freq_map']).fillna(0).astype('int32')
        else:
            # Artifact without encoding maps: use the batch-of-1 value for every row,
            # never counts within the incoming batch (see postmortem).
            for name, key in (('card1_addr1_FE', 'card1_addr1'), ('card1_FE', 'card1'), ('card2_FE', 'card2')):
                out[name] = df[key].notna().astype('int32')

        # Group aggregations
        groups = getattr(self, 'group_agg_stats', None)
        if groups:
            key_p = df['card1_addr1'] + '_' + df['P_emaildomain'].fillna('na').astype(str)
            for name, mapping in groups.items():
                if 'card1_addr1_P_emaildomain' in name:
                    key = key_p
                elif 'card1_addr1' in name:
                    key = df['card1_addr1']
                else:
                    key = df['card1']
                out[name] = key.map(mapping).fillna(-1).astype('float32')

        # Frequency encodings
        for col, freq_map in self.freq_maps.items():
            if col in df.columns:
                out[f'{col}_freq'] = df[col].map(freq_map).fillna(0.0).astype('float32')

        # Target (fraud-rate) encodings
        mean_maps = getattr(self, 'mean_encoding_maps', None)
        prior = getattr(self, 'global_fraud_rate', 0.0)
        if mean_maps:
            for col, mapping in mean_maps.items():
                out[f'{col}_mean_target'] = df[col].fillna('missing').map(mapping).fillna(prior).astype('float32')

        # Raw pass-through columns
        for col in RAW_PASSTHROUGH:
            out[col] = df[col].astype('float32')

        features = pd.DataFrame(out, index=df.index)
        # Anything an (older) artifact expects but cannot produce gets the legacy
        # default: -1.0 for group aggregates, 0.0 otherwise.
        missing = [f for f in self.feature_names if f not in features.columns]
        if missing:
            defaults = {f: (-1.0 if any(x in f for x in ('_card1_mean', '_card1_std', '_card1_addr1_mean',
                                                          '_card1_addr1_std')) else 0.0) for f in missing}
            features = pd.concat([features, pd.DataFrame(defaults, index=df.index)], axis=1)
        return features[self.feature_names]

    def get_config(self) -> Dict[str, Any]:
        return {'freq_maps': self.freq_maps, 'scaler': self.scaler,
                'feature_names': self.feature_names, 'risky_domains': self.risky_domains}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'IEEECISFeaturePipeline':
        pipeline = cls()
        pipeline.freq_maps = config.get('freq_maps', {})
        pipeline.scaler = config.get('scaler')
        pipeline.feature_names = config.get('feature_names', [])
        pipeline.risky_domains = config.get('risky_domains', set(RISKY_DOMAINS))
        return pipeline
