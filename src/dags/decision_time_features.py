"""
Decision-time features: only information that exists when a payment is authorised.

The SAME `DecisionTimeFeatures` code computes features offline (training: replay
the historical stream through an in-memory store) and online (serving: Redis
store, `src/inference/feature_store.py`). Both stores implement:

    add(key, ts, value, tag)        record one event
    window(key, start, end)         events with start <= ts < end -> [(ts, value, tag)]
    trim(key, before)               drop events older than `before`

so training/serving skew is impossible by construction: there is one feature
definition and the store is the only thing that changes.

Fraud labels are delayed: a terminal's risk features only see frauds whose
chargeback has already been reported (`record_label`), exactly as in production.
"""

import bisect
import heapq
import math
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

HOUR, DAY = 3600.0, 86400.0
RISKY_DOMAINS = {
    'anonymous.com', 'mailinator.com', 'tempmail.com', 'dispostable.com', 'yopmail.com',
    '10minutemail.com', 'guerrillamail.com', 'sharklasers.com', 'trashmail.com',
}

FEATURE_NAMES = [
    # transaction itself
    'amount', 'log_amount', 'hour', 'is_night', 'is_weekend',
    'email_risky', 'dist_from_home', 'dist_high', 'card_is_credit',
    # customer velocity / spending profile
    'cust_nb_tx_1h', 'cust_nb_tx_1d', 'cust_nb_tx_7d', 'cust_nb_tx_30d',
    'cust_amount_sum_1h', 'cust_avg_amount_1d', 'cust_avg_amount_7d', 'cust_avg_amount_30d',
    'amount_to_avg_30d', 'secs_since_last_tx',
    # customer novelty (account-takeover signals)
    'is_new_device_30d', 'is_new_email_30d', 'cust_nb_devices_7d',
    # terminal (merchant) activity and delayed-label risk
    'term_nb_tx_1d', 'term_nb_tx_7d', 'term_nb_tx_28d',
    'term_risk_1d', 'term_risk_7d', 'term_risk_28d',
    # shared-entity (minimal graph) signal
    'device_nb_customers_30d',
]


def to_epoch(value) -> float:
    """Accept epoch seconds, ISO-8601 strings or datetimes."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return datetime.now(timezone.utc).timestamp()
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


class MemoryStore:
    """In-memory store used for offline feature generation (and tests)."""

    def __init__(self):
        self._ts: Dict[str, List[float]] = {}
        self._ev: Dict[str, List[Tuple[float, float, str]]] = {}

    def add(self, key: str, ts: float, value: float = 1.0, tag: str = '') -> None:
        ts_list = self._ts.setdefault(key, [])
        events = self._ev.setdefault(key, [])
        i = bisect.bisect_right(ts_list, ts)
        ts_list.insert(i, ts)
        events.insert(i, (ts, float(value), str(tag)))

    def window(self, key: str, start: float, end: float) -> List[Tuple[float, float, str]]:
        ts_list = self._ts.get(key)
        if not ts_list:
            return []
        lo, hi = bisect.bisect_left(ts_list, start), bisect.bisect_left(ts_list, end)
        return self._ev[key][lo:hi]

    def trim(self, key: str, before: float) -> None:
        ts_list = self._ts.get(key)
        if ts_list:
            cut = bisect.bisect_left(ts_list, before)
            if cut:
                del ts_list[:cut]
                del self._ev[key][:cut]


def _norm(txn: Dict) -> Dict:
    """Map a payload (simulator or producer API names) onto canonical fields."""
    def first(*names, default=None):
        for n in names:
            v = txn.get(n)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                return v
        return default

    return {
        'ts': to_epoch(first('tx_datetime', 'timestamp_epoch')),
        'customer_id': str(first('customer_id', 'card1', default='unknown')),
        'terminal_id': str(first('terminal_id', default='unknown')),
        'amount': float(first('TransactionAmt', 'amount', default=0.0)),
        'email': str(first('P_emaildomain', 'email_domain', default='')).lower(),
        'device': str(first('DeviceInfo', 'device_id', default='')),
        'dist': float(first('dist1', 'dist_from_home', default=0.0)),
        'card_type': str(first('card6', 'card_type', default='')).lower(),
    }


class DecisionTimeFeatures:
    """One feature definition for training and serving."""

    feature_names = FEATURE_NAMES

    def __init__(self, store):
        self.store = store

    # ------------------------------------------------------------------ reads
    def compute(self, txn: Dict) -> Dict[str, float]:
        """Features from history strictly BEFORE this transaction (no self-leakage)."""
        t = _norm(txn)
        ts, amount = t['ts'], t['amount']
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        f: Dict[str, float] = {
            'amount': amount,
            'log_amount': math.log1p(max(amount, 0.0)),
            'hour': dt.hour,
            'is_night': float(dt.hour >= 22 or dt.hour < 6),
            'is_weekend': float(dt.weekday() >= 5),
            'email_risky': float(t['email'] in RISKY_DOMAINS or 'temp' in t['email'] or 'trash' in t['email']),
            'dist_from_home': t['dist'],
            'dist_high': float(t['dist'] > 1000),
            'card_is_credit': float(t['card_type'] == 'credit'),
        }

        cust = self.store.window(f"c:{t['customer_id']}", ts - 30 * DAY, ts)
        for name, w in (('1h', HOUR), ('1d', DAY), ('7d', 7 * DAY), ('30d', 30 * DAY)):
            events = [e for e in cust if e[0] >= ts - w]
            f[f'cust_nb_tx_{name}'] = float(len(events))
            if name == '1h':
                f['cust_amount_sum_1h'] = float(sum(e[1] for e in events))
            else:
                f[f'cust_avg_amount_{name}'] = float(np.mean([e[1] for e in events])) if events else 0.0
        f['amount_to_avg_30d'] = amount / (f['cust_avg_amount_30d'] + 1.0)
        f['secs_since_last_tx'] = float(ts - cust[-1][0]) if cust else 30 * DAY

        devices_30d = {e[2].split('|', 1)[0] for e in cust}
        emails_30d = {e[2].split('|', 1)[-1] for e in cust}
        f['is_new_device_30d'] = float(bool(t['device']) and t['device'] not in devices_30d)
        f['is_new_email_30d'] = float(bool(t['email']) and t['email'] not in emails_30d)
        f['cust_nb_devices_7d'] = float(len({e[2].split('|', 1)[0] for e in cust if e[0] >= ts - 7 * DAY}))

        term_tx = self.store.window(f"t:{t['terminal_id']}", ts - 28 * DAY, ts)
        term_fraud = self.store.window(f"tf:{t['terminal_id']}", ts - 28 * DAY, ts)
        for name, w in (('1d', DAY), ('7d', 7 * DAY), ('28d', 28 * DAY)):
            n_tx = sum(1 for e in term_tx if e[0] >= ts - w)
            n_fraud = sum(1 for e in term_fraud if e[0] >= ts - w)
            f[f'term_nb_tx_{name}'] = float(n_tx)
            f[f'term_risk_{name}'] = n_fraud / n_tx if n_tx else 0.0

        if t['device']:
            dev = self.store.window(f"d:{t['device']}", ts - 30 * DAY, ts)
            f['device_nb_customers_30d'] = float(len({e[2] for e in dev} - {t['customer_id']}))
        else:
            f['device_nb_customers_30d'] = 0.0
        return f

    # ----------------------------------------------------------------- writes
    def record(self, txn: Dict) -> None:
        """Add this transaction to the history (call after scoring it)."""
        t = _norm(txn)
        ts = t['ts']
        self.store.add(f"c:{t['customer_id']}", ts, t['amount'], f"{t['device']}|{t['email']}")
        self.store.add(f"t:{t['terminal_id']}", ts, 1.0, '')
        if t['device']:
            self.store.add(f"d:{t['device']}", ts, 1.0, t['customer_id'])
        for key, keep in ((f"c:{t['customer_id']}", 31 * DAY), (f"t:{t['terminal_id']}", 29 * DAY),
                          (f"d:{t['device']}", 31 * DAY)):
            self.store.trim(key, ts - keep)

    def record_label(self, terminal_id: str, tx_ts, is_fraud: bool) -> None:
        """A chargeback/confirmation arrived for a past transaction."""
        if is_fraud:
            self.store.add(f"tf:{terminal_id}", to_epoch(tx_ts), 1.0, '')

    def process(self, txn: Dict) -> Dict[str, float]:
        features = self.compute(txn)
        self.record(txn)
        return features


def build_feature_table(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Offline: replay transactions in time order through DecisionTimeFeatures.

    Column `chargeback_ts` (epoch seconds, NaN for legitimate payments) is when a
    fraud label becomes available. Labels are fed back only once that time has
    passed in the replay, so features never use labels from the future.
    """
    df = transactions.sort_values('tx_datetime').reset_index(drop=True)
    features = DecisionTimeFeatures(MemoryStore())
    known_at = (df['chargeback_ts'].fillna(np.inf).values if 'chargeback_ts' in df.columns
                else np.full(len(df), np.inf))
    pending: List[Tuple[float, str, float]] = []
    rows = []
    for i, txn in enumerate(df.to_dict(orient='records')):
        ts = to_epoch(txn['tx_datetime'])
        while pending and pending[0][0] <= ts:
            _, terminal, tx_ts = heapq.heappop(pending)
            features.record_label(terminal, tx_ts, True)
        rows.append(features.process(txn))
        if txn.get('is_fraud') and np.isfinite(known_at[i]):
            heapq.heappush(pending, (float(known_at[i]), str(txn['terminal_id']), ts))
    return pd.concat([df, pd.DataFrame(rows, columns=FEATURE_NAMES)], axis=1)
