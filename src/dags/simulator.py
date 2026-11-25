"""
Synthetic card-transaction simulator for the decision-time (live) model.

Design follows the transaction-data simulator of the ULB "Reproducible Machine
Learning for Credit Card Fraud Detection" handbook (customers and terminals on a
grid, Poisson daily activity, normal spending profiles, scenario-based frauds),
extended with the fields this system's real-time layer uses (email domain,
device, card type, distance from home) and two more attack patterns.

Fraud scenarios
  1 high_amount           amount above a fixed limit
  2 compromised_terminal  every payment at a breached terminal for 28 days
  3 compromised_card      1/3 of a victim's payments for 14 days, amount x5
  4 account_takeover      new (shared attacker) device, disposable email, far away
  5 card_testing          burst of micro-payments, then one large charge

Every fraud gets a chargeback time 7-60 days later: labels arrive late, as in
production. IEEE-CIS cannot provide this because its labels are immediate and
its features are anonymised.

Usage:  python simulator.py --out /app/data/synthetic/transactions.parquet
"""

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

DAY = 86400

COMMON_EMAILS = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'icloud.com', 'aol.com']
EMAIL_P = [0.45, 0.2, 0.12, 0.1, 0.08, 0.05]
DISPOSABLE = ['tempmail.com', 'mailinator.com', 'guerrillamail.com', 'yopmail.com', 'trashmail.com']
NETWORKS = ['visa', 'mastercard', 'american express', 'discover']
NETWORK_P = [0.6, 0.3, 0.06, 0.04]


@dataclass
class SimConfig:
    n_customers: int = 5000
    n_terminals: int = 10000
    n_days: int = 180
    start_date: str = '2025-01-01'
    radius: float = 5.0            # customers shop at terminals within this grid distance
    km_per_unit: float = 10.0      # grid unit -> km for dist1
    high_amount_limit: float = 220.0
    compromised_terminals_per_day: int = 2
    compromised_cards_per_day: int = 3
    ato_victims_per_day: int = 2
    card_testing_victims_per_day: int = 2
    attacker_devices: int = 25
    seed: int = 42


def _customers(cfg: SimConfig, rng) -> pd.DataFrame:
    n = cfg.n_customers
    mean_amount = rng.uniform(5, 100, n)
    return pd.DataFrame({
        'customer_id': np.arange(n),
        'x': rng.uniform(0, 100, n), 'y': rng.uniform(0, 100, n),
        'mean_amount': mean_amount, 'std_amount': mean_amount / 2,
        'mean_tx_per_day': rng.uniform(0, 4, n),
        'email': rng.choice(COMMON_EMAILS, n, p=EMAIL_P),
        'card4': rng.choice(NETWORKS, n, p=NETWORK_P),
        'card6': rng.choice(['debit', 'credit'], n, p=[0.6, 0.4]),
        'n_devices': rng.choice([1, 2], n, p=[0.7, 0.3]),
    })


def _terminals(cfg: SimConfig, rng) -> pd.DataFrame:
    return pd.DataFrame({'terminal_id': np.arange(cfg.n_terminals),
                         'x': rng.uniform(0, 100, cfg.n_terminals), 'y': rng.uniform(0, 100, cfg.n_terminals)})


def _genuine(cfg: SimConfig, customers, terminals, rng) -> pd.DataFrame:
    txy = terminals[['x', 'y']].values.astype('float32')
    parts = []
    for c in customers.itertuples(index=False):
        d = np.sqrt(((txy - np.array([c.x, c.y], dtype='float32')) ** 2).sum(axis=1))
        nearby = np.where(d < cfg.radius)[0]
        if nearby.size == 0:
            continue
        per_day = rng.poisson(c.mean_tx_per_day, cfg.n_days)
        n = int(per_day.sum())
        if n == 0:
            continue
        day = np.repeat(np.arange(cfg.n_days), per_day)
        secs = np.clip(rng.normal(DAY / 2, 20000, n), 0, DAY - 1)
        amount = rng.normal(c.mean_amount, c.std_amount, n)
        neg = amount < 0
        amount[neg] = rng.uniform(0, c.mean_amount * 2, neg.sum())
        term = rng.choice(nearby, n)
        device_idx = rng.choice(c.n_devices, n, p=[1.0] if c.n_devices == 1 else [0.85, 0.15])
        parts.append(pd.DataFrame({
            'customer_id': c.customer_id, 'terminal_id': term,
            'ts': day * DAY + secs, 'TransactionAmt': np.round(amount, 2),
            'P_emaildomain': c.email, 'DeviceInfo': [f'dev_{c.customer_id}_{i}' for i in device_idx],
            'card4': c.card4, 'card6': c.card6, 'dist1': np.round(d[term] * cfg.km_per_unit, 1),
        }))
    df = pd.concat(parts, ignore_index=True)
    df['is_fraud'], df['fraud_scenario'] = 0, ''
    return df


def _inject(cfg: SimConfig, df: pd.DataFrame, customers, terminals, rng) -> pd.DataFrame:
    day = (df['ts'] // DAY).astype(int)

    # 1 high amount
    mask = df['TransactionAmt'] > cfg.high_amount_limit
    df.loc[mask, ['is_fraud', 'fraud_scenario']] = [1, 'high_amount']

    for d in range(cfg.n_days):
        # 2 compromised terminals: all payments for the next 28 days
        for t in rng.choice(cfg.n_terminals, cfg.compromised_terminals_per_day, replace=False):
            m = (df['terminal_id'] == t) & (day >= d) & (day < d + 28) & (df['is_fraud'] == 0)
            df.loc[m, ['is_fraud', 'fraud_scenario']] = [1, 'compromised_terminal']
        # 3 compromised cards: 1/3 of payments for 14 days at 5x the amount
        for c in rng.choice(cfg.n_customers, cfg.compromised_cards_per_day, replace=False):
            idx = df.index[(df['customer_id'] == c) & (day >= d) & (day < d + 14) & (df['is_fraud'] == 0)]
            pick = rng.choice(idx, len(idx) // 3, replace=False) if len(idx) >= 3 else []
            df.loc[pick, 'TransactionAmt'] = (df.loc[pick, 'TransactionAmt'] * 5).round(2)
            df.loc[pick, ['is_fraud', 'fraud_scenario']] = [1, 'compromised_card']

    extra = []
    attacker_devices = [f'attacker_{i}' for i in range(cfg.attacker_devices)]
    for d in range(cfg.n_days):
        # 4 account takeover: shared attacker device, disposable email, far terminal
        for c in rng.choice(cfg.n_customers, cfg.ato_victims_per_day, replace=False):
            cust = customers.iloc[c]
            device = rng.choice(attacker_devices)
            email = rng.choice(DISPOSABLE)
            for _ in range(rng.integers(3, 7)):
                extra.append({
                    'customer_id': c, 'terminal_id': int(rng.integers(cfg.n_terminals)),
                    'ts': (d + rng.uniform(0, 2)) * DAY, 'TransactionAmt': round(float(rng.uniform(300, 2500)), 2),
                    'P_emaildomain': email, 'DeviceInfo': device, 'card4': cust.card4, 'card6': cust.card6,
                    'dist1': round(float(rng.uniform(1000, 4000)), 1), 'is_fraud': 1, 'fraud_scenario': 'account_takeover',
                })
        # 5 card testing: 5-10 micro payments within 10 minutes, then one large charge
        for c in rng.choice(cfg.n_customers, cfg.card_testing_victims_per_day, replace=False):
            cust = customers.iloc[c]
            start = (d + rng.uniform(0, 1)) * DAY
            terminal = int(rng.integers(cfg.n_terminals))
            device = rng.choice(attacker_devices)
            n_tests = int(rng.integers(5, 11))
            for k in range(n_tests + 1):
                big = k == n_tests
                extra.append({
                    'customer_id': c, 'terminal_id': terminal,
                    'ts': start + (rng.uniform(600, 1800) if big else rng.uniform(0, 600)),
                    'TransactionAmt': round(float(rng.uniform(500, 1500) if big else rng.uniform(0.5, 2.0)), 2),
                    'P_emaildomain': cust.email, 'DeviceInfo': device, 'card4': cust.card4, 'card6': cust.card6,
                    'dist1': round(float(rng.uniform(50, 800)), 1), 'is_fraud': 1, 'fraud_scenario': 'card_testing',
                })
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True)


def simulate(cfg: SimConfig = SimConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    customers, terminals = _customers(cfg, rng), _terminals(cfg, rng)
    df = _inject(cfg, _genuine(cfg, customers, terminals, rng), customers, terminals, rng)
    df = df[df['ts'] < cfg.n_days * DAY].sort_values('ts').reset_index(drop=True)

    start = pd.Timestamp(cfg.start_date, tz='UTC')
    df['tx_datetime'] = start + pd.to_timedelta(df['ts'], unit='s')
    epoch = start.timestamp() + df['ts']
    delay = rng.uniform(7, 60, len(df)) * DAY
    df['chargeback_ts'] = np.where(df['is_fraud'] == 1, epoch + delay, np.nan)
    df['transaction_id'] = np.arange(len(df))
    df['card1'] = df['customer_id']   # the real-time services key velocity/ATO by card1
    cols = ['transaction_id', 'tx_datetime', 'customer_id', 'card1', 'terminal_id', 'TransactionAmt',
            'P_emaildomain', 'DeviceInfo', 'card4', 'card6', 'dist1', 'is_fraud', 'fraud_scenario', 'chargeback_ts']
    return df[cols]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic card transactions')
    parser.add_argument('--out', default='transactions.parquet')
    parser.add_argument('--customers', type=int, default=SimConfig.n_customers)
    parser.add_argument('--terminals', type=int, default=SimConfig.n_terminals)
    parser.add_argument('--days', type=int, default=SimConfig.n_days)
    parser.add_argument('--seed', type=int, default=SimConfig.seed)
    a = parser.parse_args()
    data = simulate(SimConfig(n_customers=a.customers, n_terminals=a.terminals, n_days=a.days, seed=a.seed))
    data.to_parquet(a.out, index=False)
    print(f"{len(data):,} transactions, fraud rate {data['is_fraud'].mean():.3%} -> {a.out}")
    print(data['fraud_scenario'].replace('', 'genuine').value_counts().to_string())
