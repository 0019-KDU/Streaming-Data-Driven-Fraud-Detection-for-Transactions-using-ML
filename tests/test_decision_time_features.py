"""
Decision-time features: no self-leakage, delayed labels, and offline/online parity
(MemoryStore used for training vs RedisStore used for serving).
"""

import random

import pandas as pd
import pytest

from decision_time_features import FEATURE_NAMES, DecisionTimeFeatures, MemoryStore, build_feature_table
from inference.feature_store import RedisStore

T0 = 1_735_689_600.0  # 2025-01-01T00:00:00Z


def txn(ts, customer=1, terminal=10, amount=50.0, device="dev_1", email="gmail.com", dist=5.0):
    return {"tx_datetime": ts, "customer_id": customer, "terminal_id": terminal, "TransactionAmt": amount,
            "DeviceInfo": device, "P_emaildomain": email, "dist1": dist, "card6": "debit"}


def test_features_only_use_strictly_earlier_history():
    f = DecisionTimeFeatures(MemoryStore())
    first = f.process(txn(T0))
    assert first["cust_nb_tx_1d"] == 0          # the transaction does not count itself
    assert first["is_new_device_30d"] == 1.0
    second = f.process(txn(T0 + 60))
    assert second["cust_nb_tx_1h"] == 1
    assert second["cust_avg_amount_1d"] == 50.0
    assert second["is_new_device_30d"] == 0.0
    assert second["secs_since_last_tx"] == 60


def test_window_boundaries():
    f = DecisionTimeFeatures(MemoryStore())
    f.process(txn(T0))
    later = f.compute(txn(T0 + 2 * 3600))
    assert later["cust_nb_tx_1h"] == 0 and later["cust_nb_tx_1d"] == 1


def test_terminal_risk_only_after_the_chargeback_arrives():
    f = DecisionTimeFeatures(MemoryStore())
    f.process(txn(T0, terminal=99))
    before = f.compute(txn(T0 + 3600, customer=2, terminal=99))
    f.record_label("99", T0, True)
    after = f.compute(txn(T0 + 3600, customer=2, terminal=99))
    assert before["term_risk_1d"] == 0.0
    assert after["term_risk_1d"] == 1.0


def test_shared_device_counts_other_customers():
    f = DecisionTimeFeatures(MemoryStore())
    for c in (1, 2, 3):
        f.process(txn(T0 + c, customer=c, device="attacker_1"))
    assert f.compute(txn(T0 + 10, customer=4, device="attacker_1"))["device_nb_customers_30d"] == 3


def test_redis_store_matches_memory_store(fake_redis):
    """Serving (Redis) must produce exactly the training (memory) features."""
    rng = random.Random(0)
    offline, online = DecisionTimeFeatures(MemoryStore()), DecisionTimeFeatures(RedisStore(fake_redis))
    ts = T0
    for i in range(300):
        ts += rng.uniform(1, 4000)
        t = txn(ts, customer=rng.randint(1, 5), terminal=rng.randint(1, 8), amount=round(rng.uniform(1, 400), 2),
                device=f"dev_{rng.randint(1, 6)}", email=rng.choice(["gmail.com", "tempmail.com"]))
        if i % 25 == 0:
            for f in (offline, online):
                f.record_label(str(t["terminal_id"]), ts - 3600, True)
        a, b = offline.process(t), online.process(t)
        assert a == pytest.approx(b), f"feature mismatch at event {i}"


def test_build_feature_table_respects_label_delay():
    rows = [dict(txn(pd.Timestamp(T0 + k * 600, unit="s", tz="UTC"), customer=k, terminal=7),
                 is_fraud=int(k == 0), chargeback_ts=(T0 + 1800 if k == 0 else float("nan")))
            for k in range(6)]
    table = build_feature_table(pd.DataFrame(rows))
    assert list(table.columns[-len(FEATURE_NAMES):]) == FEATURE_NAMES
    # payments every 10 min; the first one's chargeback arrives at +30 min, so the
    # payments at +10 and +20 min cannot see it and the one at +30 min can
    assert table.loc[1:2, "term_risk_1d"].tolist() == [0.0, 0.0]
    assert table.loc[3, "term_risk_1d"] > 0.0
