"""
Training-side guarantees (no trained artifacts needed):
- the IEEE-CIS pipeline is fitted on the training period only and transform() is
  the single feature definition used for every split and for serving
- target encodings of training rows are out-of-fold
- the simulator produces every fraud scenario with delayed chargebacks
"""

import numpy as np
import pandas as pd

from feature_pipeline import MEAN_ENCODE_COLS, IEEECISFeaturePipeline
from simulator import SimConfig, simulate


def ieee_like(n=400, seed=0, t0=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "TransactionID": np.arange(n), "TransactionDT": t0 + np.sort(rng.integers(0, 30 * 86400, n)),
        "TransactionAmt": rng.uniform(1, 500, n).round(2), "ProductCD": rng.choice(["W", "C", "R"], n),
        "card1": rng.integers(1000, 1020, n), "card2": rng.choice([100.0, 200.0, np.nan], n),
        "card4": rng.choice(["visa", "discover"], n), "card6": rng.choice(["debit", "credit"], n),
        "addr1": rng.choice([300.0, 315.0, np.nan], n), "P_emaildomain": rng.choice(["gmail.com", "tempmail.com", None], n),
        "D1": rng.integers(0, 50, n).astype(float), "C1": rng.integers(0, 5, n).astype(float),
        "M4": rng.choice(["M0", "M1", None], n), "isFraud": (rng.random(n) < 0.1).astype(int),
    })


def test_transform_matches_feature_names_for_any_input():
    train = ieee_like()
    pipe = IEEECISFeaturePipeline().fit(train)
    for frame in (train, ieee_like(50, seed=1, t0=31 * 86400), pd.DataFrame([{"TransactionAmt": 10.0}])):
        out = pipe.transform(frame)
        assert list(out.columns) == pipe.feature_names


def test_statistics_come_from_training_period_only():
    train, later = ieee_like(), ieee_like(50, seed=1, t0=31 * 86400)
    later["card1"] = 9999  # a card never seen in the training period
    out = IEEECISFeaturePipeline().fit(train).transform(later)
    assert (out["card1_freq"] == 0).all()
    assert (out["TransactionAmt_card1_mean"] == -1).all()


def test_training_rows_get_out_of_fold_target_encoding():
    train = ieee_like(1000)
    pipe = IEEECISFeaturePipeline()
    oof = pipe.fit_transform(train)
    in_sample = pipe.transform(train)
    cols = [f"{c}_mean_target" for c in MEAN_ENCODE_COLS]
    assert not np.allclose(oof[cols].values, in_sample[cols].values)


def test_simulator_scenarios_and_delayed_labels():
    df = simulate(SimConfig(n_customers=200, n_terminals=400, n_days=30, seed=3))
    scenarios = set(df.loc[df["is_fraud"] == 1, "fraud_scenario"])
    assert {"compromised_terminal", "account_takeover", "card_testing"} <= scenarios
    fraud = df[df["is_fraud"] == 1]
    tx_epoch = fraud["tx_datetime"].astype("int64") / 1e9
    delay_days = (fraud["chargeback_ts"] - tx_epoch) / 86400
    assert delay_days.between(7, 60).all()
    assert df.loc[df["is_fraud"] == 0, "chargeback_ts"].isna().all()
    assert df["tx_datetime"].is_monotonic_increasing
