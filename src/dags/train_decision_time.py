"""
Train the DECISION-TIME model: the model the live system serves.

Unlike the IEEE-CIS benchmark, every feature here can be computed when a payment
is authorised, by the same code online (Redis) and offline (replay), so the
deployed model sees exactly the inputs it was trained on.

Timeline (days of the simulated stream):

    0-7     warm-up: history only, not used for training
    7-105   train
    105-120 early stopping
    120-135 calibration + thresholds
    135-150 gap (label maturity)
    150-180 test (scored against the eventual ground truth)

Training happens "on day 150": a fraud only counts as fraud in train/early-stop/
calibration if its chargeback had arrived by then. Late chargebacks look
legitimate at training time, which is exactly the label delay production
fraud teams live with.

Run:  python train_decision_time.py --config /app/config.yaml
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from decision_time_features import FEATURE_NAMES, build_feature_table  # noqa: E402
from evaluation import (  # noqa: E402
    CostModel, break_even_precision, choose_review_threshold, choose_threshold_by_cost, score_reference, summarize,
)
from simulator import SimConfig, simulate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
DAY = 86400.0


def load_or_simulate(cfg: Dict[str, Any]) -> pd.DataFrame:
    path = cfg.get("data_path", "/app/data/synthetic/transactions.parquet")
    if os.path.exists(path):
        logger.info(f"Loading simulated transactions from {path}")
        return pd.read_parquet(path)
    logger.info("Simulating transactions (first run)...")
    df = simulate(SimConfig(**cfg.get("simulator", {})))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"  {len(df):,} transactions, fraud rate {df['is_fraud'].mean():.3%} -> {path}")
    return df


def rules_baseline(df: pd.DataFrame) -> np.ndarray:
    """Hard rules from the decision engine that exist on these fields."""
    amount, hour = df["amount"], df["hour"]
    return ((df["email_risky"] == 1) | (amount >= 10000)
            | (((hour >= 22) | (hour <= 5)) & (amount >= 1000))).astype(float).values


def train(config: Dict[str, Any]) -> Dict[str, Any]:
    scfg = config.get("synthetic", {})
    tcfg = config.get("training", {})
    cost = CostModel(**tcfg.get("cost_model", {}))
    budget = int(tcfg.get("daily_review_budget", 100))

    data = load_or_simulate(scfg)
    logger.info("Replaying the stream through DecisionTimeFeatures (same code as serving)...")
    table = build_feature_table(data)

    start = pd.Timestamp(table["tx_datetime"].min()).floor("D")
    day = ((table["tx_datetime"] - start).dt.total_seconds() // DAY).astype(int)
    b = scfg.get("split_days", {"warmup": 7, "train_end": 105, "early_stop_end": 120,
                                "calibration_end": 135, "test_start": 150, "test_end": 180})

    training_time = (start + pd.Timedelta(days=b["test_start"])).timestamp()
    label_at_training = ((table["is_fraud"] == 1) & (table["chargeback_ts"] <= training_time)).astype(int)
    late = int(((table["is_fraud"] == 1) & (day < b["calibration_end"])).sum() -
               label_at_training[day < b["calibration_end"]].sum())
    logger.info(f"Frauds not yet charged back at training time (look legitimate): {late:,}")

    masks = {
        "train": (day >= b["warmup"]) & (day < b["train_end"]),
        "early_stop": (day >= b["train_end"]) & (day < b["early_stop_end"]),
        "calibration": (day >= b["early_stop_end"]) & (day < b["calibration_end"]),
        "test": (day >= b["test_start"]) & (day < b["test_end"]),
    }
    X = {k: table.loc[m, FEATURE_NAMES] for k, m in masks.items()}
    y = {k: label_at_training[m].values for k, m in masks.items() if k != "test"}
    y["test"] = table.loc[masks["test"], "is_fraud"].astype(int).values
    for k in masks:
        logger.info(f"  {k:<12} {len(X[k]):>8,} rows  fraud {y[k].mean():.3%}")

    spw = float((y["train"] == 0).sum()) / max(float(y["train"].sum()), 1.0)
    params = dict(scfg.get("xgboost", {}))
    model = XGBClassifier(
        n_estimators=params.get("n_estimators", 1000), learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 6), subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="aucpr", early_stopping_rounds=50,
        tree_method="hist", random_state=42, n_jobs=-1,
    )
    model.fit(X["train"], y["train"], eval_set=[(X["early_stop"], y["early_stop"])], verbose=100)

    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated.fit(X["calibration"], y["calibration"])

    cal_scores = calibrated.predict_proba(X["calibration"])[:, 1]
    cal_amounts = X["calibration"]["amount"].values
    cal_days = b["calibration_end"] - b["early_stop_end"]
    block_t = choose_threshold_by_cost(y["calibration"], cal_scores, cal_amounts, cost)
    review_t = choose_review_threshold(cal_scores, block_t, cal_days, budget)

    test = table.loc[masks["test"]]
    t_scores = calibrated.predict_proba(X["test"])[:, 1]
    t_amounts, t_days = test["amount"].values, day[masks["test"]].values
    rules = rules_baseline(test)

    per_scenario = {}
    for scenario, grp in test[test["is_fraud"] == 1].groupby("fraud_scenario"):
        idx = np.where(masks["test"].values)[0]
        pos = np.searchsorted(idx, grp.index.values)
        per_scenario[scenario] = {"n": int(len(grp)),
                                  "recall_review_or_block": float((t_scores[pos] >= review_t).mean())}

    metrics = {
        "dataset": "synthetic (simulator.py, Fraud Detection Handbook design + ATO and card-testing scenarios)",
        "protocol": "chronological; labels as known at training time (delayed chargebacks); test = later month",
        "late_labels_at_training": late,
        "cost_model": cost.to_dict(),
        "break_even_precision_avg_amount": break_even_precision(float(np.mean(t_amounts)), cost),
        "thresholds": {"review": review_t, "block": block_t},
        "test": {
            "model_block": summarize(y["test"], t_scores, t_amounts, block_t, cost, t_days, budget),
            "model_review_or_block": summarize(y["test"], t_scores, t_amounts, review_t, cost, t_days, budget),
            "rules_only": summarize(y["test"], rules, t_amounts, 0.5, cost, t_days, budget),
            "model_or_rules": summarize(y["test"], np.maximum(t_scores >= review_t, rules).astype(float),
                                        t_amounts, 0.5, cost, t_days, budget),
        },
        "recall_by_scenario": per_scenario,
        "feature_importance": dict(sorted(
            zip(FEATURE_NAMES, map(float, model.feature_importances_)), key=lambda kv: -kv[1])),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    out = scfg.get("model_path", "/app/models/decision_time_model.pkl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    version = datetime.now(timezone.utc).strftime("dt-%Y%m%dT%H%M%SZ")
    joblib.dump({
        "model": calibrated, "model_kind": "decision_time", "model_version": version,
        "feature_names": FEATURE_NAMES, "threshold": review_t, "decision_thresholds": metrics["thresholds"],
        "metrics": metrics, "reference_scores": score_reference(cal_scores), "training_date": metrics["trained_at"],
    }, out)
    results = scfg.get("results_path", os.path.join(os.path.dirname(out), "metrics_decision_time.json"))
    with open(results, "w") as f:
        json.dump(metrics, f, indent=2)

    m = metrics["test"]["model_block"]
    logger.info(f"TEST  AUC-ROC {m['auc_roc']:.4f}  AP {m['average_precision']:.4f}  "
                f"recall@FPR1% {m['recall_at_fpr_0.01']:.3f}  $recall {m['dollar_recall']:.3f}")
    logger.info(f"Saved {out} ({version}) and {results}")
    return {"status": "success", **metrics}


def train_decision_time_model(config_path: str = "/app/config.yaml") -> Dict[str, Any]:
    with open(config_path) as f:
        return train(yaml.safe_load(f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="/app/config.yaml")
    result = train_decision_time_model(parser.parse_args().config)
    print(json.dumps(result["test"]["model_block"], indent=2))
