"""
IEEE-CIS benchmark training: honest, out-of-time evaluation.

Protocol (everything chronological by TransactionDT, no shuffling):

    | train | early-stop | calibration + thresholds | gap | test |

- The feature pipeline is fitted on the TRAIN period only; every other split
  (and live traffic) goes through the same `transform()`. No aggregate is ever
  computed over future data.
- Target encodings on train rows are out-of-fold.
- No SMOTE: class imbalance is handled with `scale_pos_weight`; resampling
  distorts calibrated probabilities and does not help gradient boosting.
- Calibration and threshold selection use their own split; the test split is
  touched exactly once, for the final report.
- Thresholds come from a cost model (fraud loss vs. friction), plus a REVIEW band
  sized to an analyst budget, not from F1.
- Results are compared with a rules-only baseline on the same test split.

Run:  python ieee_cis_training.py --config /app/config.yaml
"""

import argparse
import json
import logging
import os
import sys
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluation import (  # noqa: E402
    CostModel, break_even_precision, choose_review_threshold, choose_threshold_by_cost, score_reference,
    summarize,
)
from feature_pipeline import RISKY_DOMAINS, IEEECISFeaturePipeline  # noqa: E402

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RNG = 42
SECONDS_PER_DAY = 24 * 60 * 60


class AdaptiveThresholdSystem:
    """
    Kept only so that artifacts trained by earlier versions of this module can
    still be unpickled (they reference `ieee_cis_training.AdaptiveThresholdSystem`).
    New artifacts store plain thresholds instead.
    """

    def __init__(self, base_threshold=0.5, window_size=1000, min_threshold=0.1, max_threshold=0.9):
        self.base_threshold = base_threshold
        self.window_size = window_size
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.recent_predictions = deque(maxlen=window_size)
        self.recent_true_labels = deque(maxlen=window_size)
        self.current_threshold = base_threshold
        self.threshold_history = []

    def get_threshold(self, velocity_risk=0.0):
        adjusted = self.current_threshold
        if velocity_risk > 0.8:
            adjusted *= 0.9
        elif velocity_risk > 0.6:
            adjusted *= 0.95
        return float(np.clip(adjusted, self.min_threshold, self.max_threshold))


# --------------------------------------------------------------------------- data
def load_data(config: Dict[str, Any]) -> pd.DataFrame:
    paths = config["data"]["ieee_cis"]
    df = pd.read_csv(paths["train_transaction_path"])
    logger.info(f"Loaded {len(df):,} transactions")
    identity_path = paths.get("train_identity_path")
    if identity_path and os.path.exists(identity_path):
        df = df.merge(pd.read_csv(identity_path), on="TransactionID", how="left")
        logger.info(f"Merged identity table: {df.shape[1]} columns")
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    logger.info(f"Fraud rate: {df['isFraud'].mean():.4%}")
    return df


def chronological_split(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Split by time into train / early_stop / calibration / test with a gap before test."""
    f_train = cfg.get("train_fraction", 0.65)
    f_es = cfg.get("early_stop_fraction", 0.10)
    f_cal = cfg.get("calibration_fraction", 0.10)
    gap = cfg.get("gap_days", 7) * SECONDS_PER_DAY

    t = df["TransactionDT"]
    q1, q2, q3 = t.quantile([f_train, f_train + f_es, f_train + f_es + f_cal]).tolist()
    splits = {
        "train": df[t < q1],
        "early_stop": df[(t >= q1) & (t < q2)],
        "calibration": df[(t >= q2) & (t < q3)],
        "test": df[t >= q3 + gap],
    }
    for name, part in splits.items():
        days = (part["TransactionDT"].max() - part["TransactionDT"].min()) / SECONDS_PER_DAY
        logger.info(f"  {name:<12} {len(part):>8,} rows  {days:5.1f} days  fraud {part['isFraud'].mean():.3%}")
    return {k: v.reset_index(drop=True) for k, v in splits.items()}


def rules_baseline(df: pd.DataFrame) -> np.ndarray:
    """The decision engine's hard rules, applied offline: 1 if any rule fires."""
    amount = pd.to_numeric(df["TransactionAmt"], errors="coerce").fillna(0)
    email = df["P_emaildomain"].astype(str).str.lower()
    hour = (df["TransactionDT"] // 3600) % 24
    fired = (
        (amount >= 10000)
        | email.isin(RISKY_DOMAINS)
        | (((hour >= 22) | (hour <= 5)) & (amount >= 1000))
        | ((df["ProductCD"] == "W") & (amount >= 5000))
    )
    return fired.astype(float).values


# ------------------------------------------------------------------------ training
def train_model(X_tr, y_tr, X_es, y_es, cfg: Dict[str, Any]):
    if XGBClassifier is None:
        raise ImportError("xgboost is required for training")
    params = dict(cfg.get("xgboost", {}))
    spw = params.pop("scale_pos_weight", None) or float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1.0)
    model = XGBClassifier(
        n_estimators=params.pop("n_estimators", 2000),
        learning_rate=params.pop("learning_rate", 0.03),
        max_depth=params.pop("max_depth", 8),
        subsample=params.pop("subsample", 0.8),
        colsample_bytree=params.pop("colsample_bytree", 0.5),
        min_child_weight=params.pop("min_child_weight", 5),
        scale_pos_weight=spw,
        eval_metric="aucpr",
        early_stopping_rounds=params.pop("early_stopping_rounds", 100),
        tree_method="hist",
        random_state=RNG,
        n_jobs=-1,
        **params,
    )
    logger.info(f"Training XGBoost (scale_pos_weight={spw:.1f}) with early stopping on the early-stop split")
    model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=100)
    logger.info(f"  best iteration: {model.best_iteration}")
    return model


def train_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    tcfg = config.get("training", {})
    cost = CostModel(**tcfg.get("cost_model", {}))
    budget = int(tcfg.get("daily_review_budget", 100))

    df = load_data(config)
    splits = chronological_split(df, tcfg.get("split", {}))

    pipeline = IEEECISFeaturePipeline()
    X = {"train": pipeline.fit_transform(splits["train"])}
    for name in ("early_stop", "calibration", "test"):
        X[name] = pipeline.transform(splits[name])
    y = {name: part["isFraud"].astype(int).values for name, part in splits.items()}
    logger.info(f"Features: {len(pipeline.feature_names)}")

    raw_model = train_model(X["train"], y["train"], X["early_stop"], y["early_stop"], tcfg)

    calibrated = CalibratedClassifierCV(raw_model, method=tcfg.get("calibration_method", "sigmoid"), cv="prefit")
    calibrated.fit(X["calibration"], y["calibration"])

    cal_scores = calibrated.predict_proba(X["calibration"])[:, 1]
    cal_amounts = splits["calibration"]["TransactionAmt"].values
    cal_days = (splits["calibration"]["TransactionDT"].max() - splits["calibration"]["TransactionDT"].min()) / SECONDS_PER_DAY
    block_t = choose_threshold_by_cost(y["calibration"], cal_scores, cal_amounts, cost)
    review_t = choose_review_threshold(cal_scores, block_t, cal_days, budget)
    logger.info(f"Thresholds from cost model: REVIEW >= {review_t:.4f}, BLOCK >= {block_t:.4f}")

    test = splits["test"]
    test_days = (test["TransactionDT"] // SECONDS_PER_DAY).values
    test_amounts = test["TransactionAmt"].values
    test_scores = calibrated.predict_proba(X["test"])[:, 1]
    rule_scores = rules_baseline(test)

    metrics = {
        "dataset": "IEEE-CIS (train_transaction + train_identity)",
        "protocol": "chronological train/early-stop/calibration/gap/test; pipeline fitted on train only",
        "splits": {k: {"rows": int(len(v)), "fraud_rate": float(v["isFraud"].mean())} for k, v in splits.items()},
        "cost_model": cost.to_dict(),
        "break_even_precision_avg_amount": break_even_precision(float(np.mean(test_amounts)), cost),
        "thresholds": {"review": review_t, "block": block_t},
        "test": {
            "model_block": summarize(y["test"], test_scores, test_amounts, block_t, cost, test_days, budget),
            "model_review_or_block": summarize(y["test"], test_scores, test_amounts, review_t, cost, test_days, budget),
            "rules_only": summarize(y["test"], rule_scores, test_amounts, 0.5, cost, test_days, budget),
            "model_or_rules": summarize(
                y["test"], np.maximum(test_scores >= review_t, rule_scores).astype(float),
                test_amounts, 0.5, cost, test_days, budget),
        },
        "best_iteration": int(getattr(raw_model, "best_iteration", -1) or -1),
        "n_features": len(pipeline.feature_names),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    log_headline(metrics)
    save_artifacts(config, calibrated, pipeline, metrics, score_reference(cal_scores))
    log_to_mlflow(config, metrics)
    return {"status": "success", **metrics}


def log_headline(metrics: Dict[str, Any]) -> None:
    m, r = metrics["test"]["model_block"], metrics["test"]["rules_only"]
    logger.info("=" * 72)
    logger.info("Out-of-time TEST results")
    logger.info(f"  model  AUC-ROC {m['auc_roc']:.4f}  AP {m['average_precision']:.4f} "
                f"(prevalence {m['prevalence']:.4f})")
    logger.info(f"  model  recall@FPR1% {m['recall_at_fpr_0.01']:.3f}  "
                f"precision {m['precision']:.3f}  recall {m['recall']:.3f}  $recall {m['dollar_recall']:.3f}")
    logger.info(f"  rules  precision {r['precision']:.3f}  recall {r['recall']:.3f}  FPR {r['false_positive_rate']:.4f}")
    logger.info("=" * 72)


def save_artifacts(config, model, pipeline, metrics, reference) -> None:
    tcfg = config["training"]
    model_path = tcfg["model_path"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    version = datetime.now(timezone.utc).strftime("ieee-%Y%m%dT%H%M%SZ")
    bundle = {
        "model": model,
        "calibrated_model": model,  # backward compatibility
        "model_kind": "ieee_cis",
        "model_version": version,
        "feature_names": pipeline.feature_names,
        "threshold": metrics["thresholds"]["review"],
        "decision_thresholds": metrics["thresholds"],
        "metrics": metrics,
        "reference_scores": reference,
        "training_date": metrics["trained_at"],
    }
    joblib.dump(bundle, model_path)
    joblib.dump(pipeline, tcfg["feature_pipeline_path"])
    results_path = tcfg.get("results_path", os.path.join(os.path.dirname(model_path), "metrics_ieee_cis.json"))
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved {model_path}, {tcfg['feature_pipeline_path']} and {results_path} (version {version})")


def log_to_mlflow(config, metrics) -> None:
    """Best effort: training must not fail because the tracking server is down."""
    try:
        import mlflow
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["training"].get("experiment_name", "ieee_cis_fraud"))
        with mlflow.start_run(run_name="ieee_cis_out_of_time"):
            mlflow.log_params({f"threshold_{k}": v for k, v in metrics["thresholds"].items()})
            mlflow.log_params({f"cost_{k}": v for k, v in metrics["cost_model"].items()})
            for split, values in metrics["test"].items():
                mlflow.log_metrics({f"{split}.{k}": float(v) for k, v in values.items()
                                    if isinstance(v, (int, float)) and np.isfinite(v)})
            mlflow.log_dict(metrics, "metrics.json")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"MLflow logging skipped: {exc}")


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def train_ieee_cis_model(config_path: str = "/app/config.yaml") -> Dict[str, Any]:
    """Entry point used by the Airflow DAG."""
    return train_pipeline(load_config(config_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="/app/config.yaml")
    args = parser.parse_args()
    result = train_ieee_cis_model(args.config)
    print(json.dumps(result["test"]["model_block"], indent=2))
