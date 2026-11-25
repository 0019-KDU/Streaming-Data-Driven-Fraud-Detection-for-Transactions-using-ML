"""
Fraud-model evaluation shared by every training pipeline.

Reports what fraud teams actually operate on, not just AUC-ROC:
- average precision (AUC-PR) against the prevalence baseline
- recall at a fixed false-positive rate (operating points)
- precision@k per day for a fixed analyst review budget
- dollar-weighted recall (share of fraudulent money caught)
- net savings of a threshold under an explicit cost model

and chooses decision thresholds from that cost model instead of maximising F1.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


@dataclass
class CostModel:
    """
    Per-transaction economics used to set thresholds.

    Flagging a fraud saves its amount plus the chargeback fee. Flagging a
    legitimate payment loses the merchant margin on it plus a fixed friction
    cost (support contact, churn risk, manual review).
    """
    margin: float = 0.10          # profit share of a legitimate sale
    chargeback_fee: float = 15.0  # fee charged on each chargeback
    friction_cost: float = 2.0    # cost of wrongly declining/reviewing a good customer

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def net_savings(y, flagged, amounts, cost: CostModel) -> float:
    """Money saved versus approving everything."""
    y, flagged, amounts = np.asarray(y), np.asarray(flagged, bool), np.asarray(amounts, float)
    caught = (amounts + cost.chargeback_fee)[flagged & (y == 1)].sum()
    lost = (cost.margin * amounts + cost.friction_cost)[flagged & (y == 0)].sum()
    return float(caught - lost)


def break_even_precision(avg_amount: float, cost: CostModel) -> float:
    """Precision at which flagging an average transaction stops paying off."""
    fp = cost.margin * avg_amount + cost.friction_cost
    tp = avg_amount + cost.chargeback_fee
    return float(fp / (tp + fp))


def choose_threshold_by_cost(y, scores, amounts, cost: CostModel, grid: int = 400) -> float:
    """Threshold that maximises net savings on the given (calibration) data."""
    scores = np.asarray(scores, float)
    candidates = np.unique(np.quantile(scores, np.linspace(0.5, 0.9999, grid)))
    best_t, best = 1.0, 0.0
    for t in candidates:
        value = net_savings(y, scores >= t, amounts, cost)
        if value > best:
            best_t, best = float(t), value
    return best_t


def choose_review_threshold(scores, block_threshold: float, n_days: float, daily_review_budget: int) -> float:
    """
    Lower threshold for the REVIEW band so that, on average, the number of
    transactions scored in [review, block) per day fits the analyst budget.
    """
    scores = np.asarray(scores, float)
    below = np.sort(scores[scores < block_threshold])[::-1]
    allowed = int(daily_review_budget * max(n_days, 1.0))
    if allowed <= 0 or below.size == 0:
        return block_threshold
    return float(below[min(allowed, below.size) - 1])


def recall_at_fpr(y, scores, fpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y, scores)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def precision_at_k_per_day(y, scores, days, k: int) -> float:
    """Mean precision of the top-k scored transactions of each day (alert budget)."""
    y, scores, days = np.asarray(y), np.asarray(scores, float), np.asarray(days)
    values = []
    for d in np.unique(days):
        idx = np.where(days == d)[0]
        top = idx[np.argsort(-scores[idx])[:k]]
        if top.size:
            values.append(y[top].mean())
    return float(np.mean(values)) if values else 0.0


def summarize(
    y,
    scores,
    amounts,
    threshold: float,
    cost: CostModel,
    days: Optional[Iterable] = None,
    k: int = 100,
    fpr_targets=(0.001, 0.005, 0.01),
) -> Dict[str, float]:
    """All headline metrics for one split at one operating threshold."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, float)
    amounts = np.asarray(amounts, float)
    flagged = scores >= threshold

    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    tn = int((~flagged & (y == 0)).sum())
    fraud_amount = amounts[y == 1].sum()

    out = {
        "n_transactions": int(y.size),
        "n_fraud": int(y.sum()),
        "prevalence": float(y.mean()),
        "auc_roc": float(roc_auc_score(y, scores)) if 0 < y.sum() < y.size else float("nan"),
        "average_precision": float(average_precision_score(y, scores)) if y.sum() else float("nan"),
        "threshold": float(threshold),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "dollar_recall": float(amounts[flagged & (y == 1)].sum() / fraud_amount) if fraud_amount else 0.0,
        "net_savings": net_savings(y, flagged, amounts, cost),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    for target in fpr_targets:
        out[f"recall_at_fpr_{target:g}"] = recall_at_fpr(y, scores, target)
    if days is not None:
        out[f"precision_at_{k}_per_day"] = precision_at_k_per_day(y, scores, days, k)
    return out


def score_reference(scores, n_bins: int = 10) -> Dict[str, list]:
    """Score quantile bins on calibration data: the baseline for live PSI drift monitoring."""
    scores = np.asarray(scores, float)
    edges = np.unique(np.quantile(scores, np.linspace(0, 1, n_bins + 1)))
    counts, _ = np.histogram(scores, bins=edges)
    return {"bin_edges": edges.tolist(), "proportions": (counts / max(counts.sum(), 1)).tolist()}


def psi(expected_proportions, actual_scores, bin_edges, eps: float = 1e-4) -> float:
    """Population Stability Index of live scores against the training reference."""
    counts, _ = np.histogram(np.clip(actual_scores, bin_edges[0], bin_edges[-1]), bins=bin_edges)
    actual = counts / max(counts.sum(), 1)
    expected = np.asarray(expected_proportions, float)
    actual, expected = np.clip(actual, eps, None), np.clip(expected, eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))
