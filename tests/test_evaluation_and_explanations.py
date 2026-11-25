"""Evaluation metrics, cost-based thresholds, PSI and explanation templates."""

import numpy as np
import pytest

from evaluation import (
    CostModel, break_even_precision, choose_review_threshold, choose_threshold_by_cost, net_savings,
    precision_at_k_per_day, psi, recall_at_fpr, score_reference, summarize,
)
from inference.explanations import explain, explain_factor


def test_recall_at_fpr_perfect_and_random():
    y = np.array([0] * 990 + [1] * 10)
    assert recall_at_fpr(y, y.astype(float), 0.001) == 1.0
    rng = np.random.default_rng(0)
    assert recall_at_fpr(y, rng.random(1000), 0.01) < 0.5


def test_cost_threshold_prefers_profitable_cut():
    y = np.array([0] * 95 + [1] * 5)
    scores = np.r_[np.linspace(0, 0.5, 95), [0.9, 0.92, 0.95, 0.97, 0.99]]
    amounts = np.full(100, 100.0)
    t = choose_threshold_by_cost(y, scores, amounts, CostModel())
    assert 0.5 < t <= 0.9
    assert net_savings(y, scores >= t, amounts, CostModel()) == pytest.approx(5 * 115.0)


def test_break_even_precision_matches_formula():
    # $26 sale, 8% margin, $15 fee, no friction: 2.08 / (41 + 2.08)
    assert break_even_precision(26.0, CostModel(margin=0.08, chargeback_fee=15.0, friction_cost=0.0)) \
        == pytest.approx(2.08 / 43.08)


def test_review_band_fits_budget():
    scores = np.linspace(0, 1, 1001)
    review = choose_review_threshold(scores, block_threshold=0.9, n_days=10, daily_review_budget=5)
    assert ((scores >= review) & (scores < 0.9)).sum() == 50


def test_precision_at_k_per_day():
    y = np.array([1, 0, 0, 1, 1, 0])
    s = np.array([0.9, 0.8, 0.1, 0.2, 0.95, 0.5])
    days = np.array([0, 0, 0, 1, 1, 1])
    assert precision_at_k_per_day(y, s, days, k=1) == 1.0
    assert precision_at_k_per_day(y, s, days, k=2) == pytest.approx(0.5)


def test_summarize_reports_operating_metrics():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.05).astype(int)
    s = np.clip(y * 0.6 + rng.random(2000) * 0.5, 0, 1)
    out = summarize(y, s, rng.uniform(1, 500, 2000), 0.6, CostModel(), days=np.arange(2000) // 100, k=10)
    for key in ("auc_roc", "average_precision", "recall_at_fpr_0.01", "dollar_recall", "net_savings",
                "precision_at_10_per_day"):
        assert key in out
    assert out["tp"] + out["fn"] == y.sum()


def test_psi_is_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(2)
    base = rng.beta(1, 20, 5000)
    ref = score_reference(base)
    assert psi(ref["proportions"], base, ref["bin_edges"]) < 0.01
    assert psi(ref["proportions"], np.clip(base * 4, 0, 1), ref["bin_edges"]) > 0.25


def test_explanations_only_render_recorded_factors():
    assert explain("APPROVE", []) == []
    assert explain("APPROVE", ["business_hours"]) == []
    text = explain("REVIEW", ["risky_email_tempmail.com", "model:is_new_device_30d", "business_hours"])
    assert text == ["Purchaser used a disposable email domain (tempmail.com)", "Model signal: new device"]
    assert explain_factor("some_unknown_code") == "some unknown code"
    assert explain("BLOCK", []) == ["Flagged by the combined risk score"]
