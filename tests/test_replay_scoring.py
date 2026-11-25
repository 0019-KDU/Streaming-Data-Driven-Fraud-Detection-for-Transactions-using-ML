"""
End-to-end scoring through the shared FraudScorer used by both serving paths
(replaces test_inference_full_payload.py).
"""

from conftest import requires_artifacts

VALID_DECISIONS = {"APPROVE", "REVIEW", "HOLD", "BLOCK"}


@requires_artifacts
def test_full_payload_scores_end_to_end(config, full_transaction):
    from inference.scoring import FraudScorer

    result = FraudScorer(config, redis_client=None).score_one(dict(full_transaction))

    assert result["transaction_id"] == "2987245"
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["decision"] in VALID_DECISIONS
    assert result["TransactionAmt"] == full_transaction["TransactionAmt"]


@requires_artifacts
def test_batch_and_single_scoring_agree(config, full_transaction, api_transaction):
    """Spark scores micro-batches, the REST API scores one row: results must match."""
    from inference.scoring import FraudScorer

    scorer = FraudScorer(config, redis_client=None)
    single = [scorer.score_one(dict(t)) for t in (full_transaction, api_transaction)]
    batch = scorer.score_batch([dict(full_transaction), dict(api_transaction)])

    for s, b in zip(single, batch):
        assert s["fraud_probability"] == b["fraud_probability"]
        assert s["decision"] == b["decision"]


@requires_artifacts
def test_risky_email_is_flagged_even_with_low_model_score(config, api_transaction):
    from inference.scoring import FraudScorer

    scorer = FraudScorer(config, redis_client=None)
    result = scorer.score_one(dict(api_transaction, P_emaildomain="tempmail.com"))

    assert result["decision"] != "APPROVE"
    assert "risky_email_tempmail.com" in result["risk_factors"]
