"""Hybrid decision engine: bands, rules and ATO override. No artifacts needed."""

import pytest

from inference.config import Config
from inference.decision_engine import Decision, HybridDecisionEngine


@pytest.fixture
def engine():
    config = Config()  # dataclass defaults: base 0.0564, hold 0.20, block 0.30
    return HybridDecisionEngine(config, {"threshold": 0.0564})


def decide(engine, prob, txn=None, **risks):
    return engine.make_decision(
        fraud_probability=prob,
        velocity_risk=risks.get("velocity", 0.0),
        amount_risk=risks.get("amount", 0.0),
        ato_risk=risks.get("ato", 0.0),
        ato_detected=risks.get("ato_detected", False),
        transaction_data=txn or {"TransactionAmt": 40.0, "P_emaildomain": "gmail.com"},
    )


@pytest.mark.parametrize(
    "prob, expected",
    [(0.01, Decision.APPROVE), (0.08, Decision.REVIEW), (0.25, Decision.HOLD), (0.50, Decision.BLOCK)],
)
def test_probability_bands(engine, prob, expected):
    assert decide(engine, prob).decision == expected


def test_hold_is_reachable_with_default_bands(engine):
    assert engine.hold_threshold < engine.block_threshold


def test_risky_email_rule_forces_review(engine):
    result = decide(engine, 0.01, {"TransactionAmt": 40.0, "P_emaildomain": "tempmail.com"})
    assert result.decision == Decision.REVIEW
    assert result.rule_triggered
    assert "risky_email_tempmail.com" in result.risk_factors


def test_ato_detection_overrides_a_low_model_score(engine):
    normal = decide(engine, 0.01)
    moderate = decide(engine, 0.01, ato=0.65, ato_detected=True)
    critical = decide(engine, 0.01, ato=0.9, ato_detected=True)

    assert normal.decision == Decision.APPROVE
    assert moderate.decision == Decision.REVIEW      # ATO flag alone forces review
    assert critical.decision == Decision.HOLD        # +20% ATO boost lifts 0.01 into the HOLD band
    assert critical.hybrid_threshold < normal.hybrid_threshold


def test_night_rule_fires_at_midnight(engine):
    """hour 0 is falsy; the rule must still treat it as a night transaction."""
    result = decide(engine, 0.01, {"TransactionAmt": 1500.0, "P_emaildomain": "gmail.com", "dt_hour": 0})
    assert any(f.startswith("night_high_amount") for f in result.risk_factors)


def test_missing_amount_does_not_crash(engine):
    assert decide(engine, 0.01, {"TransactionAmt": None}).decision == Decision.APPROVE
