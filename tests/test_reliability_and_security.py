"""Idempotent replays, API-key checks and the rules-only fallback (no artifacts needed)."""

import pytest

from inference.config import Config
from inference.decision_engine import HybridDecisionEngine
from inference.idempotency import ResultCache
from inference.scoring import rules_only_decision
from inference.security import is_authorized


@pytest.mark.parametrize("backend", ["redis", "memory"])
def test_replayed_transaction_returns_first_decision(backend, fake_redis):
    cache = ResultCache(fake_redis if backend == "redis" else None)
    cache.put("T1", {"decision": "BLOCK", "fraud_probability": 0.9})
    cache.put("T1", {"decision": "APPROVE", "fraud_probability": 0.01})  # replay must not overwrite
    assert cache.get("T1") == {"decision": "BLOCK", "fraud_probability": 0.9}
    assert cache.get("unknown") is None
    assert cache.get(None) is None


def test_memory_cache_is_bounded():
    cache = ResultCache(None, max_local=2)
    for i in range(3):
        cache.put(f"T{i}", {"i": i})
    assert cache.get("T0") is None and cache.get("T2") == {"i": 2}


def test_api_key_check():
    assert is_authorized(None, [])                  # auth disabled in development
    assert not is_authorized(None, ["k1"])
    assert not is_authorized("wrong", ["k1"])
    assert is_authorized("k2", ["k1", "k2"])


@pytest.fixture
def engine():
    return HybridDecisionEngine(Config(), {})


def test_fallback_still_applies_hard_rules(engine):
    out = rules_only_decision(engine, {"TransactionAmt": 50.0, "P_emaildomain": "tempmail.com"}, "approve")
    assert out["decision"] == "REVIEW"
    assert out["degraded"] is True
    assert out["risk_factors"].startswith("degraded_model_unavailable")


@pytest.mark.parametrize("mode, expected", [("review", "REVIEW"), ("approve", "APPROVE"), ("garbage", "REVIEW")])
def test_fallback_mode_for_clean_payment(engine, mode, expected):
    out = rules_only_decision(engine, {"TransactionAmt": 20.0, "P_emaildomain": "gmail.com", "dt_hour": 12}, mode)
    assert out["decision"] == expected
    assert out["model_kind"] == "rules_only"
