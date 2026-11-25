"""YAML rule engine: matching, hot reload, rejection of bad files, BLOCK rules."""

import os
import time

import pytest

from inference.config import Config
from inference.decision_engine import Decision, HybridDecisionEngine
from inference.rule_engine import RuleEngine

RULES_V1 = """
version: 1
rules:
  - id: big
    when: {field: TransactionAmt, op: ">=", value: 1000}
    action: REVIEW
    reason: "big_{TransactionAmt:.0f}"
"""
RULES_V2 = RULES_V1 + """
  - id: block_terminal
    when:
      all:
        - {field: terminal_id, op: in, value: ["1234"]}
        - not: {field: TransactionAmt, op: "<", value: 1}
    action: BLOCK
    reason: "emergency_block_terminal_{terminal_id}"
"""


def write(path, text):
    path.write_text(text)
    t = time.time() + 1  # make sure the mtime changes even on coarse filesystems
    os.utime(path, (t, t))


@pytest.fixture
def rules_file(tmp_path):
    p = tmp_path / "rules.yaml"
    write(p, RULES_V1)
    return p


def test_matches_and_formats_reason(rules_file):
    engine = RuleEngine(str(rules_file), reload_interval=0)
    hits = engine.evaluate({"TransactionAmt": 1500.0})
    assert [(h.rule_id, h.action, h.factor) for h in hits] == [("big", "REVIEW", "big_1500")]
    assert engine.evaluate({"TransactionAmt": 10.0}) == []
    assert engine.evaluate({}) == []  # missing field never matches


def test_hot_reload_adds_emergency_rule(rules_file):
    engine = RuleEngine(str(rules_file), reload_interval=0)
    assert engine.evaluate({"terminal_id": "1234", "TransactionAmt": 5.0}) == []
    write(rules_file, RULES_V2)
    hits = engine.evaluate({"terminal_id": "1234", "TransactionAmt": 5.0})
    assert [h.action for h in hits] == ["BLOCK"]


def test_invalid_file_keeps_last_good_rules(rules_file):
    engine = RuleEngine(str(rules_file), reload_interval=0)
    write(rules_file, "rules:\n  - id: broken\n    when: {field: x, op: '~~', value: 1}\n")
    assert [h.rule_id for h in engine.evaluate({"TransactionAmt": 2000.0})] == ["big"]
    assert engine.version == "1"


def test_block_rule_overrides_low_model_score(rules_file):
    write(rules_file, RULES_V2)
    config = Config()
    config.rules.path = str(rules_file)
    engine = HybridDecisionEngine(config, {"threshold": 0.05})
    result = engine.make_decision(0.001, 0.0, 0.0, 0.0, False, {"terminal_id": "1234", "TransactionAmt": 20.0})
    assert result.decision == Decision.BLOCK
    assert "emergency_block_terminal_1234" in result.risk_factors


def test_default_rules_file_is_valid():
    engine = RuleEngine()
    assert len(engine.rules) >= 4
