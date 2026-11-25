"""
Declarative, hot-reloadable rule engine.

Rules live in rules.yaml so analysts can add an emergency rule (for example
"block this terminal") in under a minute without a code change, a rebuild or a
model retrain. The file is re-read when it changes; a file that fails
validation is rejected and the last good rule set stays active.

Condition grammar (nestable):
    {field: TransactionAmt, op: ">=", value: 10000}
    {all: [cond, ...]}   {any: [cond, ...]}   {not: cond}
Operators: >= > <= < == != in not_in contains
Actions:   REVIEW (send to an analyst) | BLOCK (decline immediately)
"""

import operator
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

from .logging_utils import setup_logger

logger = setup_logger(__name__)

OPS = {
    '>=': operator.ge, '>': operator.gt, '<=': operator.le, '<': operator.lt,
    '==': operator.eq, '!=': operator.ne,
    'in': lambda a, b: a in b, 'not_in': lambda a, b: a not in b,
    'contains': lambda a, b: str(b).lower() in str(a).lower(),
}
ACTIONS = {'REVIEW', 'BLOCK'}
DEFAULT_RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rules.yaml')


@dataclass
class RuleHit:
    rule_id: str
    action: str   # REVIEW | BLOCK
    factor: str   # machine-readable reason code, e.g. "risky_email_tempmail.com"


class _Missing(dict):
    def __missing__(self, key):
        return ''


class RuleEngine:
    def __init__(self, path: Optional[str] = None, reload_interval: float = 5.0):
        self.path = path or DEFAULT_RULES_PATH
        self.reload_interval = reload_interval
        self.rules: List[Dict[str, Any]] = []
        self.version: str = 'none'
        self._mtime: Optional[float] = None
        self._checked = 0.0
        self._lock = threading.Lock()
        self._load(initial=True)

    # ---------------------------------------------------------------- loading
    def _load(self, initial: bool = False) -> None:
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            if initial:
                logger.warning(f"Rules file not found: {self.path} (no rules active)")
            return
        if mtime == self._mtime:
            return
        try:
            with open(self.path) as f:
                doc = yaml.safe_load(f) or {}
            rules = [r for r in doc.get('rules', []) if r.get('enabled', True)]
            for r in rules:
                self._validate(r)
        except Exception as exc:  # noqa: BLE001 - keep serving with the last good rules
            logger.error(f"Rejected rules file {self.path}: {exc} (keeping version {self.version})")
            self._mtime = mtime
            return
        with self._lock:
            self.rules, self._mtime = rules, mtime
            self.version = str(doc.get('version', int(mtime)))
        logger.info(f"Loaded {len(rules)} rules (version {self.version}) from {self.path}")

    def _validate(self, rule: Dict[str, Any]) -> None:
        if not rule.get('id'):
            raise ValueError("rule without id")
        if rule.get('action', 'REVIEW') not in ACTIONS:
            raise ValueError(f"rule {rule['id']}: action must be one of {sorted(ACTIONS)}")
        self._validate_cond(rule.get('when'), rule['id'])

    def _validate_cond(self, cond, rule_id: str) -> None:
        if not isinstance(cond, dict):
            raise ValueError(f"rule {rule_id}: condition must be a mapping")
        if 'all' in cond or 'any' in cond:
            for c in cond.get('all', cond.get('any')):
                self._validate_cond(c, rule_id)
        elif 'not' in cond:
            self._validate_cond(cond['not'], rule_id)
        elif cond.get('op') not in OPS or 'field' not in cond:
            raise ValueError(f"rule {rule_id}: bad condition {cond}")

    def maybe_reload(self) -> None:
        now = time.monotonic()
        if now - self._checked >= self.reload_interval:
            self._checked = now
            self._load()

    # ------------------------------------------------------------- evaluation
    def _match(self, cond: Dict[str, Any], txn: Dict[str, Any]) -> bool:
        if 'all' in cond:
            return all(self._match(c, txn) for c in cond['all'])
        if 'any' in cond:
            return any(self._match(c, txn) for c in cond['any'])
        if 'not' in cond:
            return not self._match(cond['not'], txn)
        value = txn.get(cond['field'])
        if value is None or (isinstance(value, float) and value != value):
            return False
        try:
            return bool(OPS[cond['op']](value, cond['value']))
        except TypeError:
            return False  # e.g. comparing a string field with a number

    def evaluate(self, txn: Dict[str, Any]) -> List[RuleHit]:
        self.maybe_reload()
        with self._lock:
            rules = list(self.rules)
        hits = []
        for rule in rules:
            if self._match(rule['when'], txn):
                try:
                    factor = rule.get('reason', rule['id']).format_map(_Missing(txn))
                except (ValueError, KeyError, IndexError):
                    factor = rule['id']
                hits.append(RuleHit(rule['id'], rule.get('action', 'REVIEW'), factor))
        return hits
