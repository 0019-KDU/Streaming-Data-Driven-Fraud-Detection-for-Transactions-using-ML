"""
Shared fraud scoring used by BOTH serving paths:

- api_direct.py        synchronous REST scoring (the authorization path)
- main_inference.py    Spark Structured Streaming (post-authorization monitoring)

Pipeline per transaction:
    features (model-specific) -> champion model -> [shadow model, logged only]
    -> velocity + ATO signals (Redis) -> hybrid decision engine + YAML rules
    -> reason codes + human-readable explanation -> audit log

Two model kinds are supported:
- decision_time: features from the online feature store (Redis), computed by the
  same DecisionTimeFeatures code that generated the training data. Default for live.
- ieee_cis: the IEEE-CIS benchmark model with its pickled feature pipeline.
"""

import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# The shared feature code lives in src/dags (also importable by Airflow)
_DAGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dags')
if _DAGS not in sys.path:
    sys.path.append(_DAGS)

from .ato_service import ATOService  # noqa: E402
from .decision_engine import HybridDecisionEngine  # noqa: E402
from .explanations import explain  # noqa: E402
from .idempotency import ResultCache  # noqa: E402
from .logging_utils import setup_logger  # noqa: E402
from .model_loader import ModelLoader  # noqa: E402
from .velocity_service import VelocityService  # noqa: E402

logger = setup_logger(__name__)


class _NoRiskVelocity:
    """Fallback when Redis is unavailable: no velocity signal."""
    velocity_risk = 0.0
    amount_risk = 0.0
    factors: List[str] = []

    def check_velocity(self, card_id, user_id, amount, timestamp=None):
        return self


class _NoRiskATO:
    """Fallback when Redis is unavailable: no ATO signal."""
    ato_risk = 0.0
    ato_detected = False
    factors: List[str] = []

    def check_ato(self, card_id, transaction_data):
        return self


def _present(value) -> bool:
    return value is not None and not (isinstance(value, float) and value != value)


def _booster(model):
    """XGBoost booster inside a (calibrated) sklearn model, or None."""
    inner = model
    if hasattr(model, 'calibrated_classifiers_'):
        cc = model.calibrated_classifiers_[0]
        inner = getattr(cc, 'estimator', None) or getattr(cc, 'base_estimator', None)
    return inner.get_booster() if hasattr(inner, 'get_booster') else None


class FraudScorer:
    """Loads all artifacts once and scores transactions end to end."""

    def __init__(self, config, redis_client=None, serving_path: str = "sync_api"):
        self.config = config
        self.serving_path = serving_path

        self.model_loader = ModelLoader(config)
        self.model_loader.load()
        self.metadata = self.model_loader.get_metadata()
        self.model_kind = self.model_loader.model_kind
        self.feature_pipeline = self.model_loader.feature_pipeline

        self.redis_enabled = redis_client is not None
        self.result_cache = ResultCache(redis_client)
        if self.redis_enabled:
            self.velocity_service = VelocityService(redis_client, config)
            self.ato_service = ATOService(redis_client, config)
        else:
            logger.warning("Redis unavailable: velocity/ATO signals disabled; decision-time history is in-process only")
            self.velocity_service = _NoRiskVelocity()
            self.ato_service = _NoRiskATO()

        self.dt_features = None
        if self.model_kind == 'decision_time':
            from decision_time_features import DecisionTimeFeatures, MemoryStore
            from .feature_store import RedisStore
            store = RedisStore(redis_client) if redis_client is not None else MemoryStore()
            self.dt_features = DecisionTimeFeatures(store)

        self.shadow = self._load_shadow()
        self.decision_engine = HybridDecisionEngine(config, self.metadata)
        self.booster = _booster(self.model_loader.model)

        self.audit = None
        dsn = getattr(getattr(config, 'serving', None), 'decision_log_dsn', '')
        if dsn:
            from .audit_log import DecisionLogger
            self.audit = DecisionLogger(dsn)

    # ------------------------------------------------------------------ setup
    def _load_shadow(self) -> Optional[ModelLoader]:
        path = getattr(getattr(self.config, 'serving', None), 'shadow_model_bundle_path', '')
        if not path:
            return None
        try:
            shadow = ModelLoader(self.config, bundle_path=path)
            shadow.load()
        except Exception as exc:  # noqa: BLE001 - a broken challenger must not stop serving
            logger.error(f"Shadow model not loaded ({exc}); continuing without it")
            return None
        if shadow.model_kind != self.model_kind:
            logger.error(f"Shadow model kind {shadow.model_kind} != champion {self.model_kind}; disabled")
            return None
        logger.info(f"Shadow model {shadow.get_metadata()['model_version']} active (scores logged, never acted on)")
        return shadow

    # --------------------------------------------------------------- features
    def _features(self, transactions: List[Dict[str, Any]]) -> pd.DataFrame:
        if self.model_kind == 'decision_time':
            # Sequential on purpose: each payment must see the ones before it
            rows = [self.dt_features.process(t) for t in transactions]
            return pd.DataFrame(rows, columns=self.model_loader.feature_names)
        return self.feature_pipeline.transform(pd.DataFrame(transactions))

    def _model_reasons(self, features: pd.DataFrame, k: int = 3) -> List[List[str]]:
        """Top positive per-feature contributions (XGBoost TreeSHAP) as reason codes."""
        if self.booster is None:
            return [[] for _ in range(len(features))]
        try:
            import xgboost as xgb
            contribs = self.booster.predict(xgb.DMatrix(features[self.model_loader.feature_names]),
                                            pred_contribs=True)[:, :-1]
        except Exception as exc:  # noqa: BLE001 - explanations must never fail a decision
            logger.debug(f"pred_contribs failed: {exc}")
            return [[] for _ in range(len(features))]
        names = self.model_loader.feature_names
        out = []
        for row in contribs:
            top = [i for i in np.argsort(-row)[:k] if row[i] > 0]
            out.append([f"model:{names[i]}" for i in top])
        return out

    # ---------------------------------------------------------------- scoring
    @staticmethod
    def _txn_id(txn: Dict[str, Any]) -> Optional[str]:
        tid = txn.get('TransactionID') or txn.get('transaction_id')
        return str(tid) if tid not in (None, '') else None

    def score_batch(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score a list of raw transaction dicts (IEEE-CIS or decision-time payloads).

        Idempotent: a transaction ID that was already scored (Kafka/Spark replay,
        client retry) returns its original decision and is NOT recorded again in
        the velocity / feature history.
        """
        if not transactions:
            return []
        results: List[Optional[Dict[str, Any]]] = [None] * len(transactions)
        fresh_idx = []
        for i, txn in enumerate(transactions):
            cached = self.result_cache.get(self._txn_id(txn))
            if cached is not None:
                results[i] = dict(cached, replayed=True)
            else:
                fresh_idx.append(i)
        if fresh_idx:
            scored = self._score_fresh([transactions[i] for i in fresh_idx])
            for i, result in zip(fresh_idx, scored):
                results[i] = result
        return results  # type: ignore[return-value]

    def _score_fresh(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        start = time.time()
        transactions = [dict(t) for t in transactions]
        for txn in transactions:
            if _present(txn.get('TransactionAmt')):
                txn['TransactionAmt'] = float(txn['TransactionAmt'])

        features = self._features(transactions)
        probabilities = [float(p) for p in self.model_loader.predict(features)]
        reasons = self._model_reasons(features)

        shadow_probs = [-1.0] * len(transactions)
        if self.shadow is not None:
            try:
                shadow_probs = [float(p) for p in self.shadow.predict(features)]
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Shadow scoring failed: {exc}")
        model_ms = (time.time() - start) * 1000

        feature_rows = features.to_dict(orient='records')
        results = []
        for i, txn in enumerate(transactions):
            result = self._decide(txn, probabilities[i], reasons[i], shadow_probs[i])
            self.result_cache.put(self._txn_id(txn), result)
            if self.audit is not None:
                self.audit.log(result, txn, feature_rows[i], self.serving_path)
            results.append(result)

        logger.info(f"Scored {len(transactions)} txn(s): features+model {model_ms:.1f} ms, "
                    f"total {(time.time() - start) * 1000:.1f} ms")
        return results

    def score_one(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        return self.score_batch([transaction])[0]

    def record_feedback(self, terminal_id: str, tx_datetime, is_fraud: bool) -> None:
        """Confirmed label (chargeback / analyst decision) for a past transaction."""
        if self.dt_features is not None:
            self.dt_features.record_label(str(terminal_id), tx_datetime, bool(is_fraud))

    def _decide(self, txn: Dict[str, Any], fraud_prob: float, model_reasons: List[str],
                shadow_prob: float) -> Dict[str, Any]:
        card1 = str(txn.get('card1')) if _present(txn.get('card1')) else str(txn.get('customer_id', 'unknown'))
        uid = f"{card1}_{txn.get('addr1', 'na')}_{txn.get('P_emaildomain', 'na')}"
        amount = float(txn['TransactionAmt']) if _present(txn.get('TransactionAmt')) else 0.0
        now = time.time()
        txn.setdefault('dt_hour', datetime.now(timezone.utc).hour)  # enables night-time rules

        velocity = self.velocity_service.check_velocity(card1, uid, amount, now)
        ato = self.ato_service.check_ato(card1, txn)
        decision = self.decision_engine.make_decision(
            fraud_probability=fraud_prob,
            velocity_risk=velocity.velocity_risk,
            amount_risk=velocity.amount_risk,
            ato_risk=ato.ato_risk,
            ato_detected=ato.ato_detected,
            transaction_data=txn,
            velocity_factors=list(velocity.factors),
            ato_factors=list(ato.factors),
        )

        risk_factors = list(decision.risk_factors)
        if ato.ato_detected:
            risk_factors.insert(0, f"ACCOUNT_TAKEOVER_{'CRITICAL' if ato.ato_risk >= 0.8 else 'HIGH'}")
        if decision.decision.value != "APPROVE":
            risk_factors.extend(model_reasons)

        shadow_meta = self.shadow.get_metadata() if self.shadow is not None else {}
        shadow_threshold = shadow_meta.get('threshold', 1.0)
        return {
            'transaction_id': str(txn.get('TransactionID') or txn.get('transaction_id') or f"TXN_{int(now * 1000)}"),
            'fraud_probability': fraud_prob,
            'adjusted_probability': float(decision.adjusted_probability),
            'decision': decision.decision.value,
            'risk_level': decision.risk_level.value,
            'risk_factors': ",".join(risk_factors),
            'explanation': " | ".join(explain(decision.decision.value, risk_factors)),
            'velocity_risk': float(velocity.velocity_risk),
            'amount_risk': float(velocity.amount_risk),
            'ato_risk': float(ato.ato_risk),
            'base_threshold': float(self.decision_engine.base_threshold),
            'hybrid_threshold': float(decision.hybrid_threshold),
            'model_kind': self.model_kind,
            'model_version': str(self.metadata.get('model_version', 'unknown')),
            'rules_version': self.decision_engine.rule_engine.version,
            'shadow_model_version': str(shadow_meta.get('model_version', '')),
            'shadow_probability': float(shadow_prob),
            'shadow_decision': ('FLAG' if shadow_prob >= shadow_threshold else 'PASS') if shadow_prob >= 0 else '',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'TransactionAmt': amount,
            'ProductCD': str(txn.get('ProductCD', 'N/A')),
            'card1': card1,
            'card4': str(txn.get('card4', 'N/A')),
            'card6': str(txn.get('card6', 'N/A')),
            'P_emaildomain': str(txn.get('P_emaildomain', 'N/A')),
            'terminal_id': str(txn.get('terminal_id', 'N/A')),
        }


def rules_only_decision(engine: HybridDecisionEngine, txn: Dict[str, Any], mode: str = "review",
                        reason: str = "model_unavailable") -> Dict[str, Any]:
    """
    Degraded-mode decision when the model is down or misses its latency budget.

    Hard rules still apply (a BLOCK rule still blocks). Everything else gets the
    configured fallback: "review" (fail safe: manual review queue) or "approve"
    (fail open: keep checkout working, accept more fraud risk). Responses are
    marked degraded so the outage is visible in logs, metrics and the audit trail.
    """
    txn = dict(txn)
    txn.setdefault('dt_hour', datetime.now(timezone.utc).hour)
    triggered, factors, action = engine._check_rules(txn)
    if action == "BLOCK":
        decision, risk = "BLOCK", "HIGH"
    elif triggered or mode != "approve":  # unknown mode fails safe
        decision, risk = "REVIEW", "MEDIUM"
    else:
        decision, risk = "APPROVE", "LOW"
    factors = [f"degraded_{reason}"] + factors
    amount = float(txn['TransactionAmt']) if _present(txn.get('TransactionAmt')) else 0.0
    return {
        'transaction_id': str(txn.get('TransactionID') or txn.get('transaction_id') or ''),
        'fraud_probability': -1.0, 'adjusted_probability': -1.0,
        'decision': decision, 'risk_level': risk,
        'risk_factors': ",".join(factors),
        'explanation': " | ".join(explain(decision, factors[1:]) or ["Scored by rules only (model unavailable)"]),
        'velocity_risk': 0.0, 'amount_risk': 0.0, 'ato_risk': 0.0,
        'base_threshold': float(engine.base_threshold), 'hybrid_threshold': -1.0,
        'model_kind': 'rules_only', 'model_version': 'none', 'rules_version': engine.rule_engine.version,
        'shadow_model_version': '', 'shadow_probability': -1.0, 'shadow_decision': '',
        'timestamp': datetime.now(timezone.utc).isoformat(), 'degraded': True,
        'TransactionAmt': amount, 'ProductCD': str(txn.get('ProductCD', 'N/A')),
        'card1': str(txn.get('card1', txn.get('customer_id', 'unknown'))),
        'card4': str(txn.get('card4', 'N/A')), 'card6': str(txn.get('card6', 'N/A')),
        'P_emaildomain': str(txn.get('P_emaildomain', 'N/A')), 'terminal_id': str(txn.get('terminal_id', 'N/A')),
    }


def connect_redis_or_none(config) -> Optional[Any]:
    """Return a Redis client, or None if Redis cannot be reached."""
    try:
        from .utils.redis_client import get_redis_client
        return get_redis_client(config)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Redis unavailable ({exc}); continuing without velocity/ATO")
        return None
