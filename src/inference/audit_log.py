"""
Decision audit log: every decision with its inputs, features, scores, thresholds,
reason codes, explanation and model/rule versions, written to Postgres.

Writes happen on a background thread with a bounded queue, so the audit log can
never add latency to (or fail) a payment decision. If the queue is full or the
database is down, records are dropped and counted, and the loss is logged.
Table DDL: src/sql/decision_log.sql (long-term retention would use monthly
partitions moved to cold storage; out of scope here).
"""

import json
import queue
import threading
import time
from typing import Any, Dict, Optional

from .logging_utils import setup_logger

logger = setup_logger(__name__)

INSERT_SQL = """
INSERT INTO decision_log (
    decided_at, transaction_id, serving_path, model_kind, model_version, rules_version,
    decision, risk_level, fraud_probability, adjusted_probability, hybrid_threshold,
    shadow_model_version, shadow_probability, risk_factors, explanation, features, request
) VALUES (%(decided_at)s, %(transaction_id)s, %(serving_path)s, %(model_kind)s, %(model_version)s,
          %(rules_version)s, %(decision)s, %(risk_level)s, %(fraud_probability)s, %(adjusted_probability)s,
          %(hybrid_threshold)s, %(shadow_model_version)s, %(shadow_probability)s, %(risk_factors)s,
          %(explanation)s, %(features)s, %(request)s)
"""


def _json(obj: Any) -> str:
    return json.dumps(obj, default=lambda o: o.item() if hasattr(o, 'item') else str(o))


class DecisionLogger:
    def __init__(self, dsn: str, max_queue: int = 10000, batch_size: int = 200):
        self.dsn = dsn
        self.batch_size = batch_size
        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max_queue)
        self.dropped = 0
        self._conn = None
        threading.Thread(target=self._run, name="decision-log-writer", daemon=True).start()
        logger.info("Decision audit log enabled")

    def log(self, result: Dict[str, Any], request: Dict[str, Any], features: Optional[Dict[str, Any]],
            serving_path: str) -> None:
        record = {
            'decided_at': result['timestamp'], 'transaction_id': result['transaction_id'],
            'serving_path': serving_path, 'model_kind': result.get('model_kind'),
            'model_version': result.get('model_version'), 'rules_version': result.get('rules_version'),
            'decision': result['decision'], 'risk_level': result['risk_level'],
            'fraud_probability': result['fraud_probability'],
            'adjusted_probability': result.get('adjusted_probability'),
            'hybrid_threshold': result.get('hybrid_threshold'),
            'shadow_model_version': result.get('shadow_model_version') or None,
            'shadow_probability': result.get('shadow_probability') if result.get('shadow_probability', -1) >= 0 else None,
            'risk_factors': [f for f in result.get('risk_factors', '').split(',') if f],
            'explanation': result.get('explanation', ''),
            'features': _json(features or {}), 'request': _json(request),
        }
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1
            if self.dropped % 1000 == 1:
                logger.error(f"Decision log queue full: {self.dropped} records dropped so far")

    def _connect(self):
        import psycopg2  # imported lazily: only needed when the audit log is enabled
        conn = psycopg2.connect(self.dsn)
        conn.autocommit = True
        return conn

    def _run(self) -> None:
        while True:
            batch = [self.queue.get()]
            while len(batch) < self.batch_size:
                try:
                    batch.append(self.queue.get_nowait())
                except queue.Empty:
                    break
            try:
                if self._conn is None or self._conn.closed:
                    self._conn = self._connect()
                with self._conn.cursor() as cur:
                    cur.executemany(INSERT_SQL, batch)
            except Exception as exc:  # noqa: BLE001 - never crash the writer
                self.dropped += len(batch)
                logger.error(f"Decision log write failed ({len(batch)} records dropped): {exc}")
                self._conn = None
                time.sleep(2)
