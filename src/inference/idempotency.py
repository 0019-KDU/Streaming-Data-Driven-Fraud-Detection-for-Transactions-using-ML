"""
Idempotent scoring: the same transaction ID always gets the same decision.

Kafka and Spark deliver at-least-once: after a crash or retry, a micro-batch can
be processed again, and clients retry HTTP calls on timeouts. Without this
cache every replay would be recorded again in the velocity / feature history
(inflating counts and causing false blocks) and could get a different decision.
The first result is stored per transaction ID (Redis, 7-day TTL; in-process
LRU when Redis is unavailable) and returned on any replay.
"""

import json
from collections import OrderedDict
from typing import Any, Dict, Optional


class ResultCache:
    def __init__(self, redis_client=None, ttl_seconds: int = 7 * 86400, max_local: int = 100_000,
                 prefix: str = "decision"):
        self.redis = redis_client
        self.ttl = ttl_seconds
        self.max_local = max_local
        self.prefix = prefix
        self._local: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    def _key(self, transaction_id: str) -> str:
        return f"{self.prefix}:{transaction_id}"

    def get(self, transaction_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not transaction_id:
            return None
        if self.redis is not None:
            raw = self.redis.get(self._key(transaction_id))
            if raw is None:
                return None
            return json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        result = self._local.get(transaction_id)
        if result is not None:
            self._local.move_to_end(transaction_id)
        return result

    def put(self, transaction_id: Optional[str], result: Dict[str, Any]) -> None:
        if not transaction_id:
            return
        if self.redis is not None:
            # nx: if two replicas race on the same ID, the first decision wins
            self.redis.set(self._key(transaction_id), json.dumps(result), nx=True, ex=self.ttl)
            return
        self._local.setdefault(transaction_id, result)
        if len(self._local) > self.max_local:
            self._local.popitem(last=False)
