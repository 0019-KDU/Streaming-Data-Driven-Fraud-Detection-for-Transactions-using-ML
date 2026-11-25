"""
Redis implementation of the feature-store interface used by DecisionTimeFeatures
(src/dags/decision_time_features.py). The in-memory MemoryStore in that module
is the offline twin; both expose add / window / trim with identical semantics,
so the live model receives exactly the features it was trained on.
"""

import uuid
from typing import List, Tuple


class RedisStore:
    """Events are sorted-set members '<ts>|<value>|<tag>|<id>' scored by timestamp."""

    def __init__(self, redis_client, prefix: str = "dt", ttl_seconds: int = 32 * 86400):
        self.redis = redis_client
        self.prefix = prefix
        self.ttl = ttl_seconds

    def _key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    def add(self, key: str, ts: float, value: float = 1.0, tag: str = '') -> None:
        k = self._key(key)
        member = f"{ts:.6f}|{float(value)}|{tag}|{uuid.uuid4().hex[:12]}"
        self.redis.zadd(k, {member: ts})
        self.redis.expire(k, self.ttl)

    def window(self, key: str, start: float, end: float) -> List[Tuple[float, float, str]]:
        """Events with start <= ts < end, oldest first."""
        events = []
        for member in self.redis.zrangebyscore(self._key(key), start, end):
            if isinstance(member, bytes):
                member = member.decode()
            # tag may itself contain '|' (e.g. "device|email"): parse ts/value from the
            # left and the random id from the right
            ts, value, rest = member.split('|', 2)
            tag = rest.rsplit('|', 1)[0]
            ts = float(ts)
            if start <= ts < end:
                events.append((ts, float(value), tag))
        events.sort(key=lambda e: e[0])
        return events

    def trim(self, key: str, before: float) -> None:
        self.redis.zremrangebyscore(self._key(key), 0, before - 1e-6)
