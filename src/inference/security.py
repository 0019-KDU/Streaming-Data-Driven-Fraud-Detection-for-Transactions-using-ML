"""
API-key authentication for the scoring API.

Callers send `X-API-Key: <key>`. Valid keys come from the comma-separated
environment variable SCORING_API_KEYS. If it is empty the API runs open (local
development only) and logs a warning at startup. Keys are compared in constant
time so response timing does not reveal how much of a key was correct.
"""

import hmac
import os
from typing import List, Optional


def load_keys(env_var: str = "SCORING_API_KEYS") -> List[str]:
    return [k.strip() for k in os.getenv(env_var, "").split(",") if k.strip()]


def is_authorized(provided: Optional[str], keys: List[str]) -> bool:
    if not keys:
        return True  # auth disabled (development)
    if not provided:
        return False
    return any(hmac.compare_digest(provided.encode(), k.encode()) for k in keys)
