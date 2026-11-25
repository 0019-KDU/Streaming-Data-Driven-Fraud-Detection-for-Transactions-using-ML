"""
Shared pytest fixtures.

Tests that need the trained artifacts (src/models/*.pkl, produced by the Airflow
training DAG and gitignored) are skipped automatically when they are absent.
"""

import importlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MODELS = SRC / "models"
FIXTURES = Path(__file__).parent / "fixtures"

# `inference` package and the top-level `feature_pipeline` / `ieee_cis_training`
# modules referenced by the pickled artifacts.
for p in (SRC, SRC / "dags"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Unpickling the model bundle imports ieee_cis_training, which imports heavy
# training-only libraries at module level. Provide empty stand-ins *only* if they
# are not installed, so the serving-path tests can run in a lightweight env.
for name in ("mlflow", "mlflow.sklearn", "seaborn", "matplotlib", "matplotlib.pyplot"):
    try:
        importlib.import_module(name)
    except ImportError:
        module = types.ModuleType(name)
        module.use = lambda *args, **kwargs: None
        sys.modules[name] = module
if not hasattr(sys.modules["mlflow"], "sklearn"):
    sys.modules["mlflow"].sklearn = sys.modules["mlflow.sklearn"]
if not hasattr(sys.modules["matplotlib"], "pyplot"):
    sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]

MODEL_PATH = MODELS / "fraud_detection_model.pkl"
PIPELINE_PATH = MODELS / "feature_pipeline.pkl"

requires_artifacts = pytest.mark.skipif(
    not (MODEL_PATH.exists() and PIPELINE_PATH.exists()),
    reason="trained artifacts not found in src/models (run the training DAG first)",
)


@pytest.fixture(scope="session")
def config():
    os.environ["MODEL_BUNDLE_PATH"] = str(MODEL_PATH)
    os.environ["FEATURE_PIPELINE_PATH"] = str(PIPELINE_PATH)
    from inference.config import Config
    return Config.load()


@pytest.fixture(scope="session")
def full_transaction():
    """A real IEEE-CIS training row (TransactionID 2987245) with all raw columns."""
    return json.loads((FIXTURES / "ieee_transaction_2987245.json").read_text())


@pytest.fixture
def api_transaction():
    """The ~10-field payload a client actually sends to the producer API."""
    return {
        "TransactionID": "TXN_TEST_001",
        "TransactionDT": 86400,
        "TransactionAmt": 29.0,
        "ProductCD": "W",
        "card1": 2616,
        "card4": "discover",
        "card6": "credit",
        "addr1": 315.0,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
    }


class FakeRedis:
    """Minimal in-memory stand-in for the sorted-set/string commands the services use."""

    def __init__(self):
        self.zsets, self.strings = {}, {}

    def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update(mapping)

    def zcount(self, key, lo, hi):
        return sum(lo <= s <= hi for s in self.zsets.get(key, {}).values())

    def zrangebyscore(self, key, lo, hi, withscores=False):
        return [m for m, s in sorted(self.zsets.get(key, {}).items(), key=lambda kv: kv[1]) if lo <= s <= hi]

    def zremrangebyscore(self, key, lo, hi):
        z = self.zsets.get(key, {})
        for m in [m for m, s in z.items() if lo <= s <= hi]:
            del z[m]

    def expire(self, key, ttl):
        return True

    def get(self, key):
        return self.strings.get(key)

    def setex(self, key, ttl, value):
        self.strings[key] = value

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.strings:
            return None
        self.strings[key] = value
        return True


@pytest.fixture
def fake_redis():
    return FakeRedis()
