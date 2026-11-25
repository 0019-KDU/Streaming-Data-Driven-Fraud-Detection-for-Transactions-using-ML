"""
Train/serve parity for the shared feature pipeline (src/dags/feature_pipeline.py).

Regression guard for the incident in docs/postmortem-train-serve-skew.md: the real
IEEECISFeaturePipeline was replaced by a no-op stub, inference silently produced
8 of 88 features, and every prediction failed.
"""

import joblib
import numpy as np
import pandas as pd

from conftest import MODEL_PATH, PIPELINE_PATH, requires_artifacts


def test_pipeline_class_is_the_real_implementation():
    import feature_pipeline

    pipeline = feature_pipeline.IEEECISFeaturePipeline()
    pipeline.feature_names = ["log_TransactionAmt", "email_risky", "card_is_discover"]
    out = pipeline.transform(pd.DataFrame([{"TransactionAmt": 99.0, "P_emaildomain": "tempmail.com", "card4": "discover"}]))

    # A pass-through stub would return the raw input columns unchanged
    assert list(out.columns) == pipeline.feature_names
    assert out.loc[0, "log_TransactionAmt"] == np.float32(np.log1p(99.0))
    assert out.loc[0, "email_risky"] == 1
    assert out.loc[0, "card_is_discover"] == 1


@requires_artifacts
def test_pipeline_produces_exactly_the_model_features(full_transaction, api_transaction):
    pipeline = joblib.load(PIPELINE_PATH)
    bundle = joblib.load(MODEL_PATH)
    expected = list(bundle["feature_names"])

    assert list(pipeline.feature_names) == expected
    for payload in (full_transaction, api_transaction):
        out = pipeline.transform(pd.DataFrame([payload]))
        assert list(out.columns) == expected, "feature names/order must match training"
        # Only raw pass-through columns may be NaN (XGBoost treats NaN as "missing")
        from feature_pipeline import RAW_PASSTHROUGH
        engineered = [c for c in expected if c not in RAW_PASSTHROUGH]
        assert not out[engineered].isna().any().any(), "engineered features must always be filled"


@requires_artifacts
def test_features_do_not_depend_on_batch_composition(full_transaction, api_transaction):
    """The same row must get the same features alone (REST API) or in a batch (Spark)."""
    pipeline = joblib.load(PIPELINE_PATH)
    same_card = dict(api_transaction, TransactionID="TXN_TEST_002")

    alone = pipeline.transform(pd.DataFrame([api_transaction]))
    batched = pipeline.transform(pd.DataFrame([api_transaction, same_card, full_transaction]))

    pd.testing.assert_frame_equal(
        alone.reset_index(drop=True),
        batched.iloc[[0]].reset_index(drop=True),
        check_dtype=False,
    )
