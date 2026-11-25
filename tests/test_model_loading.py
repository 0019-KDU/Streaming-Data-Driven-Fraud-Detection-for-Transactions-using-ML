"""Model bundle loading and prediction contract (replaces verify_xgboost_integration.py)."""

import pandas as pd
import pytest

from conftest import requires_artifacts


@requires_artifacts
def test_bundle_loads_calibrated_xgboost(config):
    from inference.model_loader import ModelLoader

    loader = ModelLoader(config)
    loader.load()

    assert loader.model is not None
    assert type(loader.model).__name__ == "CalibratedClassifierCV"
    assert loader.model_kind == "ieee_cis"
    assert loader.get_feature_names() == list(loader.feature_pipeline.feature_names)
    assert 0.0 < loader.get_metadata()["threshold"] < 1.0


@requires_artifacts
def test_predict_returns_probabilities(config, full_transaction, api_transaction):
    from inference.model_loader import ModelLoader

    loader = ModelLoader(config)
    loader.load()
    features = loader.feature_pipeline.transform(pd.DataFrame([full_transaction, api_transaction]))
    probs = loader.predict(features)

    assert probs.shape == (2,)
    assert ((probs >= 0) & (probs <= 1)).all()


@requires_artifacts
def test_missing_features_are_rejected(config, api_transaction):
    from inference.model_loader import ModelLoader

    loader = ModelLoader(config)
    loader.load()
    raw = pd.DataFrame([api_transaction])  # untransformed: most model features absent

    with pytest.raises(ValueError, match="FEATURE VALIDATION FAILED"):
        loader.predict(raw)
