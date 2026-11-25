"""
Model loader for trained fraud detection models (LightGBM/XGBoost).

Loads pre-trained artifacts from disk and provides prediction interface.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from pathlib import Path

from .logging_utils import setup_logger

logger = setup_logger(__name__)


class ModelLoader:
    """
    Loads and manages the trained fraud detection model and feature pipeline.
    Supports both LightGBM and XGBoost models.
    """

    def __init__(self, config, bundle_path: str = None, pipeline_path: str = None):
        """
        Args:
            config: Config object with model paths
            bundle_path / pipeline_path: override the configured paths
                (used to load a shadow/challenger model next to the champion)
        """
        self.config = config
        self.bundle_path = bundle_path or config.model.model_bundle_path
        self.pipeline_path = pipeline_path or config.model.feature_pipeline_path
        self.model = None
        self.feature_pipeline = None
        self.feature_names: List[str] = []
        self.model_metadata: Dict[str, Any] = {}
        self.model_type: str = 'unknown'  # 'lightgbm', 'xgboost', or 'sklearn'
        self.model_kind: str = 'ieee_cis'  # 'ieee_cis' (benchmark) or 'decision_time' (live)
        self.best_iteration: int = None

    def load(self) -> None:
        """
        Load the model bundle (and, for IEEE-CIS models, the feature pipeline).

        Raises:
            FileNotFoundError: If a required file does not exist
        """
        model_path = Path(self.bundle_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model bundle not found: {model_path}")

        logger.info(f"Loading model bundle from: {model_path}")
        bundle = joblib.load(model_path)
        if not isinstance(bundle, dict):
            bundle = {'model': bundle}

        self.model = bundle.get('model') or bundle.get('calibrated_model')
        if self.model is None:
            raise ValueError(f"No model found in bundle {model_path}")
        self.feature_names = list(bundle.get('feature_names', []))
        self.model_kind = bundle.get('model_kind', 'ieee_cis')

        name = type(self.model).__name__
        if 'LightGBM' in name or 'Booster' in name:
            self.model_type = 'lightgbm'
            self.best_iteration = bundle.get('best_iteration')
        elif 'XGB' in name:
            self.model_type = 'xgboost'
        else:
            self.model_type = 'sklearn'

        threshold = bundle.get('threshold')
        if threshold is None:
            threshold = self.config.model.base_threshold
            logger.warning(f"No threshold in bundle, using config base_threshold {threshold:.4f}")

        self.model_metadata = {
            'model_type': self.model_type,
            'model_kind': self.model_kind,
            'model_version': bundle.get('model_version', model_path.stem),
            'threshold': float(threshold),
            'decision_thresholds': bundle.get('decision_thresholds'),
            'reference_scores': bundle.get('reference_scores'),
            'metrics': bundle.get('metrics', {}),
            'n_features': len(self.feature_names),
            'training_date': bundle.get('training_date', 'unknown'),
            'best_iteration': self.best_iteration,
        }
        logger.info(
            f"Model loaded: kind={self.model_kind} type={self.model_type} "
            f"version={self.model_metadata['model_version']} features={len(self.feature_names)}"
        )

        if self.model_kind == 'decision_time':
            return  # features come from the online feature store, not a pipeline pickle

        pipeline_path = Path(self.pipeline_path)
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Feature pipeline not found: {pipeline_path}")
        logger.info(f"Loading feature pipeline from: {pipeline_path}")
        self.feature_pipeline = joblib.load(pipeline_path)

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probabilities for a batch of transactions.

        Args:
            features_df: DataFrame with engineered features (88 columns)

        Returns:
            Array of fraud probabilities [0, 1]

        Raises:
            ValueError: If model not loaded or feature mismatch
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")

        # ✅ FIX #2: Strict feature validation
        if self.feature_names:
            self._validate_features(features_df)
            # Reorder columns to match training
            features_df = features_df[self.feature_names]

        # Predict probabilities based on model type
        try:
            if self.model_type == 'lightgbm':
                # LightGBM model - use predict() with num_iteration
                if self.best_iteration:
                    probas = self.model.predict(features_df, num_iteration=self.best_iteration)
                    logger.debug(f"LightGBM prediction with best_iteration={self.best_iteration}")
                else:
                    probas = self.model.predict(features_df)
                    logger.debug("LightGBM prediction without num_iteration")
                # LightGBM predict returns probabilities directly
                return probas if len(probas.shape) == 1 else probas[:, 1]
            else:
                # XGBoost or sklearn model - use predict_proba
                probas = self.model.predict_proba(features_df)[:, 1]
                return probas
        except Exception as e:
            logger.error(f"Prediction failed for {self.model_type} model: {e}")
            raise

    def predict_single(self, features: Dict[str, float]) -> float:
        """
        Predict fraud probability for a single transaction.

        Args:
            features: Dictionary of feature_name -> value

        Returns:
            Fraud probability [0, 1]
        """
        df = pd.DataFrame([features])
        proba = self.predict(df)
        return float(proba[0])

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.

        Returns:
            Dictionary with model info
        """
        return self.model_metadata.copy()

    def get_feature_names(self) -> List[str]:
        """
        Get expected feature names in correct order.

        Returns:
            List of feature names
        """
        return self.feature_names.copy()

    def _validate_features(self, features_df: pd.DataFrame) -> None:
        """
        ✅ FIX #2: Validate feature parity with training.
        
        Raises:
            ValueError: If features don't match training
        """
        if not self.feature_names:
            logger.warning("⚠️ No feature names in model bundle, skipping validation")
            return
        
        expected = set(self.feature_names)
        actual = set(features_df.columns)
        
        missing = expected - actual
        extra = actual - expected
        
        if missing:
            raise ValueError(
                f"❌ FEATURE VALIDATION FAILED: Missing {len(missing)} features from training: "
                f"{list(missing)[:10]}{'...' if len(missing) > 10 else ''}"
            )
        
        if extra:
            logger.warning(
                f"⚠️ Extra features not in training: {list(extra)[:10]}{'...' if len(extra) > 10 else ''}"
            )
        
        # Verify feature count
        expected_count = len(self.feature_names)
        actual_count = len([c for c in features_df.columns if c in expected])
        
        if actual_count != expected_count:
            raise ValueError(
                f"❌ Feature count mismatch: Expected {expected_count}, got {actual_count}"
            )
        
        logger.debug(f"Feature validation passed: {expected_count} features")


def load_model_artifacts(config) -> Tuple[Any, Any]:
    """
    Convenience function to load model and feature pipeline.

    Args:
        config: Config object

    Returns:
        Tuple of (model, feature_pipeline)
    """
    loader = ModelLoader(config)
    loader.load()
    return loader.model, loader.feature_pipeline
