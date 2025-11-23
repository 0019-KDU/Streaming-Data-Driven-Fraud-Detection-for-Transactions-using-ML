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

    def __init__(self, config):
        """
        Initialize model loader.

        Args:
            config: Config object with model paths
        """
        self.config = config
        self.model = None
        self.feature_pipeline = None
        self.feature_names: List[str] = []
        self.model_metadata: Dict[str, Any] = {}
        self.model_type: str = 'unknown'  # 'lightgbm', 'xgboost', or 'sklearn'
        self.best_iteration: int = None  # For LightGBM num_iteration parameter

    def load(self) -> None:
        """
        Load model bundle and feature pipeline from disk.

        Raises:
            FileNotFoundError: If model files don't exist
            Exception: If loading fails
        """
        model_path = Path(self.config.model.model_bundle_path)
        pipeline_path = Path(self.config.model.feature_pipeline_path)

        # Check files exist
        if not model_path.exists():
            raise FileNotFoundError(f"Model bundle not found: {model_path}")
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Feature pipeline not found: {pipeline_path}")

        logger.info(f"Loading model bundle from: {model_path}")
        try:
            model_bundle = joblib.load(model_path)

            # Debug logging
            logger.info(f"DEBUG: model_bundle loaded successfully")
            logger.info(f"DEBUG: model_bundle type: {type(model_bundle)}")
            logger.info(f"DEBUG: model_bundle is None: {model_bundle is None}")
            logger.info(f"DEBUG: model_bundle is dict: {isinstance(model_bundle, dict)}")

            if hasattr(model_bundle, '__dict__'):
                logger.info(f"DEBUG: model_bundle attributes: {list(model_bundle.__dict__.keys())}")

            # Extract components from bundle
            if isinstance(model_bundle, dict):
                logger.info(f"DEBUG: model_bundle is a dict with keys: {list(model_bundle.keys())}")
                self.model = model_bundle.get('model') or model_bundle.get('calibrated_model')
                logger.info(f"DEBUG: Extracted model from dict, type: {type(self.model)}, is None: {self.model is None}")
                self.feature_names = model_bundle.get('feature_names', [])
                
                # Detect model type
                model_class_name = type(self.model).__name__
                if 'LightGBM' in model_class_name or 'Booster' in model_class_name:
                    self.model_type = 'lightgbm'
                    self.best_iteration = model_bundle.get('best_iteration', None)
                    logger.info(f"✅ Detected LightGBM model (best_iteration={self.best_iteration})")
                elif 'XGB' in model_class_name:
                    self.model_type = 'xgboost'
                    logger.info(f"✅ Detected XGBoost model")
                else:
                    self.model_type = 'sklearn'
                    logger.info(f"✅ Detected scikit-learn compatible model")
                
                # ✅ FIX #1: Load threshold from bundle, validate against config
                bundle_threshold = model_bundle.get('threshold')
                config_threshold = self.config.model.base_threshold
                
                if bundle_threshold is not None:
                    if abs(bundle_threshold - config_threshold) > 0.01:
                        logger.warning(
                            f"⚠️ THRESHOLD MISMATCH: Bundle={bundle_threshold:.4f}, "
                            f"Config={config_threshold:.4f}. Using bundle threshold."
                        )
                    threshold = bundle_threshold
                    logger.info(f"✅ Loaded threshold from model bundle: {threshold:.4f}")
                else:
                    threshold = config_threshold
                    logger.warning(f"⚠️ No threshold in bundle, using config: {threshold:.4f}")
                
                self.model_metadata = {
                    'model_type': self.model_type,
                    'threshold': threshold,
                    'metrics': model_bundle.get('metrics', {}),
                    'n_features': len(self.feature_names),
                    'training_date': model_bundle.get('training_date', 'unknown'),
                    'best_iteration': self.best_iteration
                }
            else:
                # Fallback: assume it's just the model (raw pickle)
                logger.info(f"DEBUG: model_bundle is NOT a dict, treating as raw model")
                self.model = model_bundle
                logger.info(f"DEBUG: Assigned model_bundle to self.model, type: {type(self.model)}, is None: {self.model is None}")
                self.feature_names = []
                self.model_metadata = {
                    'model_type': type(model_bundle).__name__,
                    'threshold': self.config.model.base_threshold,
                    'metrics': {},
                    'n_features': 88  # Default for IEEE-CIS
                }
                logger.warning(f"Model bundle is not a dict, loaded raw model: {type(model_bundle)}")

            # Final verification
            logger.info(f"DEBUG: After assignment, self.model is None: {self.model is None}")
            logger.info(f"DEBUG: After assignment, self.model type: {type(self.model)}")

            logger.info(
                f"Model loaded successfully: {self.model_metadata.get('model_type', 'unknown')}, "
                f"features={self.model_metadata.get('n_features', len(self.feature_names))}"
            )

        except Exception as e:
            logger.error(f"Failed to load model bundle: {e}")
            raise

        logger.info(f"Loading feature pipeline from: {pipeline_path}")
        try:
            self.feature_pipeline = joblib.load(pipeline_path)
            logger.info("Feature pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load feature pipeline: {e}")
            raise

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
        
        logger.info(f"✅ Feature validation passed: {expected_count} features")


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
