"""
Model loader for trained XGBoost model and feature pipeline.

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

            # Extract components from bundle
            if isinstance(model_bundle, dict):
                self.model = model_bundle.get('model')
                self.feature_names = model_bundle.get('feature_names', [])
                self.model_metadata = {
                    'model_type': model_bundle.get('model_type', 'unknown'),
                    'threshold': model_bundle.get('threshold', self.config.model.base_threshold),
                    'metrics': model_bundle.get('metrics', {}),
                    'n_features': len(self.feature_names)
                }
            else:
                # Fallback: assume it's just the model (raw pickle)
                self.model = model_bundle
                self.feature_names = []
                self.model_metadata = {
                    'model_type': type(model_bundle).__name__,
                    'threshold': self.config.model.base_threshold,
                    'metrics': {},
                    'n_features': 88  # Default for IEEE-CIS
                }
                logger.warning(f"Model bundle is not a dict, loaded raw model: {type(model_bundle)}")

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

        # Ensure feature order matches training
        if self.feature_names:
            missing_features = set(self.feature_names) - set(features_df.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # Add missing features with zeros
                for feat in missing_features:
                    features_df[feat] = 0.0

            # Reorder columns to match training
            features_df = features_df[self.feature_names]

        # Predict probabilities
        try:
            probas = self.model.predict_proba(features_df)[:, 1]
            return probas
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
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
