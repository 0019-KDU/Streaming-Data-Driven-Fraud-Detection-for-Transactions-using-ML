"""
Simple Feature Engineering Pipeline (No Spark).

Applies the same 88 features used in training using the pre-trained
feature_pipeline.pkl for exact consistency with training.
"""

import pandas as pd
import joblib

from src.inference.logging_utils import setup_logger

logger = setup_logger(__name__)


class FeaturePipelineSimple:
    """
    Simple feature pipeline wrapper (no Spark required).

    Loads the trained sklearn feature pipeline and applies it to pandas DataFrames.
    """

    def __init__(self, config):
        """
        Initialize feature pipeline.

        Args:
            config: Config object with model paths
        """
        self.config = config
        self.feature_pipeline = None
        self.feature_names = []

    def load(self) -> None:
        """Load the trained feature pipeline from disk."""
        pipeline_path = self.config.model.feature_pipeline_path

        logger.info(f"Loading feature pipeline from: {pipeline_path}")
        try:
            self.feature_pipeline = joblib.load(pipeline_path)

            # Get feature names from pipeline
            if hasattr(self.feature_pipeline, 'feature_names'):
                self.feature_names = self.feature_pipeline.feature_names
                logger.info(f"Feature pipeline loaded with {len(self.feature_names)} features")
            else:
                logger.warning("Feature pipeline does not have feature_names attribute")

        except Exception as e:
            logger.error(f"Failed to load feature pipeline: {e}")
            raise

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to a pandas DataFrame.

        Args:
            df: Input pandas DataFrame with raw transaction fields

        Returns:
            DataFrame with engineered features
        """
        if self.feature_pipeline is None:
            raise ValueError("Feature pipeline not loaded. Call load() first.")

        try:
            # Apply the trained feature pipeline
            features_df = self.feature_pipeline.transform(df)

            logger.debug(f"Transformed {len(df)} rows with {features_df.shape[1]} features")

            return features_df

        except Exception as e:
            logger.error(f"Feature transformation failed: {e}")
            raise

    def get_feature_names(self):
        """Get list of engineered feature names."""
        return self.feature_names.copy() if self.feature_names else []


def create_feature_pipeline(config):
    """
    Factory function to create and load feature pipeline.

    Args:
        config: Config object

    Returns:
        Loaded FeaturePipelineSimple instance
    """
    pipeline = FeaturePipelineSimple(config)
    pipeline.load()
    return pipeline
