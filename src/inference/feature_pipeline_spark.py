"""
Feature Engineering Pipeline for Spark Structured Streaming.

Applies the same 88 features used in training:
- Time features
- Amount features
- Email features
- Card features
- Frequency encoding
- Mean encoding (fraud rates)
- Magic UID features
- D-normalized features
- Group aggregations

NOTE: This uses the pre-trained feature_pipeline.pkl for transform() to ensure
exact consistency with training.
"""

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
import joblib

from .logging_utils import setup_logger

logger = setup_logger(__name__)


class FeaturePipelineSpark:
    """
    Spark-compatible feature pipeline wrapper.

    Loads the trained sklearn feature pipeline and applies it to Spark DataFrames
    using pandas UDFs for distributed processing.
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

    def transform_batch(self, df: DataFrame) -> DataFrame:
        """
        Apply feature engineering to a Spark DataFrame batch.

        Args:
            df: Input Spark DataFrame with raw transaction fields

        Returns:
            DataFrame with engineered features added
        """
        if self.feature_pipeline is None:
            raise ValueError("Feature pipeline not loaded. Call load() first.")

        # Convert to Pandas, apply transform, convert back
        # This is OK for micro-batches (typically < 10K rows)
        @pandas_udf(StringType())
        def apply_feature_transform(batch_df: pd.DataFrame) -> pd.Series:
            """
            Apply feature pipeline transformation using pandas UDF.

            Returns engineered features as JSON string.
            """
            try:
                # Apply the trained feature pipeline
                transformed = self.feature_pipeline.transform(batch_df)

                # Convert to JSON strings for transport
                return pd.Series([transformed.to_json(orient='records')])

            except Exception as e:
                logger.error(f"Feature transformation failed: {e}")
                # Return empty features on error
                return pd.Series(['{}'])

        # For now, we'll use a simpler approach:
        # Convert entire micro-batch to pandas, transform, merge back

        logger.info("Transforming batch with feature pipeline...")

        # Collect to pandas (OK for streaming micro-batches)
        pandas_df = df.toPandas()

        # Apply feature pipeline transform
        try:
            features_df = self.feature_pipeline.transform(pandas_df)

            # Create result DataFrame with original columns + features
            result_pandas = pd.concat([
                pandas_df[['TransactionID']],  # Keep ID
                features_df  # Add engineered features
            ], axis=1)

            # Convert back to Spark DataFrame
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()

            result_spark = spark.createDataFrame(result_pandas)

            logger.info(f"Transformed batch: {result_spark.count()} rows, {len(result_spark.columns)} columns")

            return result_spark

        except Exception as e:
            logger.error(f"Batch transformation failed: {e}")
            raise

    def get_feature_names(self):
        """Get list of engineered feature names."""
        return self.feature_names.copy()


def create_feature_pipeline(config):
    """
    Factory function to create and load feature pipeline.

    Args:
        config: Config object

    Returns:
        Loaded FeaturePipelineSpark instance
    """
    pipeline = FeaturePipelineSpark(config)
    pipeline.load()
    return pipeline
