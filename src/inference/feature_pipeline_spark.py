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
        Apply feature engineering to a Spark DataFrame batch using distributed processing.
        
        ✅ FIX #4: Uses mapInPandas for parallel execution across partitions.

        Args:
            df: Input Spark DataFrame with raw transaction fields

        Returns:
            DataFrame with engineered features added
        """
        if self.feature_pipeline is None:
            raise ValueError("Feature pipeline not loaded. Call load() first.")

        logger.info("Transforming batch with distributed feature pipeline...")
        
        # Broadcast pipeline to all executors for efficiency
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        pipeline_broadcast = spark.sparkContext.broadcast(self.feature_pipeline)
        
        # Define output schema (TransactionID + 88 features)
        from pyspark.sql.types import StructType, StructField, StringType, DoubleType
        
        output_schema = StructType([
            StructField("TransactionID", StringType(), True)
        ] + [
            StructField(fname, DoubleType(), True) 
            for fname in self.feature_names
        ])
        
        def transform_pandas_batch(iterator):
            """
            Transform each partition using pandas (runs in parallel on executors).
            
            ✅ FIX #4: This function runs on each executor, processing partitions in parallel.
            """
            pipeline = pipeline_broadcast.value
            
            for pandas_df in iterator:
                try:
                    if pandas_df.empty:
                        yield pd.DataFrame(columns=['TransactionID'] + pipeline.feature_names)
                        continue
                    
                    # Apply feature pipeline transform
                    features_df = pipeline.transform(pandas_df)
                    
                    # Add TransactionID back
                    result = pd.concat([
                        pandas_df[['TransactionID']].reset_index(drop=True),
                        features_df.reset_index(drop=True)
                    ], axis=1)
                    
                    yield result
                    
                except Exception as e:
                    logger.error(f"Partition transform failed: {e}")
                    # Return empty DataFrame with correct schema
                    yield pd.DataFrame(columns=['TransactionID'] + pipeline.feature_names)
        
        try:
            # Apply distributed transform using mapInPandas
            result_df = df.mapInPandas(transform_pandas_batch, schema=output_schema)
            
            logger.info("✅ Batch transformed using distributed processing")
            return result_df
            
        except Exception as e:
            logger.error(f"Distributed transformation failed: {e}")
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
