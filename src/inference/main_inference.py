"""
Main Spark Structured Streaming Inference Job for Fraud Detection.

Reads transactions from Kafka → Applies ML model → Routes to fraud/legit topics.
"""

import json
from datetime import datetime
from typing import Iterator
import pandas as pd

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    from_json, col, udf, pandas_udf, PandasUDFType, struct, to_json
)
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# Absolute imports for spark-submit execution
import sys
import os
sys.path.insert(0, '/app')

from src.inference.config import Config
from src.inference.schema import get_transaction_schema
from src.inference.model_loader import ModelLoader
from src.inference.feature_pipeline_spark import create_feature_pipeline
from src.inference.logging_utils import setup_logger_from_config

# Global instances (initialized once per executor)
_model_loader = None
_feature_pipeline = None
_config = None


def get_or_create_services(config):
    """Initialize services once per executor (singleton pattern)."""
    global _model_loader, _feature_pipeline, _config

    if _config is None:
        _config = config

    if _model_loader is None:
        _model_loader = ModelLoader(config)
        _model_loader.load()

    if _feature_pipeline is None:
        _feature_pipeline = create_feature_pipeline(config)

    return _model_loader, _feature_pipeline


def process_batch(batch_df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Process a batch of transactions through the full inference pipeline.

    Args:
        batch_df: Pandas DataFrame with raw transactions
        config: Config object

    Returns:
        Pandas DataFrame with predictions
    """
    # Initialize services
    model_loader, feature_pipeline = get_or_create_services(config)

    results = []

    for idx, row in batch_df.iterrows():
        transaction = row.to_dict()
        transaction_id = transaction.get('TransactionID', f'unknown_{idx}')
        
        # 📊 Performance tracking
        import time
        tx_start = time.time()

        try:
            # 1. Apply feature engineering
            feat_start = time.time()
            # ✅ FIX: Use .feature_pipeline (the loaded pickle) directly
            features_df = feature_pipeline.feature_pipeline.transform(
                pd.DataFrame([transaction])
            )
            feat_time = time.time() - feat_start

            # 2. Get ML model prediction
            pred_start = time.time()
            fraud_prob = float(model_loader.predict(features_df)[0])
            pred_time = time.time() - pred_start
            
            # 3. Determine decision based on threshold
            threshold = model_loader.get_metadata().get('threshold', 0.5)
            amount = float(transaction.get('TransactionAmt', 0.0))
            
            # Simple threshold-based decision
            if fraud_prob >= threshold:
                decision = 'FRAUD'
                risk_level = 'HIGH'
            else:
                decision = 'LEGITIMATE'
                risk_level = 'LOW'
            
            total_time = time.time() - tx_start

            # 📊 Performance + Result Logging
            logger = setup_logger_from_config(__name__, config)
            logger.info(
                f"TX {transaction_id}: "
                f"prob={fraud_prob:.4f}, "
                f"decision={decision}, "
                f"risk={risk_level}, "
                f"threshold={threshold:.4f}, "
                f"times(feat={feat_time:.2f}s, pred={pred_time:.2f}s, "
                f"total={total_time:.2f}s)"
            )

            # 4. Build output record with original transaction data
            output = {
                'transaction_id': transaction_id,
                'fraud_probability': fraud_prob,
                'decision': decision,
                'risk_level': risk_level,
                'threshold': threshold,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                # Include original transaction fields for dashboard
                'TransactionAmt': amount,
                'ProductCD': str(transaction.get('ProductCD', 'N/A')),
                'card1': str(transaction.get('card1', 'N/A')),
                'card4': str(transaction.get('card4', 'N/A')),
                'card6': str(transaction.get('card6', 'N/A')),
                'P_emaildomain': str(transaction.get('P_emaildomain', 'N/A')),
            }

            results.append(output)

        except Exception as e:
            # Log error but continue processing other transactions
            logger = setup_logger_from_config(__name__, config)
            logger.error(f"Failed to process transaction {transaction_id}: {e}")

            # Return a default safe output with original transaction data
            results.append({
                'transaction_id': transaction_id,
                'fraud_probability': 0.5,
                'decision': 'ERROR',
                'risk_level': 'UNKNOWN',
                'threshold': 0.5,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                # Include original transaction fields even in error case
                'TransactionAmt': float(transaction.get('TransactionAmt', 0.0)),
                'ProductCD': str(transaction.get('ProductCD', 'N/A')),
                'card1': str(transaction.get('card1', 'N/A')),
                'card4': str(transaction.get('card4', 'N/A')),
                'card6': str(transaction.get('card6', 'N/A')),
                'P_emaildomain': str(transaction.get('P_emaildomain', 'N/A')),
            })

    return pd.DataFrame(results)


def main():
    """Main entry point for Spark Structured Streaming job."""

    # Load configuration
    config = Config.load()
    logger = setup_logger_from_config(__name__, config)

    logger.info("="*80)
    logger.info("Starting Fraud Detection Inference Service")
    logger.info("="*80)

    # Create Spark session
    spark = SparkSession.builder \
        .appName(config.spark.app_name) \
        .config("spark.sql.shuffle.partitions", str(config.spark.shuffle_partitions)) \
        .getOrCreate()

    logger.info(f"Spark session created: {spark.version}")

    # Read from Kafka (Confluent Cloud with SASL_SSL)
    logger.info(f"Reading from Kafka topic: {config.kafka.input_topic}")
    logger.info(f"Kafka bootstrap servers: {os.getenv('KAFKA_BOOTSTRAP_SERVERS', config.kafka.brokers)}")

    kafka_df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", os.getenv('KAFKA_BOOTSTRAP_SERVERS', config.kafka.brokers)) \
        .option("kafka.security.protocol", os.getenv('KAFKA_SECURITY_PROTOCOL', 'SASL_SSL')) \
        .option("kafka.sasl.mechanism", "PLAIN") \
        .option("kafka.sasl.jaas.config",
                f'org.apache.kafka.common.security.plain.PlainLoginModule required '
                f'username="{os.getenv("KAFKA_USERNAME")}" '
                f'password="{os.getenv("KAFKA_PASSWORD")}";') \
        .option("subscribe", os.getenv('KAFKA_INPUT_TOPIC', config.kafka.input_topic)) \
        .option("startingOffsets", "latest") \
        .option("maxOffsetsPerTrigger", str(config.kafka.max_offsets_per_trigger)) \
        .load()

    # Parse JSON from Kafka value
    transaction_schema = get_transaction_schema()

    transactions_df = kafka_df.select(
        from_json(col("value").cast("string"), transaction_schema).alias("data")
    ).select("data.*")

    logger.info("Kafka stream configured")

    # Define output schema (must match process_batch return fields)
    output_schema = StructType([
        StructField("transaction_id", StringType(), False),
        StructField("fraud_probability", DoubleType(), False),
        StructField("decision", StringType(), False),
        StructField("risk_level", StringType(), False),
        StructField("threshold", DoubleType(), False),
        StructField("timestamp", StringType(), False),
        # Original transaction fields for dashboard
        StructField("TransactionAmt", DoubleType(), False),
        StructField("ProductCD", StringType(), False),
        StructField("card1", StringType(), False),
        StructField("card4", StringType(), False),
        StructField("card6", StringType(), False),
        StructField("P_emaildomain", StringType(), False),
    ])

    # Apply inference using mapInPandas (Spark 3.0+)
    @pandas_udf(output_schema, PandasUDFType.GROUPED_MAP)
    def inference_udf(batch_df: pd.DataFrame) -> pd.DataFrame:
        """Pandas UDF for batch inference."""
        return process_batch(batch_df, config)

    # Process stream
    predictions_df = transactions_df.mapInPandas(
        lambda iterator: map(lambda batch: process_batch(batch, config), iterator),
        schema=output_schema
    )

    # Route to fraud/legit topics based on decision
    fraud_df = predictions_df.filter(
        col("decision") == "FRAUD"
    )

    legit_df = predictions_df.filter(
        col("decision") == "LEGITIMATE"
    )

    # Write fraud predictions to Kafka
    logger.info(f"Writing fraud predictions to: {os.getenv('KAFKA_FRAUD_TOPIC', config.kafka.fraud_output_topic)}")

    fraud_query = fraud_df \
        .select(to_json(struct("*")).alias("value")) \
        .writeStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", os.getenv('KAFKA_BOOTSTRAP_SERVERS', config.kafka.brokers)) \
        .option("kafka.security.protocol", os.getenv('KAFKA_SECURITY_PROTOCOL', 'SASL_SSL')) \
        .option("kafka.sasl.mechanism", "PLAIN") \
        .option("kafka.sasl.jaas.config",
                f'org.apache.kafka.common.security.plain.PlainLoginModule required '
                f'username="{os.getenv("KAFKA_USERNAME")}" '
                f'password="{os.getenv("KAFKA_PASSWORD")}";') \
        .option("topic", os.getenv('KAFKA_FRAUD_TOPIC', config.kafka.fraud_output_topic)) \
        .option("checkpointLocation", f"{os.getenv('SPARK_CHECKPOINT_LOCATION', config.spark.checkpoint_location)}/fraud") \
        .trigger(processingTime="10 seconds") \
        .start()

    # Write legit predictions to Kafka
    logger.info(f"Writing legit predictions to: {os.getenv('KAFKA_LEGIT_TOPIC', config.kafka.legit_output_topic)}")

    legit_query = legit_df \
        .select(to_json(struct("*")).alias("value")) \
        .writeStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", os.getenv('KAFKA_BOOTSTRAP_SERVERS', config.kafka.brokers)) \
        .option("kafka.security.protocol", os.getenv('KAFKA_SECURITY_PROTOCOL', 'SASL_SSL')) \
        .option("kafka.sasl.mechanism", "PLAIN") \
        .option("kafka.sasl.jaas.config",
                f'org.apache.kafka.common.security.plain.PlainLoginModule required '
                f'username="{os.getenv("KAFKA_USERNAME")}" '
                f'password="{os.getenv("KAFKA_PASSWORD")}";') \
        .option("topic", os.getenv('KAFKA_LEGIT_TOPIC', config.kafka.legit_output_topic)) \
        .option("checkpointLocation", f"{os.getenv('SPARK_CHECKPOINT_LOCATION', config.spark.checkpoint_location)}/legit") \
        .trigger(processingTime="10 seconds") \
        .start()

    logger.info("Streaming queries started successfully")
    logger.info(f"Checkpoint location: {os.getenv('SPARK_CHECKPOINT_LOCATION', config.spark.checkpoint_location)}")
    logger.info(f"Trigger interval: 10 seconds")

    # Wait for termination
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
