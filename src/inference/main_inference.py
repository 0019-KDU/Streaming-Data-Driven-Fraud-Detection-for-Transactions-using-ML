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
from pyspark.sql.types import StructType, StringType, DoubleType

from .config import Config
from .schema import get_transaction_schema
from .model_loader import ModelLoader
from .feature_pipeline_spark import create_feature_pipeline
from .velocity_service import VelocityService
from .ato_service import ATOService
from .decision_engine import DecisionEngine
from .logging_utils import setup_logger_from_config

# Global instances (initialized once per executor)
_model_loader = None
_feature_pipeline = None
_velocity_service = None
_ato_service = None
_decision_engine = None
_config = None


def get_or_create_services(config):
    """Initialize services once per executor (singleton pattern)."""
    global _model_loader, _feature_pipeline, _velocity_service, _ato_service, _decision_engine, _config

    if _config is None:
        _config = config

    if _model_loader is None:
        _model_loader = ModelLoader(config)
        _model_loader.load()

    if _feature_pipeline is None:
        _feature_pipeline = create_feature_pipeline(config)

    if _velocity_service is None:
        _velocity_service = VelocityService(config)

    if _ato_service is None:
        _ato_service = ATOService(config)

    if _decision_engine is None:
        _decision_engine = DecisionEngine(config)

    return _model_loader, _feature_pipeline, _velocity_service, _ato_service, _decision_engine


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
    model_loader, feature_pipeline, velocity_service, ato_service, decision_engine = \
        get_or_create_services(config)

    results = []

    for idx, row in batch_df.iterrows():
        transaction = row.to_dict()
        transaction_id = transaction.get('TransactionID', f'unknown_{idx}')

        try:
            # 1. Apply feature engineering
            features_df = feature_pipeline.feature_pipeline.transform(
                pd.DataFrame([transaction])
            )

            # 2. Get ML model prediction
            fraud_prob = float(model_loader.predict(features_df)[0])

            # 3. Analyze velocity
            card1 = str(transaction.get('card1', 'unknown'))
            uid = f"{card1}_{transaction.get('addr1', 'na')}_{transaction.get('P_emaildomain', 'na')}"
            amount = float(transaction.get('TransactionAmt', 0.0))
            timestamp = float(transaction.get('TransactionDT', 0.0))

            velocity_result = velocity_service.analyze_velocity(
                card1, uid, amount, timestamp
            )

            # Record transaction for future velocity calculations
            velocity_service.record_transaction(card1, uid, amount, timestamp)

            # 4. Analyze ATO
            ato_result = ato_service.process_transaction(card1, transaction)

            # 5. Make final decision
            decision_result = decision_engine.make_decision(
                fraud_probability=fraud_prob,
                velocity_risk=velocity_result.velocity_risk,
                amount_risk=velocity_result.amount_risk,
                ato_risk=ato_result.ato_risk,
                ato_detected=ato_result.ato_detected,
                transaction=transaction,
                velocity_factors=velocity_result.factors,
                ato_factors=ato_result.factors
            )

            # 6. Build output record
            output = {
                'transaction_id': transaction_id,
                'fraud_probability': fraud_prob,
                'decision': decision_result.decision.value,
                'risk_level': decision_result.risk_level.value,
                'risk_factors': json.dumps(decision_result.risk_factors),
                'ato_risk': ato_result.ato_risk,
                'velocity_risk': velocity_result.velocity_risk,
                'amount_risk': velocity_result.amount_risk,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }

            results.append(output)

        except Exception as e:
            # Log error but continue processing other transactions
            logger = setup_logger_from_config(__name__, config)
            logger.error(f"Failed to process transaction {transaction_id}: {e}")

            # Return a default safe output
            results.append({
                'transaction_id': transaction_id,
                'fraud_probability': 0.5,
                'decision': 'REVIEW',
                'risk_level': 'MEDIUM',
                'risk_factors': json.dumps(['processing_error']),
                'ato_risk': 0.0,
                'velocity_risk': 0.0,
                'amount_risk': 0.0,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
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
        .config("spark.sql.shuffle.partitions", config.spark.shuffle_partitions) \
        .getOrCreate()

    logger.info(f"Spark session created: {spark.version}")

    # Read from Kafka
    logger.info(f"Reading from Kafka topic: {config.kafka.input_topic}")

    kafka_df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", config.kafka.brokers) \
        .option("subscribe", config.kafka.input_topic) \
        .option("startingOffsets", "latest") \
        .option("maxOffsetsPerTrigger", config.kafka.max_offsets_per_trigger) \
        .load()

    # Parse JSON from Kafka value
    transaction_schema = get_transaction_schema()

    transactions_df = kafka_df.select(
        from_json(col("value").cast("string"), transaction_schema).alias("data")
    ).select("data.*")

    logger.info("Kafka stream configured")

    # Define output schema
    output_schema = StructType([
        StructField("transaction_id", StringType(), False),
        StructField("fraud_probability", DoubleType(), False),
        StructField("decision", StringType(), False),
        StructField("risk_level", StringType(), False),
        StructField("risk_factors", StringType(), False),
        StructField("ato_risk", DoubleType(), False),
        StructField("velocity_risk", DoubleType(), False),
        StructField("amount_risk", DoubleType(), False),
        StructField("timestamp", StringType(), False),
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
        col("decision").isin(["BLOCK", "HOLD", "REVIEW"])
    )

    legit_df = predictions_df.filter(
        col("decision") == "APPROVE"
    )

    # Write fraud predictions to Kafka
    logger.info(f"Writing fraud predictions to: {config.kafka.fraud_output_topic}")

    fraud_query = fraud_df \
        .select(to_json(struct("*")).alias("value")) \
        .writeStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", config.kafka.brokers) \
        .option("topic", config.kafka.fraud_output_topic) \
        .option("checkpointLocation", f"{config.spark.checkpoint_location}/fraud") \
        .trigger(processingTime=config.spark.trigger_interval) \
        .start()

    # Write legit predictions to Kafka
    logger.info(f"Writing legit predictions to: {config.kafka.legit_output_topic}")

    legit_query = legit_df \
        .select(to_json(struct("*")).alias("value")) \
        .writeStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", config.kafka.brokers) \
        .option("topic", config.kafka.legit_output_topic) \
        .option("checkpointLocation", f"{config.spark.checkpoint_location}/legit") \
        .trigger(processingTime=config.spark.trigger_interval) \
        .start()

    logger.info("Streaming queries started successfully")
    logger.info(f"Checkpoint location: {config.spark.checkpoint_location}")
    logger.info(f"Trigger interval: {config.spark.trigger_interval}")

    # Wait for termination
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
