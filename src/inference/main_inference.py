"""
Spark Structured Streaming inference job (post-authorization monitoring path).

Kafka `transactions` → shared FraudScorer (feature pipeline + XGBoost + velocity/ATO
+ hybrid decision engine) → Kafka `legit_predictions` (APPROVE) or
`fraud_predictions` (REVIEW / HOLD / BLOCK).

The synchronous authorization path (api_direct.py) uses the exact same FraudScorer,
so both entry points return identical decisions for identical input.
"""

import os
import sys
from typing import Iterator

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, struct, to_json
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

sys.path.insert(0, '/app')

from src.inference.config import Config
from src.inference.logging_utils import setup_logger_from_config
from src.inference.schema import get_transaction_schema
from src.inference.scoring import FraudScorer, connect_redis_or_none

# One scorer per Python worker process (artifacts are loaded once, not per batch)
_scorer = None

OUTPUT_SCHEMA = StructType([
    StructField("transaction_id", StringType(), False),
    StructField("fraud_probability", DoubleType(), False),
    StructField("adjusted_probability", DoubleType(), False),
    StructField("decision", StringType(), False),
    StructField("risk_level", StringType(), False),
    StructField("risk_factors", StringType(), False),
    StructField("explanation", StringType(), False),
    StructField("velocity_risk", DoubleType(), False),
    StructField("amount_risk", DoubleType(), False),
    StructField("ato_risk", DoubleType(), False),
    StructField("base_threshold", DoubleType(), False),
    StructField("hybrid_threshold", DoubleType(), False),
    StructField("model_kind", StringType(), False),
    StructField("model_version", StringType(), False),
    StructField("rules_version", StringType(), False),
    StructField("shadow_model_version", StringType(), False),
    StructField("shadow_probability", DoubleType(), False),
    StructField("shadow_decision", StringType(), False),
    StructField("timestamp", StringType(), False),
    StructField("TransactionAmt", DoubleType(), False),
    StructField("ProductCD", StringType(), False),
    StructField("card1", StringType(), False),
    StructField("card4", StringType(), False),
    StructField("card6", StringType(), False),
    StructField("P_emaildomain", StringType(), False),
    StructField("terminal_id", StringType(), False),
])
OUTPUT_COLUMNS = [f.name for f in OUTPUT_SCHEMA.fields]


def get_scorer(config) -> FraudScorer:
    global _scorer
    if _scorer is None:
        _scorer = FraudScorer(config, connect_redis_or_none(config), serving_path="spark_stream")
    return _scorer


def score_partition(batches: Iterator[pd.DataFrame], config) -> Iterator[pd.DataFrame]:
    """mapInPandas function: score each Arrow batch in one vectorised call."""
    logger = setup_logger_from_config(__name__, config)
    scorer = get_scorer(config)

    for batch_df in batches:
        if batch_df.empty:
            yield pd.DataFrame(columns=OUTPUT_COLUMNS)
            continue

        records = batch_df.where(pd.notna(batch_df), None).to_dict(orient="records")
        try:
            results = scorer.score_batch(records)
        except Exception as exc:  # noqa: BLE001 - never kill the stream on one bad batch
            logger.error(f"Scoring failed for batch of {len(records)}: {exc}", exc_info=True)
            continue

        for r in results:
            logger.info(
                f"TX {r['transaction_id']}: prob={r['fraud_probability']:.4f} "
                f"decision={r['decision']} factors=[{r['risk_factors']}]"
            )
        yield pd.DataFrame(results, columns=OUTPUT_COLUMNS)


def kafka_options(config) -> dict:
    """Common Kafka connection options (credentials come from the environment)."""
    opts = {
        "kafka.bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", config.kafka.brokers),
    }
    username, password = os.getenv("KAFKA_USERNAME"), os.getenv("KAFKA_PASSWORD")
    if username and password:
        opts.update({
            "kafka.security.protocol": os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL"),
            "kafka.sasl.mechanism": "PLAIN",
            "kafka.sasl.jaas.config": (
                "org.apache.kafka.common.security.plain.PlainLoginModule required "
                f'username="{username}" password="{password}";'
            ),
        })
    return opts


def main():
    config = Config.load()
    logger = setup_logger_from_config(__name__, config)
    logger.info("Starting Fraud Detection Inference Service (Spark Structured Streaming)")

    spark = (
        SparkSession.builder
        .appName(config.spark.app_name)
        .config("spark.sql.shuffle.partitions", str(config.spark.shuffle_partitions))
        .getOrCreate()
    )

    kafka = kafka_options(config)
    input_topic = os.getenv("KAFKA_INPUT_TOPIC", config.kafka.input_topic)
    fraud_topic = os.getenv("KAFKA_FRAUD_TOPIC", config.kafka.fraud_output_topic)
    legit_topic = os.getenv("KAFKA_LEGIT_TOPIC", config.kafka.legit_output_topic)
    checkpoint_root = os.getenv("SPARK_CHECKPOINT_LOCATION", config.spark.checkpoint_location)
    trigger = config.spark.trigger_interval

    transactions = (
        spark.readStream.format("kafka")
        .options(**kafka)
        .option("subscribe", input_topic)
        .option("startingOffsets", "latest")
        .option("maxOffsetsPerTrigger", str(config.kafka.max_offsets_per_trigger))
        .load()
        .select(from_json(col("value").cast("string"), get_transaction_schema()).alias("data"))
        .select("data.*")
    )

    predictions = transactions.mapInPandas(
        lambda batches: score_partition(batches, config), schema=OUTPUT_SCHEMA
    )

    def write(df, topic, name):
        return (
            df.select(to_json(struct("*")).alias("value"))
            .writeStream.format("kafka")
            .options(**kafka)
            .option("topic", topic)
            .option("checkpointLocation", f"{checkpoint_root}/{name}")
            .trigger(processingTime=trigger)
            .start()
        )

    write(predictions.filter(col("decision") != "APPROVE"), fraud_topic, "fraud")
    write(predictions.filter(col("decision") == "APPROVE"), legit_topic, "legit")

    logger.info(f"Reading {input_topic} → writing {fraud_topic} / {legit_topic} every {trigger}")
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
