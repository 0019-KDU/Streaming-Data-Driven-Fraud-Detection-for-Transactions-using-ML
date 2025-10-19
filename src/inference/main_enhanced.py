"""
Enhanced Real-time Fraud Detection Inference Pipeline

Key enhancements:
- Loads both model and feature pipeline for train/serve consistency
- Computes fraud probability with configurable threshold
- Assigns risk levels (HIGH/MEDIUM/LOW)
- Publishes to dual topics: fraud_predictions and legit_predictions
- Includes decision field (BLOCK/APPROVE)
- Maps IEEE-CIS features to internal schema
"""

import logging
import os
import json
from datetime import datetime

import joblib
import yaml
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, hour, dayofweek, dayofmonth,
    when, lit, coalesce, udf, to_json, struct
)
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import (
    StructType, StructField, StringType,
    IntegerType, DoubleType, TimestampType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class EnhancedFraudDetectionInference:
    """
    Enhanced fraud detection inference with:
    - Feature pipeline support
    - Risk level classification
    - Dual-topic output (fraud/legit)
    - Decision flags (BLOCK/APPROVE)
    """

    def __init__(self, config_path="/app/config.yaml"):
        """Initialize enhanced inference pipeline"""
        load_dotenv(dotenv_path="/app/.env")

        self.config = self._load_config(config_path)
        self.spark = self._init_spark_session()

        # Load model and feature pipeline
        self.model = self._load_model(self.config["model"]["path"])
        self.feature_pipeline = self._load_feature_pipeline(
            self.config["model"]["feature_pipeline_path"]
        )

        # Broadcast to workers
        self.broadcast_model = self.spark.sparkContext.broadcast(self.model)
        self.broadcast_pipeline = self.spark.sparkContext.broadcast(self.feature_pipeline)

        # Load inference configuration
        self.threshold = self.config["inference"]["threshold"]
        self.risk_bands = self.config["inference"]["risk_bands"]

        logger.info(f"Inference threshold: {self.threshold}")
        logger.info(f"Risk bands: {self.risk_bands}")

    @staticmethod
    def _load_config(config_path):
        """Load YAML configuration"""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def _load_model(self, model_path):
        """Load trained model"""
        try:
            if not os.path.exists(model_path):
                logger.warning(f"Model not found at {model_path}, using fallback")
                return None

            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _load_feature_pipeline(self, pipeline_path):
        """Load feature preprocessing pipeline"""
        try:
            if not os.path.exists(pipeline_path):
                logger.warning(f"Feature pipeline not found at {pipeline_path}")
                return None

            pipeline = joblib.load(pipeline_path)
            logger.info(f"Feature pipeline loaded from {pipeline_path}")
            return pipeline
        except Exception as e:
            logger.warning(f"Could not load feature pipeline: {str(e)}")
            return None

    def _init_spark_session(self):
        """Initialize Spark session"""
        try:
            packages = self.config.get("spark", {}).get("packages", "")
            builder = SparkSession.builder.appName("FraudDetectionInferenceEnhanced")

            if packages:
                builder = builder.config("spark.jars.packages", packages)

            spark = builder.getOrCreate()
            logger.info("Spark Session initialized")
            return spark
        except Exception as e:
            logger.error(f"Error initializing Spark: {str(e)}")
            raise

    def read_from_kafka(self):
        """Read streaming data from Kafka"""
        logger.info("Reading data from Kafka topic...")

        kafka_config = self.config["kafka"]
        kafka_bootstrap_servers = kafka_config.get("bootstrap_servers")
        kafka_topic = kafka_config["topic"]
        kafka_security_protocol = kafka_config.get("security_protocol", "SASL_SSL")
        kafka_sasl_mechanism = kafka_config.get("sasl_mechanism", "PLAIN")
        kafka_username = kafka_config.get("username")
        kafka_password = kafka_config.get("password")

        kafka_sasl_jaas_config = (
            f'org.apache.kafka.common.security.plain.PlainLoginModule required '
            f'username="{kafka_username}" password="{kafka_password}";'
        )

        # Store for reuse
        self.kafka_config_dict = {
            "bootstrap_servers": kafka_bootstrap_servers,
            "security_protocol": kafka_security_protocol,
            "sasl_mechanism": kafka_sasl_mechanism,
            "sasl_jaas_config": kafka_sasl_jaas_config
        }

        # Define schema for incoming transactions (supports both synthetic and IEEE-CIS formats)
        json_schema = StructType([
            # Core fields (always present)
            StructField("transaction_id", StringType(), True),
            StructField("timestamp", TimestampType(), True),

            # Synthetic producer fields
            StructField("user_id", IntegerType(), True),
            StructField("amount", DoubleType(), True),
            StructField("currency", StringType(), True),
            StructField("merchant", StringType(), True),
            StructField("location", StringType(), True),

            # IEEE-CIS fields (optional, for REST API submissions)
            StructField("TransactionID", StringType(), True),
            StructField("TransactionDT", DoubleType(), True),
            StructField("TransactionAmt", DoubleType(), True),
            StructField("ProductCD", StringType(), True),
            StructField("card1", IntegerType(), True),
            StructField("card2", IntegerType(), True),
            StructField("addr1", IntegerType(), True),
            StructField("P_emaildomain", StringType(), True),
            StructField("R_emaildomain", StringType(), True),
        ])

        # Create streaming DataFrame
        df = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
            .option("subscribe", kafka_topic) \
            .option("startingOffsets", "latest") \
            .option("kafka.security.protocol", kafka_security_protocol) \
            .option("kafka.sasl.mechanism", kafka_sasl_mechanism) \
            .option("kafka.sasl.jaas.config", kafka_sasl_jaas_config) \
            .load()

        # Parse JSON
        parsed_df = df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), json_schema).alias("data")) \
            .select("data.*")

        return parsed_df

    def normalize_features(self, df):
        """
        Normalize incoming data to unified internal schema

        Maps both synthetic producer format and IEEE-CIS format to internal schema
        """
        logger.info("Normalizing features...")

        # Map TransactionID (IEEE-CIS) to transaction_id (internal)
        df = df.withColumn(
            "transaction_id",
            coalesce(col("transaction_id"), col("TransactionID"))
        )

        # Map TransactionAmt (IEEE-CIS) to amount (internal)
        df = df.withColumn(
            "amount",
            coalesce(col("amount"), col("TransactionAmt"))
        )

        # Map TransactionDT to timestamp if needed
        # Assume TransactionDT is seconds offset; convert to approximate timestamp
        df = df.withColumn(
            "timestamp",
            coalesce(
                col("timestamp"),
                # If TransactionDT exists, convert to timestamp (approximate)
                when(col("TransactionDT").isNotNull(),
                     lit(datetime(2017, 12, 1).timestamp()) + col("TransactionDT")).cast(TimestampType())
            )
        )

        # Ensure all required fields exist with defaults
        df = df.withColumn("user_id", coalesce(col("user_id"), lit(-1)))
        df = df.withColumn("amount", coalesce(col("amount"), lit(0.0)))
        df = df.withColumn("currency", coalesce(col("currency"), lit("USD")))
        df = df.withColumn("merchant", coalesce(col("merchant"), lit("unknown")))
        df = df.withColumn("location", coalesce(col("location"), lit("US")))

        # IEEE-CIS specific fields (fill with defaults if missing)
        df = df.withColumn("card1", coalesce(col("card1"), lit(-1)))
        df = df.withColumn("card2", coalesce(col("card2"), lit(-1)))
        df = df.withColumn("addr1", coalesce(col("addr1"), lit(-1)))
        df = df.withColumn("P_emaildomain", coalesce(col("P_emaildomain"), lit(None)))
        df = df.withColumn("R_emaildomain", coalesce(col("R_emaildomain"), lit(None)))
        df = df.withColumn("ProductCD", coalesce(col("ProductCD"), lit("W")))

        return df

    def add_features(self, df):
        """Add engineered features for model inference"""
        # Temporal features
        df = df.withColumn("transaction_hour", hour(col("timestamp")))
        df = df.withColumn("transaction_day_of_week", dayofweek(col("timestamp")))
        df = df.withColumn("is_weekend",
                           when((col("transaction_day_of_week") == 1) | (col("transaction_day_of_week") == 7), 1).otherwise(0))
        df = df.withColumn("is_night",
                           when((col("transaction_hour") >= 22) | (col("transaction_hour") <= 6), 1).otherwise(0))

        # Amount features
        df = df.withColumn("log_amt", when(col("amount") > 0, lit(1) + col("amount")).otherwise(lit(1)))
        df = df.withColumn("log_amt", lit(1))  # Placeholder - will be computed in UDF
        df = df.withColumn("sqrt_amt", lit(1))  # Placeholder

        # Email features
        df = df.withColumn("email_match",
                           when((col("P_emaildomain") == col("R_emaildomain")) &
                                col("P_emaildomain").isNotNull(), 1).otherwise(0))

        risky_domains = ['anonymous.com', 'mailinator.com', 'tempmail.com']
        df = df.withColumn("email_is_risky",
                           when(col("P_emaildomain").isin(risky_domains), 1).otherwise(0))

        generic_domains = ['gmail.com', 'yahoo.com', 'hotmail.com']
        df = df.withColumn("email_is_generic",
                           when(col("P_emaildomain").isin(generic_domains), 1).otherwise(0))

        return df

    def run_inference(self):
        """Main pipeline execution with enhanced outputs"""
        import pandas as pd

        # Read from Kafka
        df = self.read_from_kafka()

        # Normalize to internal schema
        df = self.normalize_features(df)

        # Add watermark
        df = df.withWatermark("timestamp", "24 hours")

        # Add features
        feature_df = self.add_features(df)

        # Get broadcast references
        broadcast_model = self.broadcast_model
        broadcast_pipeline = self.broadcast_pipeline
        threshold = self.threshold
        risk_high = self.risk_bands["high"]
        risk_medium = self.risk_bands["medium"]

        # Define prediction UDF with probability and risk level
        @pandas_udf("struct<probability:double,prediction:int,risk_level:string,decision:string>")
        def predict_with_risk_udf(
            transaction_id: pd.Series,
            amount: pd.Series,
            card1: pd.Series,
            card2: pd.Series,
            addr1: pd.Series,
            P_emaildomain: pd.Series,
            R_emaildomain: pd.Series,
            ProductCD: pd.Series,
            transaction_hour: pd.Series,
            is_weekend: pd.Series,
            is_night: pd.Series,
            email_match: pd.Series,
            email_is_risky: pd.Series,
            email_is_generic: pd.Series
        ) -> pd.DataFrame:
            """
            Vectorized UDF for fraud prediction with risk levels

            Returns struct with:
            - probability: Fraud probability [0, 1]
            - prediction: Binary prediction (0=legit, 1=fraud)
            - risk_level: HIGH/MEDIUM/LOW
            - decision: BLOCK/APPROVE
            """
            # Build input dataframe matching training features
            input_df = pd.DataFrame({
                "TransactionAmt": amount,
                "log_amt": np.log1p(amount),
                "sqrt_amt": np.sqrt(amount),
                "card1": card1,
                "card2": card2,
                "addr1": addr1,
                "P_emaildomain": P_emaildomain.fillna("unknown"),
                "R_emaildomain": R_emaildomain.fillna("unknown"),
                "transaction_hour": transaction_hour,
                "transaction_day_of_week": 1,  # Placeholder
                "is_weekend": is_weekend,
                "is_night": is_night,
                "email_match": email_match,
                "email_is_risky": email_is_risky,
                "email_is_generic": email_is_generic,
                "ProductCD": ProductCD.fillna("W")
            })

            try:
                model = broadcast_model.value
                pipeline = broadcast_pipeline.value

                # Apply feature pipeline if available
                if pipeline is not None:
                    input_transformed = pipeline.transform(input_df)
                else:
                    # Fallback: basic encoding
                    input_transformed = input_df.copy()
                    for col in input_transformed.select_dtypes(include=['object']).columns:
                        input_transformed[col] = input_transformed[col].astype('category').cat.codes

                # Get probabilities
                if model is not None and hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(input_transformed)[:, 1]
                else:
                    # Fallback: simple heuristic (not production-ready)
                    probabilities = (amount > 1000).astype(float) * 0.3

                # Apply threshold
                predictions = (probabilities >= threshold).astype(int)

                # Assign risk levels
                risk_levels = np.where(
                    probabilities >= risk_high, "HIGH",
                    np.where(probabilities >= risk_medium, "MEDIUM", "LOW")
                )

                # Assign decisions
                decisions = np.where(predictions == 1, "BLOCK", "APPROVE")

                # Return as DataFrame with struct schema
                return pd.DataFrame({
                    "probability": probabilities,
                    "prediction": predictions,
                    "risk_level": risk_levels,
                    "decision": decisions
                })

            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                # Return safe defaults
                n = len(amount)
                return pd.DataFrame({
                    "probability": [0.0] * n,
                    "prediction": [0] * n,
                    "risk_level": ["LOW"] * n,
                    "decision": ["APPROVE"] * n
                })

        # Apply predictions
        prediction_df = feature_df.withColumn(
            "prediction_result",
            predict_with_risk_udf(
                col("transaction_id"),
                col("amount"),
                col("card1"),
                col("card2"),
                col("addr1"),
                col("P_emaildomain"),
                col("R_emaildomain"),
                col("ProductCD"),
                col("transaction_hour"),
                col("is_weekend"),
                col("is_night"),
                col("email_match"),
                col("email_is_risky"),
                col("email_is_generic")
            )
        )

        # Extract fields from struct
        prediction_df = prediction_df.withColumn("probability", col("prediction_result.probability"))
        prediction_df = prediction_df.withColumn("prediction", col("prediction_result.prediction"))
        prediction_df = prediction_df.withColumn("risk_level", col("prediction_result.risk_level"))
        prediction_df = prediction_df.withColumn("decision", col("prediction_result.decision"))

        # Prepare output payload
        output_df = prediction_df.select(
            col("transaction_id"),
            col("probability"),
            col("risk_level"),
            col("decision"),
            col("timestamp"),
            col("amount"),
            col("user_id"),
            col("merchant"),
            col("location")
        )

        # Split into fraud and legit streams
        fraud_df = output_df.filter(col("prediction") == 1)
        legit_df = output_df.filter(col("prediction") == 0)

        # Kafka configuration
        kafka_cfg = self.kafka_config_dict

        # Write fraud predictions to fraud_predictions topic
        fraud_query = (fraud_df.selectExpr(
            "CAST(transaction_id AS STRING) AS key",
            "to_json(struct(*)) AS value"
        )
        .writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", kafka_cfg["bootstrap_servers"])
        .option("topic", self.config["kafka"]["output_topic"])
        .option("kafka.security.protocol", kafka_cfg["security_protocol"])
        .option("kafka.sasl.mechanism", kafka_cfg["sasl_mechanism"])
        .option("kafka.sasl.jaas.config", kafka_cfg["sasl_jaas_config"])
        .option("checkpointLocation", "checkpoints/fraud_checkpoint")
        .outputMode("append")
        .start())

        # Write legit predictions to legit_predictions topic
        legit_query = (legit_df.selectExpr(
            "CAST(transaction_id AS STRING) AS key",
            "to_json(struct(*)) AS value"
        )
        .writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", kafka_cfg["bootstrap_servers"])
        .option("topic", self.config["kafka"]["legit_topic"])
        .option("kafka.security.protocol", kafka_cfg["security_protocol"])
        .option("kafka.sasl.mechanism", kafka_cfg["sasl_mechanism"])
        .option("kafka.sasl.jaas.config", kafka_cfg["sasl_jaas_config"])
        .option("checkpointLocation", "checkpoints/legit_checkpoint")
        .outputMode("append")
        .start())

        logger.info("Streaming queries started. Writing to fraud_predictions and legit_predictions topics.")

        # Await termination
        fraud_query.awaitTermination()


if __name__ == "__main__":
    inference = EnhancedFraudDetectionInference("/app/config.yaml")
    inference.run_inference()
