"""
Enhanced Real-time Fraud Detection Inference Pipeline - NO VAE VERSION

This version removes VAE components while preserving core functionality:
1. Feature names match training exactly (dt_is_weekend, dt_is_night, email_risky, etc.)
2. Real amount values are used (not placeholders)
3. Feature pipeline has proper transform() method
4. Exception handling doesn't silently approve transactions
5. Enhanced risk assessment with:
   - Velocity monitoring
   - Rule-based detection
   - Granular decision logic
"""

# Standard library imports
import logging
import os
import json
from datetime import datetime
from collections import deque

# Third-party imports
import numpy as np
import pandas as pd
import joblib
import yaml
from dotenv import load_dotenv

# Spark imports
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

# Import velocity service and ATO detection
from velocity_service import VelocityFeatureService
from ato_detection_service import ATODetectionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Constants matching training
RISKY_DOMAINS = {
    'anonymous.com', 'mailinator.com', 'tempmail.com', 'dispostable.com',
    'yopmail.com', '10minutemail.com', 'guerrillamail.com'
}


# ==================== ADAPTIVE THRESHOLD SYSTEM ====================
class AdaptiveThresholdSystem:
    """
    Adaptive threshold that adjusts based on recent fraud rates
    and velocity risk patterns (copied from training for compatibility)
    """
    def __init__(self, base_threshold=0.5, window_size=1000,
                 min_threshold=0.1, max_threshold=0.9):
        self.base_threshold = base_threshold
        self.window_size = window_size
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.recent_predictions = deque(maxlen=window_size)
        self.recent_true_labels = deque(maxlen=window_size)
        self.current_threshold = base_threshold
        self.threshold_history = []

    def update(self, y_true, y_pred_proba, velocity_risk):
        """Update threshold based on recent performance"""
        self.recent_predictions.append(y_pred_proba)
        self.recent_true_labels.append(y_true)

        if len(self.recent_predictions) < 100:
            return self.current_threshold

        recent_fraud_rate = np.mean(self.recent_true_labels)
        recent_preds_binary = np.array(self.recent_predictions) >= self.current_threshold
        recent_accuracy = np.mean(recent_preds_binary == np.array(self.recent_true_labels))

        adjustment = 0
        if recent_fraud_rate > 0.05:
            adjustment = -0.02
        elif recent_fraud_rate < 0.02:
            adjustment = 0.02

        if velocity_risk > 0.7:
            adjustment -= 0.01

        self.current_threshold = np.clip(
            self.current_threshold + adjustment,
            self.min_threshold,
            self.max_threshold
        )

        self.threshold_history.append({
            'threshold': self.current_threshold,
            'fraud_rate': recent_fraud_rate,
            'accuracy': recent_accuracy
        })

        return self.current_threshold

    def get_threshold(self, velocity_risk=0.0):
        """Get current threshold, adjusted for velocity risk"""
        adjusted = self.current_threshold

        if velocity_risk > 0.8:
            adjusted *= 0.9
        elif velocity_risk > 0.6:
            adjusted *= 0.95

        return np.clip(adjusted, self.min_threshold, self.max_threshold)

    def get_hybrid_threshold(self, velocity_risk, amount_risk=0.0, 
                            method='weighted', w1=0.6, w2=0.3, w3=0.1):
        """
        Hybrid threshold combining F1-optimal with risk-based adjustments
        
        Industry best practice: τ_hybrid = w₁·τ_A + w₂·τ_V + w₃·τ_Amount
        
        Args:
            velocity_risk: 0-1 velocity risk score
            amount_risk: 0-1 amount risk score
            method: 'weighted', 'dynamic', 'max', or 'min'
            w1, w2, w3: Weights (sum to 1.0)
        
        Returns:
            float: Hybrid threshold
        """
        tau_A = self.current_threshold
        tau_V = self.base_threshold - (velocity_risk * 0.15)
        tau_Amount = self.base_threshold - (amount_risk * 0.10)
        
        if method == 'max':
            tau = max(tau_A, tau_V, tau_Amount)
        elif method == 'min':
            tau = min(tau_A, tau_V, tau_Amount)
        elif method == 'dynamic':
            if velocity_risk > 0.7:
                w1, w2, w3 = 0.3, 0.5, 0.2
            elif amount_risk > 0.7:
                w1, w2, w3 = 0.3, 0.2, 0.5
            else:
                w1, w2, w3 = 0.6, 0.2, 0.2
            tau = w1 * tau_A + w2 * tau_V + w3 * tau_Amount
        else:  # weighted
            tau = w1 * tau_A + w2 * tau_V + w3 * tau_Amount
        
        return np.clip(tau, self.min_threshold, self.max_threshold)


class EnhancedFraudDetectionInference:
    """
    Enhanced fraud detection inference with FIXED train/serve consistency
    """

    def __init__(self, config_path="/app/config.yaml"):
        """Initialize enhanced inference pipeline"""
        load_dotenv(dotenv_path="/app/.env")

        self.config = self._load_config(config_path)
        
        # Check MLflow connectivity first
        self._check_mlflow_connectivity()
        
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
        if self.model and 'adaptive_threshold_system' in self.model:
            self.adaptive_threshold_system = self.model['adaptive_threshold_system']
            self.threshold = self.adaptive_threshold_system.current_threshold
            logger.info(f"✓ Using adaptive threshold from trained model: {self.threshold:.4f}")
        else:
            self.adaptive_threshold_system = None
            self.threshold = self.config["inference"]["threshold"]
            logger.info(f"Using fixed threshold from config: {self.threshold}")

        self.risk_bands = self.config["inference"]["risk_bands"]
        logger.info(f"Risk bands: HIGH={self.risk_bands['high']}, MEDIUM={self.risk_bands['medium']}")
        
    def _check_mlflow_connectivity(self):
        """Check MLflow server connectivity and model availability"""
        if not self.config.get("mlflow", {}).get("tracking_uri"):
            logger.warning("MLflow not configured, will use local models")
            return False

        try:
            import mlflow
            
            # Try each tracking URI in order
            tracking_uris = self.config["mlflow"]["tracking_uri"]
            if isinstance(tracking_uris, str):
                tracking_uris = [tracking_uris]
                
            for uri in tracking_uris:
                try:
                    mlflow.set_tracking_uri(uri)
                    # Test connection by listing experiments
                    mlflow.search_experiments()
                    logger.info(f"✓ Connected to MLflow server at {uri}")
                    
                    # Check if our model exists
                    model_name = self.config["mlflow"]["registered_model_name"]
                    stage = self.config["mlflow"].get("model_stage", "Production")
                    
                    try:
                        model_versions = mlflow.tracking.MlflowClient().get_latest_versions(model_name, stages=[stage])
                        if model_versions:
                            logger.info(f"✓ Found model {model_name} in stage {stage}")
                            return True
                        else:
                            logger.warning(f"No {stage} version found for model {model_name}")
                    except Exception as e:
                        logger.warning(f"Could not verify model in MLflow: {str(e)}")
                        
                except Exception as e:
                    logger.warning(f"Could not connect to MLflow at {uri}: {str(e)}")
                    continue
                    
            logger.warning("Could not connect to any MLflow server, will use local models")
            return False
                
        except ImportError:
            logger.warning("MLflow package not available, will use local models")
            return False
        except Exception as e:
            logger.error(f"Error checking MLflow: {str(e)}")
            return False

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
        """Load trained model bundle with MLflow fallback"""
        try:
            # First try MLflow if configured
            if self.config.get("mlflow", {}).get("tracking_uri"):
                try:
                    import mlflow
                    mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
                    
                    # Try to load latest model from MLflow
                    model_name = self.config["mlflow"]["registered_model_name"]
                    stage = "Production"
                    
                    try:
                        model_bundle = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")
                        logger.info(f"✓ Model loaded from MLflow: {model_name} ({stage})")
                        return model_bundle
                    except Exception as e:
                        logger.warning(f"Could not load model from MLflow: {str(e)}")
                        logger.warning("Falling back to local model file")
                except ImportError:
                    logger.warning("MLflow not available, falling back to local model file")
                except Exception as e:
                    logger.warning(f"MLflow error: {str(e)}, falling back to local model file")

            # Fallback to local file
            if not os.path.exists(model_path):
                # Try to find model in common locations
                common_paths = [
                    model_path,
                    os.path.join(os.path.dirname(__file__), "models", "fraud_detection_model.pkl"),
                    "/app/models/fraud_detection_model.pkl",
                    "models/fraud_detection_model.pkl"
                ]
                
                for path in common_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
                else:
                    logger.error("No model found in any location")
                    return None

            # Load local model bundle
            model_bundle = joblib.load(model_path)
            logger.info(f"✓ Model loaded from local file: {model_path}")
            return model_bundle

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
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

            # Verify it has transform method
            if hasattr(pipeline, 'transform'):
                logger.info(f"  ✓ Pipeline has transform() method")
            else:
                logger.warning(f"  ✗ Pipeline missing transform() method - using fallback")

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

        # Define schema for incoming transactions
        json_schema = StructType([
            StructField("transaction_id", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("user_id", IntegerType(), True),
            StructField("amount", DoubleType(), True),
            StructField("currency", StringType(), True),
            StructField("merchant", StringType(), True),
            StructField("location", StringType(), True),
            # IEEE-CIS fields
            StructField("TransactionID", StringType(), True),
            StructField("TransactionDT", DoubleType(), True),
            StructField("TransactionAmt", DoubleType(), True),
            StructField("ProductCD", StringType(), True),
            StructField("card1", IntegerType(), True),
            StructField("card2", IntegerType(), True),
            StructField("card3", IntegerType(), True),
            StructField("card4", StringType(), True),
            StructField("card5", IntegerType(), True),
            StructField("card6", StringType(), True),
            StructField("addr1", IntegerType(), True),
            StructField("addr2", IntegerType(), True),
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
        """Normalize incoming data to unified internal schema"""
        logger.info("Normalizing features...")

        # Map TransactionID to transaction_id
        df = df.withColumn(
            "transaction_id",
            coalesce(col("transaction_id"), col("TransactionID"))
        )

        # Map TransactionAmt to amount - KEEP REAL VALUES (NO PLACEHOLDERS)
        df = df.withColumn(
            "amount",
            coalesce(col("amount"), col("TransactionAmt"))
        )

        # Map TransactionDT to timestamp if needed
        df = df.withColumn(
            "timestamp",
            coalesce(
                col("timestamp"),
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

        # IEEE-CIS specific fields
        df = df.withColumn("card1", coalesce(col("card1"), lit(-1)))
        df = df.withColumn("card2", coalesce(col("card2"), lit(-1)))
        df = df.withColumn("card3", coalesce(col("card3"), lit(-1)))
        df = df.withColumn("card4", coalesce(col("card4"), lit("visa")))
        df = df.withColumn("card5", coalesce(col("card5"), lit(-1)))
        df = df.withColumn("card6", coalesce(col("card6"), lit("debit")))
        df = df.withColumn("addr1", coalesce(col("addr1"), lit(-1)))
        df = df.withColumn("addr2", coalesce(col("addr2"), lit(-1)))
        df = df.withColumn("P_emaildomain", coalesce(col("P_emaildomain"), lit(None)))
        df = df.withColumn("R_emaildomain", coalesce(col("R_emaildomain"), lit(None)))
        df = df.withColumn("ProductCD", coalesce(col("ProductCD"), lit("W")))

        # Create TransactionAmt column (maps from amount) - CRITICAL FOR FEATURE ENGINEERING
        df = df.withColumn("TransactionAmt", col("amount"))

        return df

    def add_features(self, df):
        """
        Add engineered features EXACTLY matching training feature names

        CRITICAL: Feature names must match training exactly:
        - dt_is_weekend (not is_weekend)
        - dt_is_night (not is_night)
        - dt_hour (not transaction_hour)
        - email_risky (not email_is_risky)
        - log_TransactionAmt (not log_amt)
        - sqrt_TransactionAmt (not sqrt_amt)
        """
        # Temporal features - MATCH TRAINING NAMES EXACTLY
        df = df.withColumn("dt_hour", hour(col("timestamp")))
        df = df.withColumn("dt_wday", dayofweek(col("timestamp")))
        df = df.withColumn("dt_day", dayofmonth(col("timestamp")))
        df = df.withColumn("dt_is_weekend",
                           when((col("dt_wday") == 1) | (col("dt_wday") == 7), 1).otherwise(0))
        df = df.withColumn("dt_is_night",
                           when((col("dt_hour") >= 22) | (col("dt_hour") <= 6), 1).otherwise(0))

        # Amount features - MATCH TRAINING NAMES EXACTLY (USE REAL VALUES, NOT PLACEHOLDERS)
        # Note: We compute these in Spark SQL but also recompute in pandas UDF for consistency
        df = df.withColumn("log_TransactionAmt", lit(0.0))  # Placeholder, will be computed in UDF
        df = df.withColumn("sqrt_TransactionAmt", lit(0.0))  # Placeholder, will be computed in UDF

        # Email features - MATCH TRAINING NAMES EXACTLY
        df = df.withColumn("email_match",
                           when((col("P_emaildomain") == col("R_emaildomain")) &
                                col("P_emaildomain").isNotNull(), 1).otherwise(0))

        # CRITICAL: Use 'email_risky' NOT 'email_is_risky'
        risky_domains_list = list(RISKY_DOMAINS)
        df = df.withColumn("email_risky",
                           when(col("P_emaildomain").isin(risky_domains_list), 1).otherwise(0))

        generic_domains = ['gmail.com', 'yahoo.com', 'hotmail.com']
        df = df.withColumn("email_is_generic",
                           when(col("P_emaildomain").isin(generic_domains), 1).otherwise(0))

        return df

    def run_inference(self):
        """Main pipeline execution with enhanced outputs"""
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
        @pandas_udf("struct<probability:double,prediction:int,risk_level:string,decision:string,risk_factors:string>")
        def predict_with_risk_udf(
            transaction_id: pd.Series,
            TransactionAmt: pd.Series,
            log_TransactionAmt: pd.Series,
            sqrt_TransactionAmt: pd.Series,
            card1: pd.Series,
            card2: pd.Series,
            card3: pd.Series,
            card4: pd.Series,
            card5: pd.Series,
            card6: pd.Series,
            addr1: pd.Series,
            addr2: pd.Series,
            P_emaildomain: pd.Series,
            R_emaildomain: pd.Series,
            ProductCD: pd.Series,
            dt_hour: pd.Series,
            dt_wday: pd.Series,
            dt_day: pd.Series,
            dt_is_weekend: pd.Series,
            dt_is_night: pd.Series,
            email_match: pd.Series,
            email_risky: pd.Series,
            email_is_generic: pd.Series
        ) -> pd.DataFrame:
            """
            Vectorized UDF for fraud prediction with FIXED feature engineering

            CRITICAL FIXES:
            1. Feature names match training exactly
            2. VAE anomaly scores are computed
            3. Real amount values (not placeholders)
            4. Proper exception handling that doesn't approve fraudulent transactions
            """
            n = len(TransactionAmt)

            try:
                model_bundle = broadcast_model.value
                pipeline = broadcast_pipeline.value

                if model_bundle is None:
                    logger.error("Model bundle is None")
                    raise ValueError("Model not loaded")

                # Build base input dataframe with EXACT training feature names
                input_df = pd.DataFrame({
                    'TransactionAmt': TransactionAmt.fillna(0).astype('float32'),
                    'log_TransactionAmt': np.log1p(TransactionAmt.fillna(0)).astype('float32'),
                    'sqrt_TransactionAmt': np.sqrt(TransactionAmt.fillna(0)).astype('float32'),
                    'card1': card1.fillna(-1).astype('int32'),
                    'card2': card2.fillna(-1).astype('int32'),
                    'card3': card3.fillna(-1).astype('int32'),
                    'card4': card4.fillna("visa"),
                    'card5': card5.fillna(-1).astype('int32'),
                    'card6': card6.fillna("debit"),
                    'addr1': addr1.fillna(-1).astype('int32'),
                    'addr2': addr2.fillna(-1).astype('int32'),
                    'P_emaildomain': P_emaildomain.fillna("unknown"),
                    'R_emaildomain': R_emaildomain.fillna("unknown"),
                    'ProductCD': ProductCD.fillna("W"),
                    'dt_hour': dt_hour.fillna(0).astype('int16'),
                    'dt_wday': dt_wday.fillna(0).astype('int8'),
                    'dt_day': dt_day.fillna(1).astype('int32'),
                    'dt_is_weekend': dt_is_weekend.fillna(0).astype('int8'),
                    'dt_is_night': dt_is_night.fillna(0).astype('int8'),
                    'email_match': email_match.fillna(0).astype('int8'),
                    'email_risky': email_risky.fillna(0).astype('int8'),
                    'email_is_generic': email_is_generic.fillna(0).astype('int8'),
                })

                # ✅ CRITICAL: Compute velocity features using Redis-backed distributed service
                import time
                import os
                from velocity_service import VelocityFeatureService

                current_time = time.time()

                # Add timestamp column for velocity computation
                input_df['timestamp'] = current_time

                # Create velocity service instance with Redis support
                # Each Spark worker shares Redis state (distributed tracking)
                redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
                velocity_service = VelocityFeatureService(
                    redis_url=redis_url,
                    max_history_seconds=7 * 24 * 3600,  # 7 days
                    use_fallback=True  # Enable local fallback on Redis failure
                )
                
                # Create ATO detection service instance with Redis support
                ato_service = ATODetectionService(
                    redis_url=redis_url,
                    history_days=90,  # 90-day behavioral history
                    connection_timeout=2
                )
                
                # Log service health on first use
                velocity_health = velocity_service.health_check()
                ato_health = ato_service.health_check()
                if not velocity_health['redis_connected']:
                    logger.warning(f"Velocity service health: {velocity_health}")
                if not ato_health['redis_connected']:
                    logger.warning(f"ATO service health: {ato_health}")
                
                # ✅ NEW: Multi-entity velocity features (card + user + device levels)
                # This provides comprehensive fraud detection across:
                # - Card-level: Traditional card+addr+email tracking
                # - User-level: Detect multi-card fraud (same user, different cards)
                # - Device-level: Detect fraud rings (same device, multiple accounts)
                input_df_with_velocity = velocity_service.process_batch(input_df, multi_entity=True)

                # Remove temporary timestamp column
                input_df = input_df_with_velocity.drop(columns=['timestamp'], errors='ignore')
                
                # ✅ ACCOUNT TAKEOVER (ATO) DETECTION
                # Analyze each transaction for ATO patterns (geo-velocity, device changes, etc.)
                ato_risk_scores = []
                ato_confidences = []
                ato_flags = []
                
                for i in range(n):
                    # Extract transaction data for ATO analysis
                    transaction = {
                        'card1': card1.iloc[i] if i < len(card1) else None,
                        'TransactionAmt': TransactionAmt.iloc[i] if i < len(TransactionAmt) else 0.0,
                    }
                    
                    # Extract optional fields (device, location, session)
                    # These would come from enhanced data collection in production
                    device_id = None  # TODO: Extract from transaction metadata
                    device_type = None
                    session_id = None
                    latitude = None
                    longitude = None
                    
                    # Perform ATO analysis
                    ato_result = ato_service.analyze_transaction(
                        transaction=transaction,
                        latitude=latitude,
                        longitude=longitude,
                        device_id=device_id,
                        device_type=device_type,
                        session_id=session_id
                    )
                    
                    # Store ATO detection results
                    ato_risk_scores.append(ato_result['ato_risk_score'])
                    ato_confidences.append(ato_result['ato_confidence'])
                    ato_flags.append(1 if ato_result['ato_detected'] else 0)
                
                # Add ATO features to dataframe
                input_df['ato_risk_score'] = ato_risk_scores
                input_df['ato_confidence'] = ato_confidences
                input_df['ato_detected'] = ato_flags

                # Apply frequency encoding using pipeline
                if pipeline is not None and hasattr(pipeline, 'transform'):
                    # Get base features for VAE
                    base_features = input_df.copy()

                    # Apply frequency encoding
                    input_transformed = pipeline.transform(input_df)

                    # Enhanced risk assessment without VAE
                    # Velocity-based patterns already captured above
                    # Rule-based flags will catch obvious fraud patterns
                    pass
                else:
                    # Fallback: basic encoding
                    logger.warning("Pipeline transform not available, using fallback")
                    input_transformed = input_df.copy()
                    for col_name in input_transformed.select_dtypes(include=['object']).columns:
                        input_transformed[col_name] = input_transformed[col_name].astype('category').cat.codes

                # Get calibrated model
                calibrated_model = model_bundle.get('calibrated_model')
                if calibrated_model is None:
                    raise ValueError("Calibrated model not found in bundle")

                # Get probabilities (handle both single models and stacked ensembles)
                if isinstance(calibrated_model, dict) and calibrated_model.get('type') == 'stacked_ensemble':
                    # 🔥 STACKED ENSEMBLE: Get predictions from all base models + meta-learner
                    base_models = calibrated_model['base_models']
                    meta_model = calibrated_model['meta_model']
                    
                    # Generate meta-features from base models
                    meta_features = np.zeros((len(input_transformed), len(base_models)))
                    for idx, (name, base_model) in enumerate(base_models):
                        meta_features[:, idx] = base_model.predict_proba(input_transformed)[:, 1]
                    
                    # Get final prediction from meta-learner
                    probabilities = meta_model.predict_proba(meta_features)[:, 1]
                else:
                    # Single model (calibrated or uncalibrated)
                    probabilities = calibrated_model.predict_proba(input_transformed)[:, 1]

                # RULE-BASED OVERRIDES: Flag obvious fraud patterns even if model probability is low
                # This addresses model threshold being too conservative
                rule_based_flags = np.zeros(n, dtype=bool)

                for i in range(n):
                    # Flag 1: Risky email domains (anonymous.com, mailinator.com, etc.)
                    if email_risky.iloc[i] == 1:
                        rule_based_flags[i] = True

                    # Flag 2: Very low amount (<$1) transactions
                    if TransactionAmt.iloc[i] < 1.0:
                        rule_based_flags[i] = True

                    # Flag 3: Very high amount (>$10k) transactions
                    if TransactionAmt.iloc[i] > 10000:
                        rule_based_flags[i] = True

                    # Flag 4: High velocity risk from velocity service
                    if 'velocity_risk_score' in input_df.columns:
                        velocity_risk = input_df.loc[input_df.index[i], 'velocity_risk_score']
                        if velocity_risk > 0.7:
                            rule_based_flags[i] = True
                    
                    # Flag 5: ATO detected (CRITICAL - Account Takeover)
                    if 'ato_detected' in input_df.columns:
                        ato_detected = input_df.loc[input_df.index[i], 'ato_detected']
                        if ato_detected == 1:
                            rule_based_flags[i] = True

                # Boost probabilities for rule-based flags (ensure they get flagged)
                adjusted_probabilities = probabilities.copy()
                adjusted_probabilities[rule_based_flags] = np.maximum(
                    adjusted_probabilities[rule_based_flags],
                    0.15  # Minimum probability for rule-based flags (above typical thresholds)
                )

                # ========== HYBRID THRESHOLD APPLICATION ==========
                # Calculate per-transaction adaptive thresholds
                per_transaction_thresholds = np.zeros(n)
                
                for i in range(n):
                    # Calculate velocity risk
                    velocity_risk = 0.0
                    if 'velocity_risk_score' in input_df.columns:
                        velocity_risk = input_df.loc[input_df.index[i], 'velocity_risk_score']
                    
                    # Calculate ATO risk (highest priority)
                    ato_risk = 0.0
                    if 'ato_risk_score' in input_df.columns:
                        ato_risk = input_df.loc[input_df.index[i], 'ato_risk_score']
                    
                    # Calculate amount risk
                    amount = TransactionAmt.iloc[i]
                    if amount > 1000:
                        amount_risk = min(0.9, 0.5 + (amount - 1000) / 10000)
                    elif amount > 500:
                        amount_risk = 0.7
                    elif amount > 200:
                        amount_risk = 0.5
                    elif amount < 1.0:
                        amount_risk = 0.9  # Very low amounts are suspicious
                    else:
                        amount_risk = 0.1
                    
                    # Use hybrid threshold if adaptive system available
                    if hasattr(broadcast_model.value, 'get') and \
                       'adaptive_threshold_system' in broadcast_model.value and \
                       broadcast_model.value['adaptive_threshold_system'] is not None:
                        adaptive_system = broadcast_model.value['adaptive_threshold_system']
                        per_transaction_thresholds[i] = adaptive_system.get_hybrid_threshold(
                            velocity_risk=velocity_risk,
                            amount_risk=amount_risk,
                            method='dynamic'  # Use dynamic weights
                        )
                    else:
                        # Fallback: manual hybrid calculation with ATO integration
                        base_threshold = threshold
                        tau_V = base_threshold - (velocity_risk * 0.15)
                        tau_Amount = base_threshold - (amount_risk * 0.10)
                        tau_ATO = base_threshold - (ato_risk * 0.25)  # ATO has highest weight
                        
                        # Dynamic weights (ATO takes priority)
                        if ato_risk > 0.8:
                            # Critical ATO risk - aggressive threshold
                            w1, w2, w3, w4 = 0.1, 0.2, 0.1, 0.6  # 60% weight to ATO
                        elif velocity_risk > 0.7:
                            w1, w2, w3, w4 = 0.2, 0.4, 0.2, 0.2
                        elif amount_risk > 0.7:
                            w1, w2, w3, w4 = 0.2, 0.2, 0.4, 0.2
                        else:
                            w1, w2, w3, w4 = 0.5, 0.2, 0.1, 0.2
                        
                        tau_hybrid = w1 * base_threshold + w2 * tau_V + w3 * tau_Amount + w4 * tau_ATO
                        per_transaction_thresholds[i] = np.clip(tau_hybrid, 0.10, 0.85)  # Lower min for ATO

                # Apply per-transaction thresholds (using adjusted probabilities)
                predictions = (adjusted_probabilities >= per_transaction_thresholds).astype(int)

                # Also flag as fraud if rule-based criteria are met, regardless of threshold
                predictions[rule_based_flags] = 1

                # Assign risk levels (use original probabilities for display, but flag high risk)
                risk_levels = np.where(
                    adjusted_probabilities >= risk_high, "HIGH",
                    np.where(adjusted_probabilities >= risk_medium, "MEDIUM", "LOW")
                )
                # Ensure rule-based flags get at least MEDIUM risk
                risk_levels[rule_based_flags & (risk_levels == "LOW")] = "MEDIUM"

                # Assign decisions with more granularity
                decisions = np.where(
                    adjusted_probabilities >= 0.9, "BLOCK",
                    np.where(adjusted_probabilities >= 0.7, "HOLD",
                             np.where(predictions == 1, "REVIEW", "APPROVE"))
                )

                # Generate risk factors (including velocity-based, rule-based, and threshold info)
                risk_factors_list = []
                for i in range(n):
                    factors = []
                    # Collect risk factors for any flagged transaction (model-based OR rule-based)
                    if predictions[i] == 1 or probabilities[i] >= per_transaction_thresholds[i] or rule_based_flags[i]:
                        # Threshold adjustment info
                        threshold_diff = per_transaction_thresholds[i] - threshold
                        if abs(threshold_diff) > 0.05:
                            if threshold_diff < 0:
                                factors.append(f"threshold_lowered_{abs(threshold_diff):.2f}")
                            else:
                                factors.append(f"threshold_raised_{threshold_diff:.2f}")
                        
                        # Email-based risk
                        if email_risky.iloc[i] == 1:
                            factors.append("risky_email_domain")

                        # Amount-based risk
                        if TransactionAmt.iloc[i] > 10000:
                            factors.append("very_high_amount")
                        elif TransactionAmt.iloc[i] > 1000:
                            factors.append("high_amount")
                        if TransactionAmt.iloc[i] < 1.0:
                            factors.append("very_low_amount")

                        # Time-based risk
                        if dt_is_night.iloc[i] == 1:
                            factors.append("night_transaction")

                        # Velocity-based risk
                        if 'velocity_risk_score' in input_df.columns:
                            velocity_risk = input_df.loc[input_df.index[i], 'velocity_risk_score']
                            if velocity_risk > 0.7:
                                factors.append("high_velocity_risk")
                            if 'txn_count_1h' in input_df.columns and input_df.loc[input_df.index[i], 'txn_count_1h'] > 5:
                                factors.append("rapid_transactions")
                            if 'amt_spike_1h' in input_df.columns and input_df.loc[input_df.index[i], 'amt_spike_1h'] > 0.8:
                                factors.append("amount_spike")
                        
                        # ATO-based risk (HIGHEST PRIORITY)
                        if 'ato_detected' in input_df.columns and input_df.loc[input_df.index[i], 'ato_detected'] == 1:
                            ato_confidence = input_df.loc[input_df.index[i], 'ato_confidence']
                            factors.append(f"ACCOUNT_TAKEOVER_{ato_confidence}")
                        if 'ato_risk_score' in input_df.columns:
                            ato_risk = input_df.loc[input_df.index[i], 'ato_risk_score']
                            if ato_risk > 0.8:
                                factors.append("critical_ato_risk")
                            elif ato_risk > 0.6:
                                factors.append("high_ato_risk")

                        # Rule-based flag indicator
                        if rule_based_flags[i]:
                            factors.append("rule_based_flag")

                    risk_factors_list.append(",".join(factors) if factors else "none")

                # Return as DataFrame with struct schema
                return pd.DataFrame({
                    "probability": probabilities,
                    "prediction": predictions,
                    "risk_level": risk_levels,
                    "decision": decisions,
                    "risk_factors": risk_factors_list
                })

            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")

                # CRITICAL: Don't approve suspicious transactions on error
                # Instead, flag them for manual review
                return pd.DataFrame({
                    "probability": [0.5] * n,  # Neutral probability
                    "prediction": [0] * n,
                    "risk_level": ["MEDIUM"] * n,  # Medium risk for safety
                    "decision": ["REVIEW"] * n,  # Manual review required
                    "risk_factors": ["processing_error"] * n
                })

        # Apply predictions
        prediction_df = feature_df.withColumn(
            "prediction_result",
            predict_with_risk_udf(
                col("transaction_id"),
                col("TransactionAmt"),
                col("log_TransactionAmt"),
                col("sqrt_TransactionAmt"),
                col("card1"),
                col("card2"),
                col("card3"),
                col("card4"),
                col("card5"),
                col("card6"),
                col("addr1"),
                col("addr2"),
                col("P_emaildomain"),
                col("R_emaildomain"),
                col("ProductCD"),
                col("dt_hour"),
                col("dt_wday"),
                col("dt_day"),
                col("dt_is_weekend"),
                col("dt_is_night"),
                col("email_match"),
                col("email_risky"),
                col("email_is_generic")
            )
        )

        # Extract fields from struct
        prediction_df = prediction_df.withColumn("probability", col("prediction_result.probability"))
        prediction_df = prediction_df.withColumn("prediction", col("prediction_result.prediction"))
        prediction_df = prediction_df.withColumn("risk_level", col("prediction_result.risk_level"))
        prediction_df = prediction_df.withColumn("decision", col("prediction_result.decision"))
        prediction_df = prediction_df.withColumn("risk_factors", col("prediction_result.risk_factors"))

        # Prepare output payload
        output_df = prediction_df.select(
            col("transaction_id"),
            col("probability"),
            col("risk_level"),
            col("decision"),
            col("risk_factors"),
            col("timestamp"),
            col("amount"),
            col("user_id"),
            col("merchant"),
            col("location")
        )

        # Split into fraud and legit streams
        fraud_df = output_df.filter((col("prediction") == 1) | (col("decision") == "REVIEW"))
        legit_df = output_df.filter((col("prediction") == 0) & (col("decision") == "APPROVE"))

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
