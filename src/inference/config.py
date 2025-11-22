"""
Configuration loader for fraud detection inference service.

Reads from:
1. config.yaml (default configuration)
2. Environment variables (override YAML values)

Usage:
    from config import Config
    config = Config.load()
    kafka_brokers = config.kafka.brokers
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class KafkaConfig:
    """Kafka connection and topic configuration."""
    brokers: str = "localhost:9092"
    input_topic: str = "transactions"
    fraud_output_topic: str = "fraud_predictions"
    legit_output_topic: str = "legit_predictions"
    group_id: str = "fraud-detection-inference"
    max_offsets_per_trigger: int = 1000


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 50
    socket_timeout: int = 5

    # TTL settings (in seconds)
    velocity_ttl: int = 604800  # 7 days
    ato_baseline_ttl: int = 2592000  # 30 days


@dataclass
class ModelConfig:
    """Model artifacts configuration."""
    # Default paths - can be overridden via environment variables
    # For Docker: /app/models/
    # For local dev: src/models/
    model_bundle_path: str = "src/models/fraud_detection_model.pkl"
    feature_pipeline_path: str = "src/models/feature_pipeline.pkl"

    # Base threshold from training (F1-optimal)
    base_threshold: float = 0.0564

    # Decision thresholds (using proper scale)
    approve_threshold: float = 0.0564  # < base_threshold
    review_threshold: float = 0.10     # τ_base ≤ prob < 0.10
    hold_threshold: float = 0.20       # 0.10 ≤ prob < 0.20
    block_threshold: float = 0.20      # prob ≥ 0.20


@dataclass
class ThresholdWeights:
    """Weights for hybrid adaptive threshold calculation."""
    high_ato: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.1, 0.6])
    high_velocity: List[float] = field(default_factory=lambda: [0.2, 0.5, 0.2, 0.1])
    high_amount: List[float] = field(default_factory=lambda: [0.2, 0.2, 0.5, 0.1])
    normal: List[float] = field(default_factory=lambda: [0.6, 0.2, 0.1, 0.1])


@dataclass
class VelocityConfig:
    """Velocity detection thresholds."""
    # Transaction count thresholds
    high_1h_count: int = 5
    high_6h_count: int = 15
    high_24h_count: int = 30

    # Amount spike multipliers
    amount_spike_3x: float = 3.0
    amount_spike_5x: float = 5.0

    # Risk thresholds
    velocity_risk_threshold: float = 0.8
    amount_risk_threshold: float = 0.8


@dataclass
class ATOConfig:
    """Account Takeover detection thresholds."""
    # Distance threshold (km)
    geo_anomaly_distance: float = 1000.0

    # Risk weights
    geo_anomaly_weight: float = 0.30
    new_address_ip_weight: float = 0.20
    email_mismatch_weight: float = 0.35
    high_amount_weight: float = 0.15
    unusual_card_weight: float = 0.10

    # High amount threshold
    high_amount_threshold: float = 2000.0

    # ATO detection threshold
    ato_detection_threshold: float = 0.6

    # Risky email domains
    risky_email_domains: List[str] = field(default_factory=lambda: [
        'anonymous.com', 'mailinator.com', 'tempmail.com', 'dispostable.com',
        'yopmail.com', '10minutemail.com', 'guerrillamail.com', 'protonmail.com',
        'sharklasers.com', 'guerrillamail.info', 'trashmail.com'
    ])


@dataclass
class SparkConfig:
    """Spark configuration."""
    app_name: str = "FraudDetectionInference"
    checkpoint_location: str = "/tmp/spark-checkpoints/fraud-detection"
    trigger_interval: str = "5 seconds"
    shuffle_partitions: int = 200


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class Config:
    """Master configuration object."""
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    threshold_weights: ThresholdWeights = field(default_factory=ThresholdWeights)
    velocity: VelocityConfig = field(default_factory=VelocityConfig)
    ato: ATOConfig = field(default_factory=ATOConfig)
    spark: SparkConfig = field(default_factory=SparkConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'Config':
        """
        Load configuration from YAML file and environment variables.

        Args:
            config_path: Path to config.yaml. If None, looks for config.yaml in:
                        1. ./config.yaml
                        2. ./src/inference/config.yaml
                        3. /app/config/config.yaml

        Returns:
            Config object with merged settings
        """
        config = cls()

        # Try to find config.yaml
        if config_path is None:
            possible_paths = [
                Path("config.yaml"),
                Path("src/inference/config.yaml"),
                Path("/app/config/config.yaml"),
            ]
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break

        # Load from YAML if exists
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                config._merge_yaml(yaml_config)

        # Override with environment variables
        config._merge_env()

        return config

    def _merge_yaml(self, yaml_config: Dict) -> None:
        """Merge YAML configuration into dataclass fields."""
        if not yaml_config:
            return

        # Kafka
        if 'kafka' in yaml_config:
            for key, value in yaml_config['kafka'].items():
                if hasattr(self.kafka, key):
                    setattr(self.kafka, key, value)

        # Redis
        if 'redis' in yaml_config:
            for key, value in yaml_config['redis'].items():
                if hasattr(self.redis, key):
                    setattr(self.redis, key, value)

        # Model
        if 'model' in yaml_config:
            for key, value in yaml_config['model'].items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)

        # Threshold weights
        if 'threshold_weights' in yaml_config:
            for key, value in yaml_config['threshold_weights'].items():
                if hasattr(self.threshold_weights, key):
                    setattr(self.threshold_weights, key, value)

        # Velocity
        if 'velocity' in yaml_config:
            for key, value in yaml_config['velocity'].items():
                if hasattr(self.velocity, key):
                    setattr(self.velocity, key, value)

        # ATO
        if 'ato' in yaml_config:
            for key, value in yaml_config['ato'].items():
                if hasattr(self.ato, key):
                    setattr(self.ato, key, value)

        # Spark
        if 'spark' in yaml_config:
            for key, value in yaml_config['spark'].items():
                if hasattr(self.spark, key):
                    setattr(self.spark, key, value)

        # Logging
        if 'logging' in yaml_config:
            for key, value in yaml_config['logging'].items():
                if hasattr(self.logging, key):
                    setattr(self.logging, key, value)

    def _merge_env(self) -> None:
        """Override configuration with environment variables."""
        # Kafka
        self.kafka.brokers = os.getenv('KAFKA_BROKERS', self.kafka.brokers)
        self.kafka.input_topic = os.getenv('KAFKA_INPUT_TOPIC', self.kafka.input_topic)
        self.kafka.fraud_output_topic = os.getenv('KAFKA_FRAUD_TOPIC', self.kafka.fraud_output_topic)
        self.kafka.legit_output_topic = os.getenv('KAFKA_LEGIT_TOPIC', self.kafka.legit_output_topic)

        # Redis
        self.redis.host = os.getenv('REDIS_HOST', self.redis.host)
        self.redis.port = int(os.getenv('REDIS_PORT', str(self.redis.port)))
        self.redis.password = os.getenv('REDIS_PASSWORD', self.redis.password)

        # Model
        self.model.model_bundle_path = os.getenv('MODEL_BUNDLE_PATH', self.model.model_bundle_path)
        self.model.feature_pipeline_path = os.getenv('FEATURE_PIPELINE_PATH', self.model.feature_pipeline_path)

        # Thresholds (can be tuned in production)
        if os.getenv('BASE_THRESHOLD'):
            self.model.base_threshold = float(os.getenv('BASE_THRESHOLD'))
        if os.getenv('REVIEW_THRESHOLD'):
            self.model.review_threshold = float(os.getenv('REVIEW_THRESHOLD'))
        if os.getenv('HOLD_THRESHOLD'):
            self.model.hold_threshold = float(os.getenv('HOLD_THRESHOLD'))
        if os.getenv('BLOCK_THRESHOLD'):
            self.model.block_threshold = float(os.getenv('BLOCK_THRESHOLD'))

        # Spark
        self.spark.checkpoint_location = os.getenv('SPARK_CHECKPOINT', self.spark.checkpoint_location)

        # Logging
        self.logging.level = os.getenv('LOG_LEVEL', self.logging.level)
