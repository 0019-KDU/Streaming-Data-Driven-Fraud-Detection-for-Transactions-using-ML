# Fraud Detection Inference Service

Production-ready real-time fraud detection inference service using **Spark Structured Streaming**, **Redis**, and **XGBoost**.

## 📁 Architecture

```
src/inference/
├── __init__.py                    # Package initialization
├── config.py                      # Configuration loader (YAML + env vars)
├── config.yaml                    # Default configuration
├── schema.py                      # Spark schemas for IEEE-CIS transactions
├── model_loader.py                # XGBoost model + pipeline loader
├── feature_pipeline_spark.py      # Feature engineering (88 features)
├── velocity_service.py            # Redis-based velocity tracking
├── ato_service.py                 # Redis-based ATO detection
├── decision_engine.py             # Hybrid adaptive threshold + final decision
├── main_inference.py              # Spark Structured Streaming entry point
├── logging_utils.py               # Standardized logging
└── utils/
    ├── __init__.py
    └── redis_client.py            # Shared Redis connection pool
```

## 🚀 Quick Start

### 1. Run the Inference Service

```bash
# Make sure Redis and Kafka are running
# Run the Spark Structured Streaming job
spark-submit \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0 \
    src/inference/main_inference.py
```

### 2. Environment Variables

```bash
export KAFKA_BROKERS="localhost:9092"
export REDIS_HOST="localhost"
export MODEL_BUNDLE_PATH="/app/models/fraud_detection_model.pkl"
export FEATURE_PIPELINE_PATH="/app/models/feature_pipeline.pkl"
export LOG_LEVEL="INFO"
```

## 📊 How It Works

### Pipeline Flow

```
Kafka (transactions)
    ↓
Parse JSON → Validate Schema
    ↓
Feature Engineering (88 features)
    ↓
ML Model Prediction (XGBoost)
    ↓
Velocity Analysis (Redis)
    ↓
ATO Detection (Redis)
    ↓
Decision Engine (Adaptive Threshold)
    ↓
Route to Kafka Topics:
├── fraud_predictions (BLOCK/HOLD/REVIEW)
└── legit_predictions (APPROVE)
```

### Decision Logic

Uses **base threshold = 0.0564** (F1-optimal from training), **NOT 0.5**.

**Adaptive Threshold:**
```python
τ_velocity = τ_base - velocity_risk * 0.15
τ_amount   = τ_base - amount_risk * 0.10
τ_ato      = τ_base - ato_risk * 0.25

# Weighted combination based on dominant risk
τ_hybrid = w0*τ_base + w1*τ_velocity + w2*τ_amount + w3*τ_ato
```

**Final Decision:**
- `prob >= 0.20` → **BLOCK** (HIGH risk)
- `prob >= 0.10` → **HOLD** (HIGH risk)
- `prob >= τ_hybrid OR rules_triggered OR ato_detected` → **REVIEW** (MEDIUM risk)
- `else` → **APPROVE** (LOW risk)

## 🔧 Configuration

Edit `config.yaml` or use environment variables:

### Key Thresholds

```yaml
model:
  base_threshold: 0.0564      # F1-optimal from training
  review_threshold: 0.10      # REVIEW cutoff
  hold_threshold: 0.20        # HOLD cutoff
  block_threshold: 0.20       # BLOCK cutoff
```

### Velocity Detection

```yaml
velocity:
  high_1h_count: 5           # ≥5 txns in 1h → velocity_risk 0.8
  high_6h_count: 15          # ≥15 txns in 6h → velocity_risk 0.6
  amount_spike_5x: 5.0       # 5x avg amount → amount_risk 0.9
```

### ATO Detection

```yaml
ato:
  geo_anomaly_distance: 1000.0     # >1000km → ato_risk +0.30
  ato_detection_threshold: 0.6     # ato_risk ≥0.6 → ATO detected
```

## 📦 Output Format

### Fraud Predictions (Kafka topic: `fraud_predictions`)

```json
{
  "transaction_id": "TXN_123",
  "fraud_probability": 0.1875,
  "decision": "HOLD",
  "risk_level": "HIGH",
  "risk_factors": ["high_velocity_risk", "night_transaction"],
  "ato_risk": 0.7,
  "velocity_risk": 0.85,
  "amount_risk": 0.4,
  "timestamp": "2025-11-22T12:34:56Z"
}
```

### Legit Predictions (Kafka topic: `legit_predictions`)

```json
{
  "transaction_id": "TXN_124",
  "fraud_probability": 0.0245,
  "decision": "APPROVE",
  "risk_level": "LOW",
  "risk_factors": [],
  "ato_risk": 0.0,
  "velocity_risk": 0.1,
  "amount_risk": 0.0,
  "timestamp": "2025-11-22T12:35:10Z"
}
```

## 🧪 Testing

### Test with Single Transaction

```python
from config import Config
from model_loader import ModelLoader
from decision_engine import DecisionEngine

config = Config.load()

# Load model
loader = ModelLoader(config)
loader.load()

# Test transaction
transaction = {
    'TransactionAmt': 150.0,
    'card1': 12345,
    'addr1': 315,
    'P_emaildomain': 'gmail.com',
    # ... other fields
}

# Make prediction
# (See main_inference.py for full pipeline)
```

## 🔍 Monitoring

### Key Metrics to Track

1. **Throughput**: Transactions/second
2. **Latency**: End-to-end processing time
3. **Decision Distribution**: APPROVE / REVIEW / HOLD / BLOCK rates
4. **Model Performance**: Fraud detection rate vs false positive rate
5. **Redis Health**: Connection pool status, latency
6. **Kafka Lag**: Consumer lag on input topic

### Logs

All components use structured logging:

```
2025-11-22 12:34:56 - decision_engine - INFO - Decision: HOLD, RiskLevel: HIGH, Prob: 0.1875, Adjusted: 0.2375, Threshold: 0.0450, Factors: 3
```

## 🐛 Troubleshooting

### Issue: Model predictions are all 0.5

**Cause**: Feature pipeline mismatch between training and inference.

**Solution**: Ensure `feature_pipeline.pkl` is from the same training run as `fraud_detection_model.pkl`.

### Issue: Redis connection errors

**Cause**: Redis not running or wrong host/port.

**Solution**:
```bash
# Check Redis is running
redis-cli ping  # Should return PONG

# Update config
export REDIS_HOST="your-redis-host"
export REDIS_PORT=6379
```

### Issue: Kafka offset errors

**Cause**: Checkpoint corruption or topic deleted.

**Solution**:
```bash
# Delete checkpoints and restart
rm -rf /tmp/spark-checkpoints/fraud-detection/*
```

## 📚 Dependencies

```
pyspark>=3.3.0
redis>=4.5.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
joblib>=1.2.0
pyyaml>=6.0
```

## 🔐 Security Notes

- **Model artifacts**: Ensure `/app/models/` is read-only in production
- **Redis**: Use password authentication in production
- **Kafka**: Enable SSL/SASL authentication
- **Logging**: Avoid logging sensitive transaction data

## 📈 Performance Tuning

### Spark Configuration

```python
spark = SparkSession.builder \
    .config("spark.sql.shuffle.partitions", 200) \
    .config("spark.default.parallelism", 200) \
    .config("spark.streaming.kafka.maxOffsetsPerTrigger", 1000) \
    .getOrCreate()
```

### Redis Connection Pool

```python
redis:
  max_connections: 50  # Increase for high throughput
  socket_timeout: 5    # Reduce for low latency
```

## 📝 License

See main project LICENSE file.

## 👥 Contact

For issues or questions, please open a GitHub issue.
