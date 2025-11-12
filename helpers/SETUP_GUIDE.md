# Streaming Data-Driven Fraud Detection for Transactions using ML

## MLOps-Ready Real-time Fraud Detection System

A comprehensive end-to-end fraud detection system built with **Kafka + Spark Structured Streaming + Airflow + MLflow + MinIO** for production-grade ML operations.

---

## 🎯 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                          │
├─────────────────────────────────────────────────────────────────────┤
│  Producer (Synthetic) ──┐                                           │
│  REST API (Postman)  ───┼──► Kafka Topic: transactions             │
│  IEEE-CIS Batch      ───┘                                           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     REAL-TIME INFERENCE LAYER                        │
├─────────────────────────────────────────────────────────────────────┤
│  Spark Structured Streaming                                          │
│    ├─ Load Model + Feature Pipeline                                 │
│    ├─ Feature Engineering                                            │
│    ├─ Fraud Prediction (Probability + Risk Level)                   │
│    └─ Decision: BLOCK / APPROVE                                     │
└─────────────────────────────────────────────────────────────────────┘
                    │                          │
                    ▼                          ▼
┌──────────────────────────────┐   ┌──────────────────────────────┐
│  Kafka: fraud_predictions    │   │  Kafka: legit_predictions    │
│  (BLOCK decisions)           │   │  (APPROVE decisions)         │
└──────────────────────────────┘   └──────────────────────────────┘
         │                                    │
         ▼                                    ▼
┌────────────────────┐          ┌──────────────────────────────────┐
│ Notification       │          │  Dashboard (Streamlit)           │
│ Service (Email)    │          │  Real-time Monitoring            │
└────────────────────┘          └──────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      TRAINING & MLOPS LAYER                          │
├─────────────────────────────────────────────────────────────────────┤
│  Airflow DAG (ieee_cis_training_dag)                                │
│    ├─ Load IEEE-CIS Dataset                                         │
│    ├─ Feature Engineering                                            │
│    ├─ Train XGBoost/LightGBM/CatBoost                              │
│    ├─ Calibrate Probabilities                                       │
│    ├─ Log to MLflow (Metrics + Model Registry)                     │
│    └─ Save Artifacts (Model + Feature Pipeline)                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Dataset Setup](#dataset-setup)
6. [Running the System](#running-the-system)
7. [API Usage](#api-usage)
8. [Dashboard](#dashboard)
9. [Training Models](#training-models)
10. [Architecture Details](#architecture-details)
11. [Troubleshooting](#troubleshooting)

---

## ✨ Features

### Production-Ready ML Pipeline
- ✅ **Real-time Inference**: Spark Structured Streaming with <100ms latency
- ✅ **Feature Pipeline**: Train/serve consistency with serialized transformers
- ✅ **Risk Stratification**: HIGH/MEDIUM/LOW risk levels with configurable thresholds
- ✅ **Dual-Topic Output**: Separate streams for fraud and legitimate transactions
- ✅ **Model Calibration**: Calibrated probabilities for reliable confidence scores

### MLOps & Experiment Tracking
- ✅ **MLflow Integration**: Full experiment tracking and model registry
- ✅ **MinIO Storage**: S3-compatible artifact storage
- ✅ **Airflow Orchestration**: Automated training pipelines with scheduling
- ✅ **Model Versioning**: Track multiple model versions with metadata

### API & Integration
- ✅ **REST API**: FastAPI endpoint for transaction submission (Postman-ready)
- ✅ **Synchronous Scoring**: Optional request-reply pattern for immediate decisions
- ✅ **Batch Training**: IEEE-CIS dataset support with chronological splits

### Monitoring & Alerting
- ✅ **Real-time Dashboard**: Streamlit dashboard with live transaction feed
- ✅ **Email Alerts**: Automated fraud notifications via MailDev
- ✅ **Visual Analytics**: Risk distribution, volume charts, and statistics

---

## 🛠️ Prerequisites

- **Docker** & **Docker Compose** (v2.0+)
- **Minimum Resources**:
  - 8 GB RAM (16 GB recommended)
  - 4 CPU cores (8 recommended)
  - 20 GB disk space
- **Optional**: CUDA-capable GPU for accelerated training

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/fraud-detection-mlops.git
cd fraud-detection-mlops/src
```

### 2. Create Environment File

```bash
cp .env.example .env
```

Edit `.env` with your Kafka credentials:

```env
# Kafka Configuration (Confluent Cloud)
KAFKA_BOOTSTRAP_SERVERS=pkc-921jm.us-east-2.aws.confluent.cloud:9092
KAFKA_USERNAME=TUISIFY5HCFLGXIH
KAFKA_PASSWORD=your_kafka_password_here

KAFKA_TOPIC=transactions
KAFKA_FRAUD_TOPIC=fraud_predictions

# MinIO (S3-compatible storage)
AWS_ACCESS_KEY_ID=minio
AWS_SECRET_ACCESS_KEY=minio123
MINIO_USERNAME=minio
MINIO_PASSWORD=minio123

# MLflow
MLFLOW_TRACKING_URI=http://mlflow-server:5500

# SMTP (MailDev)
SMTP_HOST=maildev
SMTP_PORT=1025
SMTP_FROM=fraud.alerts@bank.com

# Airflow
AIRFLOW_UID=50000
```

### 3. Build and Start Services

```bash
# Build all services
docker-compose build

# Start core services (Airflow, MLflow, MinIO, MailDev)
docker-compose up -d postgres redis minio mc mlflow-server maildev

# Wait for MLflow to be healthy
docker-compose up -d airflow-init
docker-compose up -d airflow-webserver airflow-scheduler airflow-worker

# Start fraud detection pipeline
docker-compose up -d producer-api inference notification dashboard
```

---

## ⚙️ Configuration

### config.yaml Overview

```yaml
# Kafka Topics
kafka:
  topic: "transactions"
  output_topic: "fraud_predictions"
  legit_topic: "legit_predictions"

# IEEE-CIS Dataset
data:
  ieee_cis:
    train_transaction_path: "/app/data/ieee_cis/train_transaction.csv"
    train_identity_path: "/app/data/ieee_cis/train_identity.csv"

# Training
training:
  use_external_model: true  # Use model from notebook
  model_path: "/app/models/fraud_detection_model.pkl"
  feature_pipeline_path: "/app/models/feature_pipeline.pkl"

# Inference
inference:
  threshold: 0.6  # Fraud probability threshold
  risk_bands:
    high: 0.9     # >= 0.9 = HIGH
    medium: 0.6   # >= 0.6 = MEDIUM

# Dashboard
dashboard:
  port: 8501
  refresh_interval_seconds: 2
```

---

## 📊 Dataset Setup

### Option A: Use Exported Model from Notebook

If you've already trained a model in the Jupyter notebook ([fyp.ipynb](notebook/fyp.IPYNB)):

1. **Export model artifacts**:

```python
# In your notebook (after training)
import joblib

# Save calibrated model
joblib.dump(calibrated_model, 'fraud_detection_model.pkl')

# Save feature pipeline (optional but recommended)
joblib.dump(feature_pipeline, 'feature_pipeline.pkl')
```

2. **Copy to models directory**:

```bash
cp fraud_detection_model.pkl src/models/
cp feature_pipeline.pkl src/models/
```

3. **Set config flag**:

```yaml
# config.yaml
training:
  use_external_model: true
```

### Option B: Train on IEEE-CIS Dataset

1. **Download IEEE-CIS Fraud Detection dataset** from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data)

2. **Place files in data directory**:

```bash
mkdir -p src/data/ieee_cis
cp train_transaction.csv src/data/ieee_cis/
cp train_identity.csv src/data/ieee_cis/
```

3. **Update config**:

```yaml
# config.yaml
training:
  use_external_model: false
```

4. **Trigger Airflow DAG**:

```bash
# Access Airflow UI: http://localhost:8080
# Username: airflow, Password: airflow
# Trigger DAG: ieee_cis_fraud_detection_training
```

---

## 🚀 Running the System

### Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| **REST API** | http://localhost:8000 | N/A |
| **Dashboard** | http://localhost:8501 | N/A |
| **Airflow** | http://localhost:8080 | airflow / airflow |
| **MLflow** | http://localhost:5500 | N/A |
| **MinIO** | http://localhost:9001 | minio / minio123 |
| **MailDev** | http://localhost:1080 | N/A |

### Quick Start

```bash
# 1. Check all services are running
docker-compose ps

# 2. Submit test transaction via API
curl -X POST http://localhost:8000/api/v1/transactions/submit \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 299.99,
    "ProductCD": "W",
    "card1": 12345,
    "P_emaildomain": "gmail.com"
  }'

# 3. View dashboard
open http://localhost:8501

# 4. Check email alerts
open http://localhost:1080
```

---

## 📡 API Usage

### REST Endpoints

#### 1. Health Check

```bash
curl http://localhost:8000/health
```

**Response:**

```json
{
  "status": "healthy",
  "service": "fraud-detection-transaction-api",
  "timestamp": "2025-10-19T07:45:00Z",
  "kafka_topic": "transactions"
}
```

#### 2. Submit Transaction (Async)

```bash
curl -X POST http://localhost:8000/api/v1/transactions/submit \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": "T123456789",
    "TransactionDT": 15000000,
    "TransactionAmt": 129.99,
    "ProductCD": "W",
    "card1": 12345,
    "card2": 150,
    "addr1": 204,
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com",
    "timestamp": "2025-10-19T07:45:00Z"
  }'
```

**Response:**

```json
{
  "status": "accepted",
  "transaction_id": "T123456789",
  "message": "Transaction submitted successfully for fraud detection",
  "timestamp": "2025-10-19T07:45:01Z"
}
```

#### 3. Synchronous Scoring (Optional)

```bash
curl -X POST http://localhost:8000/api/v1/transactions/score-sync \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 5000.00,
    "ProductCD": "W",
    "card1": 99999,
    "P_emaildomain": "mailinator.com"
  }'
```

**Response (Fraud):**

```json
{
  "status": "blocked",
  "transaction_id": "TXN_ABC123",
  "decision": "BLOCK",
  "probability": 0.9142,
  "risk_level": "HIGH",
  "latency_ms": 87.32
}
```

**Response (Legit):**

```json
{
  "status": "approved",
  "transaction_id": "TXN_XYZ789",
  "decision": "APPROVE",
  "probability": 0.1234,
  "risk_level": "LOW",
  "latency_ms": 45.67
}
```

### Postman Collection

Import the provided [Postman collection](postman/fraud_detection_api.json):

1. Open Postman
2. Import → File → Select `postman/fraud_detection_api.json`
3. Set environment variable: `API_BASE_URL = http://localhost:8000`
4. Run requests from collection

---

## 📊 Dashboard

### Streamlit Dashboard Features

Access at: **http://localhost:8501**

**Real-time Metrics:**
- Total transactions processed
- Fraud detection count
- Legitimate transaction count
- Real-time fraud rate

**Visualizations:**
- Transaction volume chart (fraud vs legit)
- Risk level distribution (pie chart)
- Recent transactions table (color-coded)
- Fraud alerts with transaction details

**Configuration:**
- Adjustable refresh rate (1-10 seconds)
- Max display rows (10-200 transactions)

---

## 🎓 Training Models

### Airflow DAG: ieee_cis_fraud_detection_training

**Schedule**: Daily at 3:00 AM (or on-demand)

**Pipeline Steps:**

1. **Environment Validation**
   - Check config.yaml exists
   - Verify IEEE-CIS dataset availability
   - Ensure model output directory is writable

2. **Training Execution**
   - Load and merge `train_transaction.csv` + `train_identity.csv`
   - Feature engineering (lean production set)
   - Chronological split (80% train, 20% validation)
   - Train XGBoost/LightGBM/CatBoost (best AUC-PR selected)
   - Calibrate probabilities (sigmoid)
   - Find F1-optimal threshold

3. **MLflow Logging**
   - Log metrics: AUC-PR, AUC-ROC, precision, recall, F1
   - Log parameters: model config, threshold
   - Log artifacts: model, feature pipeline, confusion matrix
   - Register model in Model Registry

4. **Artifact Saving**
   - Save model: `/app/models/fraud_detection_model.pkl`
   - Save feature pipeline: `/app/models/feature_pipeline.pkl`

**Manual Trigger:**

```bash
# Via Airflow UI
# Navigate to http://localhost:8080 → DAGs → ieee_cis_fraud_detection_training → Trigger

# Via CLI
docker exec -it <airflow-scheduler-container> \
  airflow dags trigger ieee_cis_fraud_detection_training
```

---

## 🏗️ Architecture Details

### Kafka Topics

| Topic | Producer | Consumer | Schema |
|-------|----------|----------|--------|
| `transactions` | Producer API / Synthetic | Inference | IEEE-CIS compatible |
| `fraud_predictions` | Inference | Notification + Dashboard | Fraud decision payload |
| `legit_predictions` | Inference | Dashboard | Legit decision payload |

### Output Payload Schema

**fraud_predictions / legit_predictions**:

```json
{
  "transaction_id": "T123456789",
  "probability": 0.9142,
  "risk_level": "HIGH|MEDIUM|LOW",
  "decision": "BLOCK|APPROVE",
  "timestamp": "2025-10-19T07:45:01Z",
  "amount": 299.99,
  "user_id": 1234,
  "merchant": "TechStore",
  "location": "US"
}
```

### Feature Set (Lean for Latency)

**Total Features: ~16** (optimized for <50ms inference)

- **Amount Features**: TransactionAmt, log_amt, sqrt_amt
- **Card Features**: card1, card2
- **Address**: addr1
- **Email Features**: P_emaildomain, R_emaildomain, email_match, email_is_risky, email_is_generic
- **Temporal**: transaction_hour, transaction_day_of_week, is_weekend, is_night
- **Product**: ProductCD

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Kafka Connection Errors

**Symptom**: `Failed to connect to Kafka broker`

**Solution**:
```bash
# Verify Kafka credentials in .env
# Test connection
docker-compose exec producer-api python -c "from confluent_kafka import Producer; print('OK')"
```

#### 2. Model Not Found

**Symptom**: `Model not found at /app/models/fraud_detection_model.pkl`

**Solution**:
```bash
# Option A: Export from notebook and copy to src/models/
# Option B: Train via Airflow DAG
# Option C: Set use_external_model: false in config.yaml
```

#### 3. Airflow DAG Fails

**Symptom**: Training DAG shows "Failed" status

**Solution**:
```bash
# Check logs
docker-compose logs airflow-scheduler

# Verify dataset exists
ls -lh src/data/ieee_cis/

# Check Airflow task logs in UI
# Navigate to failed task → Log
```

#### 4. Dashboard Not Loading

**Symptom**: Streamlit dashboard shows error

**Solution**:
```bash
# Restart dashboard
docker-compose restart dashboard

# Check logs
docker-compose logs dashboard

# Verify Kafka topics exist
# kafka-topics --bootstrap-server ... --list
```

---

## 📚 Additional Resources

### Documentation
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Confluent Kafka Python](https://docs.confluent.io/kafka-clients/python/current/overview.html)
- [Apache Airflow](https://airflow.apache.org/docs/)
- [Spark Structured Streaming](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)

### Model Training
- Notebook: `notebook/fyp.ipynb` (full training pipeline with VAE ensemble)
- Training module: `src/dags/ieee_cis_training.py`

### API Documentation
- Interactive API docs: http://localhost:8000/docs
- Redoc: http://localhost:8000/redoc

---

## 📧 Support

For issues and questions:
- Email: chirantharavishka@gmail.com
- GitHub Issues: [Create an issue](https://github.com/yourusername/fraud-detection-mlops/issues)

---

## 📄 License

This project is licensed under the MIT License.

---

## 🎉 Acknowledgments

- IEEE-CIS Fraud Detection Dataset (Kaggle)
- Confluent Cloud Kafka
- MLflow Community
- Apache Airflow Community

---

**Built with ❤️ for production-ready MLOps**
