# Quick Start Guide - Fraud Detection MLOps System

## 🚀 Get Running in 5 Minutes

### Prerequisites Check

```bash
# Verify Docker is running
docker --version
docker-compose --version

# Check resources
docker info | grep "Total Memory"  # Should show >= 8GB
```

---

## Step 1: Clone and Configure (1 min)

```bash
cd d:\Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML\src

# Create .env file
cp .env.example .env

# Edit .env with your Kafka credentials (or use defaults for local testing)
```

---

## Step 2: Start Core Services (2 min)

```bash
# Build images
docker-compose build

# Start infrastructure
docker-compose up -d postgres redis minio mc mlflow-server maildev

# Initialize Airflow
docker-compose up -d airflow-init

# Start Airflow services
docker-compose up -d airflow-webserver airflow-scheduler airflow-worker
```

**Wait 30 seconds for services to initialize...**

---

## Step 3: Start Fraud Detection Pipeline (1 min)

```bash
# Start all fraud detection services
docker-compose up -d producer-api inference notification dashboard
```

---

## Step 4: Test the System (1 min)

### A. Check Service Health

```bash
# API Health
curl http://localhost:8000/health

# Dashboard
open http://localhost:8501

# Airflow
open http://localhost:8080  # Login: airflow/airflow
```

### B. Submit Test Transaction

```bash
curl -X POST http://localhost:8000/api/v1/transactions/submit \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 299.99,
    "ProductCD": "W",
    "card1": 12345,
    "P_emaildomain": "gmail.com"
  }'
```

**Expected Response:**

```json
{
  "status": "accepted",
  "transaction_id": "TXN_ABC123...",
  "message": "Transaction submitted successfully for fraud detection",
  "timestamp": "2025-10-19T..."
}
```

### C. Monitor Results

1. **Dashboard**: http://localhost:8501
   - Watch transaction appear in real-time table
   - Check risk level and decision

2. **Email Alerts** (if fraud): http://localhost:1080
   - View MailDev inbox for fraud notifications

---

## 📊 Service URLs - Quick Reference

| Service | URL | Purpose |
|---------|-----|---------|
| **REST API** | http://localhost:8000 | Submit transactions |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |
| **Dashboard** | http://localhost:8501 | Real-time monitoring |
| **Airflow** | http://localhost:8080 | Training pipeline orchestration |
| **MLflow** | http://localhost:5500 | Experiment tracking |
| **MinIO** | http://localhost:9001 | S3 artifact storage |
| **MailDev** | http://localhost:1080 | Email testing |

---

## 🎯 Test Scenarios

### Scenario 1: Legitimate Transaction (LOW risk)

```bash
curl -X POST http://localhost:8000/api/v1/transactions/submit \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 49.99,
    "ProductCD": "W",
    "card1": 12345,
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com"
  }'
```

**Expected Outcome:**
- ✅ Decision: APPROVE
- ✅ Risk Level: LOW
- ✅ Probability: < 0.3
- ✅ Appears in `legit_predictions` topic

---

### Scenario 2: Fraudulent Transaction (HIGH risk)

```bash
curl -X POST http://localhost:8000/api/v1/transactions/submit \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 5000.00,
    "ProductCD": "W",
    "card1": 99999,
    "card2": 999,
    "P_emaildomain": "mailinator.com",
    "R_emaildomain": "tempmail.com"
  }'
```

**Expected Outcome:**
- 🔴 Decision: BLOCK
- 🔴 Risk Level: HIGH
- 🔴 Probability: > 0.8
- 🔴 Email alert sent
- 🔴 Appears in `fraud_predictions` topic
- 🔴 Dashboard shows red banner alert

---

### Scenario 3: Synchronous Scoring (Immediate Decision)

```bash
curl -X POST http://localhost:8000/api/v1/transactions/score-sync \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 750.00,
    "ProductCD": "H",
    "card1": 54321,
    "P_emaildomain": "yahoo.com"
  }'
```

**Expected Response:**

```json
{
  "status": "approved",
  "transaction_id": "TXN_XYZ...",
  "decision": "APPROVE",
  "probability": 0.35,
  "risk_level": "LOW",
  "latency_ms": 87.5
}
```

---

## 🎓 Training Your Own Model

### Option A: Use Pre-trained Model from Notebook

1. **Train in Jupyter Notebook** ([notebook/fyp.ipynb](notebook/fyp.IPYNB))

2. **Export model**:

```python
import joblib

# After training
joblib.dump(calibrated_model, 'fraud_detection_model.pkl')
joblib.dump(feature_pipeline, 'feature_pipeline.pkl')
```

3. **Copy to project**:

```bash
cp fraud_detection_model.pkl src/models/
cp feature_pipeline.pkl src/models/
```

4. **Restart inference service**:

```bash
docker-compose restart inference
```

---

### Option B: Train via Airflow on IEEE-CIS Dataset

1. **Download IEEE-CIS dataset** from Kaggle

2. **Place files**:

```bash
mkdir -p src/data/ieee_cis
cp train_transaction.csv src/data/ieee_cis/
cp train_identity.csv src/data/ieee_cis/
```

3. **Update config**:

```yaml
# src/config.yaml
training:
  use_external_model: false
```

4. **Trigger Airflow DAG**:

- Open http://localhost:8080
- Login: `airflow` / `airflow`
- Find DAG: `ieee_cis_fraud_detection_training`
- Click "Trigger DAG" (play button)

5. **Monitor progress**:

- Watch DAG execution in Airflow UI
- Check MLflow for logged metrics: http://localhost:5500

---

## 🛠️ Troubleshooting

### Services Not Starting

```bash
# Check logs
docker-compose logs --tail=50 <service-name>

# Common fixes
docker-compose down
docker system prune -f
docker-compose up -d --build
```

### Kafka Connection Issues

```bash
# Verify .env has correct Kafka credentials
cat .env | grep KAFKA

# Test producer API connectivity
docker-compose logs producer-api
```

### Model Not Loading

```bash
# Check if model exists
ls -lh src/models/

# If missing, either:
# 1. Export from notebook (Option A above)
# 2. Train via Airflow (Option B above)
# 3. Use fallback synthetic model (set use_external_model: false)
```

---

## 📈 Monitoring & Observability

### Real-time Dashboard Metrics

- **Transaction Volume**: Fraud vs Legit (last 100 transactions)
- **Risk Distribution**: HIGH / MEDIUM / LOW breakdown
- **Fraud Rate**: Percentage of flagged transactions
- **Recent Alerts**: Latest fraud detections with details

### Email Alerts

- **MailDev Inbox**: http://localhost:1080
- All fraud detections trigger email
- Includes: Transaction ID, Amount, Risk Level, Probability

### MLflow Experiments

- **Tracking UI**: http://localhost:5500
- View all training runs
- Compare model metrics
- Download artifacts

---

## 🎉 Next Steps

1. **Customize Thresholds**: Edit `src/config.yaml` to adjust fraud detection sensitivity

2. **Add Custom Features**: Modify `src/dags/ieee_cis_training.py` to include additional features

3. **Scale Up**: Increase producer replicas in `docker-compose.yml`

4. **Production Deploy**: Configure proper Kafka cluster, PostgreSQL, and S3 storage

5. **Advanced Monitoring**: Integrate Prometheus + Grafana for production observability

---

## 📚 Full Documentation

For comprehensive details, see:
- **Setup Guide**: [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **API Documentation**: http://localhost:8000/docs (when running)
- **Postman Collection**: [postman/fraud_detection_api.json](postman/fraud_detection_api.json)

---

**Questions?** Email: chirantharavishka@gmail.com

**Happy Fraud Detecting! 🛡️**
