# Inference Service Setup - Issue Resolved

## Problem Identified
The inference service was **defined but not running** because:
1. ❌ No explicit `command` specified (container exits immediately)
2. ❌ No `restart: always` policy
3. ❌ Missing dependencies (redis, postgres, mlflow-server)
4. ❌ No MLflow environment variables

## Changes Made

### 1. Updated `docker-compose.yml` - Inference Service
Added critical configuration:
```yaml
inference:
  build: ./inference
  container_name: fraud-inference
  env_file: .env
  command: python main_enhanced.py          # ✅ Explicit command
  restart: always                            # ✅ Auto-restart policy
  depends_on:                                # ✅ Service dependencies
    - redis
    - postgres
    - mlflow-server
  environment:                               # ✅ MLflow connection
    - MLFLOW_TRACKING_URI=http://mlflow-server:5500
    - MLFLOW_S3_ENDPOINT_URL=http://minio:9000
  volumes:
    - ./models:/app/models
    - ./config.yaml:/app/config.yaml
    - ./.env:/app/.env
    - ./data:/app/data
  networks:
    - fraud-detection
```

### 2. Created Helper Script: `start_inference.sh`
Quick script to rebuild and start the inference service.

## Deployment Steps on VM

### Step 1: Pull Latest Changes
```bash
cd ~/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML
git pull origin main
```

### Step 2: Navigate to src Directory
```bash
cd src
```

### Step 3: Start Inference Service
**Option A - Using the helper script** (recommended):
```bash
cd ..
chmod +x start_inference.sh
./start_inference.sh
```

**Option B - Manual commands**:
```bash
# From src directory
docker compose rm -f inference
docker compose build inference
docker compose up -d inference
```

### Step 4: Verify It's Running
```bash
docker compose ps | grep inference
```

**Expected output**:
```
fraud-inference    inference    "python main_enhanced.py"    inference    X minutes ago    Up X minutes    fraud-detection
```

### Step 5: Monitor Logs
```bash
# Follow logs in real-time
docker compose logs -f inference

# View last 50 lines
docker compose logs inference | tail -50
```

## What the Inference Service Does

1. **Loads trained model** from `/app/models/` with hybrid threshold system
2. **Connects to Kafka/streaming source** to consume incoming transactions
3. **Performs real-time fraud detection**:
   - Calculates 75 features (11 base + 64 engineered)
   - Computes velocity features (transactions_last_hour, avg_amount_last_24h, etc.)
   - Applies **per-transaction hybrid thresholds** (τ = w₁·τ_A + w₂·τ_V + w₃·τ_Amount)
   - Makes fraud prediction
4. **Publishes results** to output topics/databases
5. **Logs to MLflow** (if configured)

## Expected Behavior

### On Startup:
```
INFO: Loading model from /app/models/fraud_detection_model_hybrid/
INFO: Adaptive threshold system loaded
INFO: Base threshold (F1-optimal): 0.42
INFO: Starting Spark Streaming...
INFO: Waiting for transactions...
```

### During Processing:
```
INFO: Transaction T123456 - Score: 0.85, Threshold: 0.30 (lowered), Prediction: FRAUD
INFO: Transaction T789012 - Score: 0.15, Threshold: 0.42, Prediction: LEGITIMATE
```

## Troubleshooting

### If service exits immediately:
```bash
# Check build logs
docker compose logs inference

# Check for errors in main_enhanced.py
docker compose run --rm inference python -c "import main_enhanced"
```

### If "model not found":
```bash
# Verify model exists
ls -lh models/

# Train a new model via Airflow
# Go to http://64.23.228.115:8080
# Trigger: ieee_cis_training_dag
```

### If Kafka connection fails:
Check your `config.yaml` for Kafka broker settings:
```bash
cat config.yaml | grep -A 5 kafka
```

## Testing with Inference Running

Once the service is up, you can test it:

### 1. Via Producer API (Recommended)
```bash
curl -X POST http://64.23.228.115:8000/submit_transaction \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": "TEST001",
    "TransactionAmt": 5000.00,
    "P_emaildomain": "suspicious.ru",
    "TransactionHour": 2
  }'
```

### 2. Via Postman
- Import: `postman/fraud_detection_api.json`
- URL: `http://64.23.228.115:8000/submit_transaction`
- Body: Use the test JSON from previous conversation

### 3. Check Results
```bash
# Watch inference logs
docker compose logs -f inference

# Check dashboard (if configured)
# http://64.23.228.115:8501

# Check notification emails
# http://64.23.228.115:1080  (MailDev)
```

## Architecture Overview

```
[Producer API:8000] 
       ↓ (publish transaction)
[Kafka/Redis Queue]
       ↓ (consume)
[Inference Service] ← uses model with hybrid thresholds
       ↓ (fraud detected?)
[Notification Service:1025] → Email Alerts
[Dashboard:8501] → Real-time Visualization
```

## Next Steps

1. ✅ **Pull code changes** from git
2. ✅ **Start inference service** using script or manual commands
3. ✅ **Verify it's running** with `docker compose ps`
4. ✅ **Monitor logs** to ensure it's processing
5. ✅ **Test with sample transaction** via Producer API
6. ✅ **View results** in dashboard and MLflow

## Key Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| **Inference** | (internal) | Real-time fraud detection |
| **Producer API** | http://64.23.228.115:8000 | Submit transactions |
| **Airflow** | http://64.23.228.115:8080 | Train models |
| **MLflow** | http://64.23.228.115:5500 | View experiments |
| **Dashboard** | http://64.23.228.115:8501 | Real-time monitoring |
| **MailDev** | http://64.23.228.115:1080 | Email alerts |

## Summary

The inference service is now **properly configured** to:
- ✅ Stay running continuously (`restart: always`)
- ✅ Execute the correct command (`python main_enhanced.py`)
- ✅ Connect to dependencies (redis, postgres, mlflow)
- ✅ Load models with hybrid threshold system
- ✅ Process transactions in real-time
- ✅ Apply per-transaction adaptive thresholds

**Action Required**: Run the commands on your VM to start the service!
