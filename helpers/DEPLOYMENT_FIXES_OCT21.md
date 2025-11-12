# Deployment Fixes - October 21, 2025

## Issues Fixed

### 1. ✅ Inference Service - AttributeError: `get_hybrid_threshold()`
**Symptom**: 
```
ERROR:__main__:Prediction error: 'AdaptiveThresholdSystem' object has no attribute 'get_hybrid_threshold'
```

**Root Cause**: 
- `src/inference/ieee_cis_training.py` had outdated `AdaptiveThresholdSystem` class
- Missing the `get_hybrid_threshold()` method added in training DAG
- Caused all transactions to be incorrectly classified as FRAUD

**Fix Applied**: 
- Added complete `get_hybrid_threshold()` method to inference version
- Supports 4 threshold strategies: weighted, dynamic, max, min
- Implements industry best practice: τ_hybrid = w₁·τ_A + w₂·τ_V + w₃·τ_Amount

---

### 2. ✅ MLflow Server - 403 Forbidden (Invalid Host Header)
**Symptom**:
```
WARNING mlflow.server.fastapi_security: Rejected request with invalid Host header: mlflow-server:5500
INFO: 172.18.0.15:42480 - "POST /api/2.0/mlflow/experiments/search HTTP/1.1" 403 Forbidden
```

**Root Cause**:
- MLflow 3.5+ introduced security middleware
- Rejects requests without explicitly allowed Host headers
- Blocked: other containers, browser access, training DAG logging

**Fix Applied**:
- Added `MLFLOW_SERVER_ALLOWED_HOSTS` environment variable
  - Allows: `mlflow-server`, `localhost`, `127.0.0.1`, `64.23.228.115`, `172.18.*`
- Added `MLFLOW_SERVER_CORS_ALLOWED_ORIGINS` for cross-origin requests
  - Allows: `http://localhost:*`, `http://mlflow-server:*`, `http://64.23.228.115:*`

---

## Deployment Steps on VM (64.23.228.115)

### Step 1: Pull Latest Code
```bash
cd /root/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML
git pull origin main
```

**Expected Output**:
```
remote: Counting objects: 5, done.
...
Updating 1a2b3c4..3a009e1
Fast-forward
 src/inference/ieee_cis_training.py | 73 +++++++++++++++++++++++++++++++++++++
 src/docker-compose.yml             | 10 +++--
 rebuild_and_restart_inference.sh   | 22 +++++++++++
 3 files changed, 102 insertions(+), 3 deletions(-)
```

### Step 2: Rebuild Inference Service
```bash
cd src

# Stop current inference
docker compose stop inference

# Rebuild with no cache (ensures fresh code)
docker compose build --no-cache inference

# Start inference
docker compose up -d inference
```

**Expected Build Time**: ~2-3 minutes

### Step 3: Restart MLflow Server
```bash
# Restart to apply new security middleware settings
docker compose restart mlflow-server

# Wait 10 seconds for startup
sleep 10
```

### Step 4: Verify Inference is Working
```bash
# Check logs (should see NO AttributeError)
docker compose logs --tail=50 inference
```

**Expected Output (SUCCESS)**:
```
fraud-inference  | INFO:__main__:✓ MLflow connectivity verified: http://mlflow-server:5500
fraud-inference  | INFO:__main__:Model loaded successfully
fraud-inference  | INFO:__main__:Streaming queries started. Writing to fraud_predictions and legit_predictions topics.
fraud-inference  | INFO:__main__:Batch: 1, Rows: 1
fraud-inference  | INFO:__main__:Prediction successful - Fraud Probability: 0.0523
```

**Expected Output (BEFORE FIX - should NOT see this anymore)**:
```
fraud-inference  | ERROR:__main__:Prediction error: 'AdaptiveThresholdSystem' object has no attribute 'get_hybrid_threshold'
```

### Step 5: Verify MLflow is Accessible
```bash
# Test from container network
docker compose exec inference curl -s http://mlflow-server:5500/health

# Test from host
curl -s http://localhost:5500/health

# Test from external IP
curl -s http://64.23.228.115:5500/health
```

**Expected Output (all 3 should return)**:
```json
{"status": "healthy"}
```

**Before Fix (returned 403)**:
```html
<html><body><h1>403 Forbidden</h1>Invalid Host header</body></html>
```

---

## Testing the Fixes

### Test 1: Submit Legitimate Transaction
```bash
cat > /tmp/legit_test.json << 'EOF'
{
  "TransactionID": "LEGIT_TEST_001",
  "TransactionDT": 86400,
  "TransactionAmt": 45.99,
  "ProductCD": "W",
  "card1": 13926,
  "card2": 150,
  "card3": 150,
  "card4": "visa",
  "card5": 226,
  "card6": "credit",
  "addr1": 315,
  "addr2": 87,
  "P_emaildomain": "gmail.com",
  "R_emaildomain": "yahoo.com",
  "timestamp": "2024-10-21T14:30:00Z"
}
EOF

curl -X POST http://localhost:8000/submit_transaction \
  -H "Content-Type: application/json" \
  -d @/tmp/legit_test.json
```

**Expected Result**:
- Decision: `APPROVE` or `MONITOR` (NOT `REVIEW`)
- Risk Level: `LOW` or `MEDIUM` (NOT `HIGH`)
- Fraud Probability: < 0.3
- Routed to: `legit_predictions` topic

### Test 2: Submit Fraudulent Transaction
```bash
cat > /tmp/fraud_test.json << 'EOF'
{
  "TransactionID": "FRAUD_TEST_007",
  "TransactionDT": 86400,
  "TransactionAmt": 2500.00,
  "ProductCD": "W",
  "card1": 99999,
  "card2": 999,
  "card3": 999,
  "card4": "visa",
  "card5": 999,
  "card6": "credit",
  "addr1": 999,
  "addr2": 999,
  "P_emaildomain": "suspicious.xyz",
  "R_emaildomain": "temporary.tk",
  "timestamp": "2024-10-21T23:45:00Z"
}
EOF

curl -X POST http://localhost:8000/submit_transaction \
  -H "Content-Type: application/json" \
  -d @/tmp/fraud_test.json
```

**Expected Result**:
- Decision: `REVIEW` or `DECLINE`
- Risk Level: `HIGH`
- Fraud Probability: > 0.5
- Routed to: `fraud_predictions` topic

### Test 3: Access MLflow UI
```bash
# Open in browser
http://64.23.228.115:5500
```

**Expected**: MLflow UI loads (NOT 403 Forbidden)

### Test 4: Retrain Model with MLflow Logging
```bash
# Trigger training DAG via Airflow UI
# http://64.23.228.115:8080 (airflow/airflow)
# Or via CLI:
docker exec src-airflow-scheduler-1 airflow dags trigger ieee_cis_training_dag

# Monitor logs
docker compose logs -f airflow-worker
```

**Expected Output (END of training)**:
```
[2025-10-21, XX:XX:XX UTC] {ieee_cis_training.py:1XXX} INFO - Successfully created and logged all visualizations
[2025-10-21, XX:XX:XX UTC] {ieee_cis_training.py:1XXX} INFO - Model registered as: fraud_detection_model
[2025-10-21, XX:XX:XX UTC] {ieee_cis_training.py:1XXX} INFO - Experiment logged to MLflow: ieee_cis_fraud_detection
```

**Before Fix (WARNING at end)**:
```
WARNING: MLflow logging failed (non-critical): unhashable type: 'list'
```

---

## Verification Checklist

After deployment, verify:

- [ ] Inference service running: `docker compose ps inference` shows `Up`
- [ ] No AttributeError in logs: `docker compose logs inference | grep -i attributeerror` returns empty
- [ ] MLflow accessible from containers: `docker compose exec inference curl http://mlflow-server:5500/health`
- [ ] MLflow accessible from browser: Open `http://64.23.228.115:5500` in browser
- [ ] Legitimate transactions classified correctly (not all as fraud)
- [ ] Dashboard shows mixed results (fraud + legit): `http://64.23.228.115:8501`
- [ ] MLflow experiment logging works (no 403 errors during training)

---

## Rollback Plan (if needed)

If issues occur after deployment:

```bash
cd /root/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/src

# Rollback to previous commit
git reset --hard HEAD~1

# Rebuild and restart
docker compose down
docker compose build --no-cache
docker compose up -d

# Monitor
docker compose logs -f
```

---

## Technical Details

### Files Modified

1. **src/inference/ieee_cis_training.py** (lines 259-331)
   - Added `get_hybrid_threshold()` method to `AdaptiveThresholdSystem` class
   - Implements 4 threshold strategies:
     - `weighted`: Fixed weighted average (default: w1=0.6, w2=0.3, w3=0.1)
     - `dynamic`: Adaptive weights based on risk confidence
     - `max`: Most conservative (highest threshold)
     - `min`: Most aggressive (lowest threshold)
   
2. **src/docker-compose.yml** (mlflow-server service)
   - Added environment variables:
     ```yaml
     - MLFLOW_SERVER_ALLOWED_HOSTS=mlflow-server,localhost,127.0.0.1,64.23.228.115,172.18.*
     - MLFLOW_SERVER_CORS_ALLOWED_ORIGINS=http://localhost:*,http://127.0.0.1:*,http://mlflow-server:*,http://64.23.228.115:*
     ```
   - Updated command to multi-line format for readability

3. **rebuild_and_restart_inference.sh** (new helper script)
   - Automates inference rebuild process
   - Includes --no-cache flag to ensure fresh build

### Hybrid Threshold Formula

```
τ_hybrid = w₁·τ_A + w₂·τ_V + w₃·τ_Amount

Where:
- τ_A = Adaptive threshold (F1-optimal, statistical)
- τ_V = Velocity-adjusted threshold (high velocity → lower threshold)
- τ_Amount = Amount-adjusted threshold (high amount → lower threshold)
- w₁, w₂, w₃ = Weights (default: 0.6, 0.3, 0.1)
```

**Example Calculations**:
```python
# Low risk transaction (small amount, normal velocity)
velocity_risk = 0.1, amount_risk = 0.05
τ_hybrid ≈ 0.42 (near F1-optimal ~0.45)
→ Requires high confidence to flag as fraud

# High velocity transaction (rapid succession)
velocity_risk = 0.9, amount_risk = 0.2
τ_hybrid ≈ 0.28 (much lower threshold)
→ More sensitive to fraud indicators

# High amount transaction (unusual spending)
velocity_risk = 0.2, amount_risk = 0.85
τ_hybrid ≈ 0.35 (moderately lower threshold)
→ Extra scrutiny on large transactions
```

---

## Support

If issues persist after deployment:

1. **Check logs**: `docker compose logs -f inference mlflow-server`
2. **Verify code version**: `git log -1 --oneline` should show `3a009e1 Fix inference AttributeError...`
3. **Check container build time**: `docker inspect fraud-inference | grep Created`
   - Should be AFTER the git pull timestamp
   - If older → rebuild with `--no-cache` flag

---

## Success Metrics

After successful deployment:

✅ **Inference Service**:
- 0% AttributeError rate
- Proper threshold calculations (per-transaction hybrid thresholds)
- Balanced fraud/legit predictions (~3-5% fraud rate, not 100%)

✅ **MLflow Server**:
- 0% 403 Forbidden errors
- Browser access working
- Training DAG logging experiments successfully
- UI shows all 6 visualizations for new runs

✅ **Overall System**:
- Dashboard shows realistic fraud rates
- API responses include proper risk scores
- Kafka topics have balanced traffic (not all fraud_predictions)

---

**Deployment Date**: October 21, 2025  
**Git Commit**: `3a009e1`  
**Tested On**: Ubuntu 22.04 VM (64.23.228.115), Docker Compose v2.x
