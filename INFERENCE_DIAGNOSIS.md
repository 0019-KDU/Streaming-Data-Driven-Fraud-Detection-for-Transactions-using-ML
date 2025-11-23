# Inference Service Data Flow Issue - Root Cause Analysis

**Date:** November 23, 2025  
**Issue:** Dashboard shows transaction but data appears incomplete/incorrect

---

## 🔍 Root Cause Analysis

### From Spark Logs:
```json
{
  "batchId": 27,
  "numInputRows": 1,           ← ✅ Received transaction
  "processedRowsPerSecond": 0.051,
  "durationMs": {
    "addBatch": 17820          ← ⚠️ Took 17.8 seconds (SLOW!)
  },
  "numOutputRows": 1           ← ✅ Wrote 1 output
}
```

**Observations:**
1. ✅ Spark IS receiving transactions (numInputRows: 1)
2. ✅ Spark IS processing them (numOutputRows: 1)
3. ⚠️ **Processing is VERY SLOW** (17.8 seconds per transaction)
4. ❓ **Where is the output going?** (fraud_predictions or legit_predictions?)

---

## 🐛 Potential Issues

### Issue #1: Feature Pipeline Performance
**Location:** `src/inference/main_inference.py` Line 103-106

```python
# Current code (SLOW!)
features_df = feature_pipeline.feature_pipeline.transform(
    pd.DataFrame([transaction])
)
```

**Problem:**
- Creates new pandas DataFrame for EACH transaction
- Sklearn pipeline overhead for single-row transform
- No batch processing optimization

**Expected:** < 100ms per transaction  
**Actual:** 17,800ms (17.8 seconds) ← **178x slower**!

**Fix:** Add logging + error handling to see what's failing:

```python
try:
    import time
    start = time.time()
    
    # Apply feature engineering
    features_df = feature_pipeline.feature_pipeline.transform(
        pd.DataFrame([transaction])
    )
    
    duration = time.time() - start
    if duration > 1.0:
        logger.warning(f"Slow feature transform: {duration:.2f}s for TX {transaction_id}")
    
except Exception as e:
    logger.error(f"Feature engineering failed for TX {transaction_id}: {e}")
    raise
```

### Issue #2: Output Topic Routing
**Location:** `src/inference/main_inference.py` Line 310-315

```python
# Route to fraud/legit topics based on decision
fraud_df = predictions_df.filter(
    col("decision").isin(["BLOCK", "HOLD", "REVIEW"])
)

legit_df = predictions_df.filter(
    col("decision") == "APPROVE"
)
```

**Your Transaction:** Decision=APPROVE → Goes to **legit_predictions** topic

**Dashboard Consumption:**
- ✅ Reads from both "fraud_predictions" AND "legit_predictions"
- ✅ Should display APPROVE transactions

**Hypothesis:** Dashboard IS showing the transaction correctly, but the **data itself** looks wrong.

### Issue #3: Dashboard Data Display
**From your screenshot:**
```
Transaction ID: 2987245
Decision: APPROVE  ✅ Correct
Risk Level: LOW    ✅ Correct
Probability: 0.019 ✅ Correct (1.9% fraud risk)
Risk Factors: []   ✅ Correct after our fix
```

**This is EXACTLY what we expect!** The system is working correctly.

---

## ✅ Expected vs Actual Behavior

### Expected for Your Transaction:
- **TransactionAmt:** $37.10
- **Fraud Probability:** ~0.019 (1.9%) - LOW risk
- **Decision:** APPROVE
- **Risk Level:** LOW
- **Risk Factors:** [] (empty - no red flags)

### What You're Seeing:
```
0  2025-11-23T18:04:25  2987245  APPROVE  LOW  0.019  $37.10  13413.0 (visa)  hotmail.com  C  —  []
```

**Analysis:** This is **100% CORRECT**! ✅

---

## ❓ What's the "Issue"?

You mentioned:
> "postman request data not pass through the model correctly"
> "it's not show correctly in the dashboard"

**Questions to clarify:**

1. **What did you EXPECT to see?**
   - Higher fraud probability?
   - Different decision (BLOCK/REVIEW)?
   - Different risk factors?

2. **Is the transaction actually fraudulent?**
   - If YES → Model might be making a FALSE NEGATIVE (missing fraud)
   - If NO → Model is CORRECT (approving legit transaction)

3. **What looks "incorrect" about the dashboard?**
   - Missing fields?
   - Wrong values?
   - Formatting issues?

---

## 🧪 Diagnostic Script

Save this as `test_inference_detailed.py` and run locally:

```python
"""
Test Inference Pipeline Locally
Simulates exactly what Spark does on the VM
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
from inference.config import Config
from inference.feature_pipeline_spark import create_feature_pipeline
from inference.model_loader import ModelLoader

# Your transaction
transaction = {
    "TransactionID": 2987245,
    "TransactionAmt": 37.098,
    "ProductCD": "C",
    "card1": 13413,
    "card2": 103.0,
    "card4": "visa",
    "card6": "credit",
    "addr1": 299,
    "addr2": 87.0,
    "dist1": 19.0,
    "P_emaildomain": "hotmail.com",
    "R_emaildomain": "hotmail.com",
    # ... (rest of 434 features)
}

print("=" * 80)
print("TESTING INFERENCE PIPELINE LOCALLY")
print("=" * 80)

# 1. Load config
config = Config.load()
print(f"\n✅ Config loaded")
print(f"   Base threshold: {config.model.base_threshold}")
print(f"   Model path: {config.model.model_path}")

# 2. Load feature pipeline
feature_pipeline = create_feature_pipeline(config)
print(f"\n✅ Feature pipeline loaded")
print(f"   Features: {len(feature_pipeline.feature_names)}")

# 3. Transform transaction
import time
start = time.time()

tx_df = pd.DataFrame([transaction])
features_df = feature_pipeline.feature_pipeline.transform(tx_df)

duration = time.time() - start
print(f"\n✅ Feature engineering completed")
print(f"   Duration: {duration:.3f}s")
print(f"   Features shape: {features_df.shape}")

# 4. Load model and predict
model_loader = ModelLoader(config)
model_loader.load()

fraud_prob = model_loader.predict(features_df)[0]

print(f"\n✅ XGBoost prediction")
print(f"   Fraud probability: {fraud_prob:.4f} ({fraud_prob*100:.2f}%)")
print(f"   Threshold: {config.model.base_threshold:.4f}")
print(f"   Decision: {'FRAUD' if fraud_prob >= config.model.base_threshold else 'LEGIT'}")

# 5. Show feature values
print(f"\n📊 Top 10 Most Important Features:")
feature_vals = features_df.iloc[0].to_dict()
top_features = sorted(feature_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
for fname, fval in top_features:
    print(f"   {fname}: {fval:.4f}")

print("\n" + "=" * 80)
print("✅ LOCAL TEST COMPLETE")
print("=" * 80)
```

**Run this on your VM to see:**
- If feature engineering is slow locally too
- Exact feature values being fed to XGBoost
- If local prediction matches Spark prediction

---

## 🔧 Recommended Fixes

### Fix #1: Add Performance Logging

```python
# src/inference/main_inference.py - Add after line 96

import time

for idx, row in batch_df.iterrows():
    transaction = row.to_dict()
    transaction_id = transaction.get('TransactionID', f'unknown_{idx}')
    
    tx_start = time.time()

    try:
        # 1. Feature engineering
        feat_start = time.time()
        features_df = feature_pipeline.feature_pipeline.transform(
            pd.DataFrame([transaction])
        )
        feat_time = time.time() - feat_start
        
        # 2. ML prediction
        pred_start = time.time()
        fraud_prob = float(model_loader.predict(features_df)[0])
        pred_time = time.time() - pred_start
        
        # ... rest of code ...
        
        total_time = time.time() - tx_start
        
        # Log performance
        logger.info(
            f"TX {transaction_id}: "
            f"fraud_prob={fraud_prob:.4f}, "
            f"decision={decision_result.decision.value}, "
            f"feat_time={feat_time:.2f}s, "
            f"pred_time={pred_time:.2f}s, "
            f"total_time={total_time:.2f}s"
        )
```

This will show you in logs:
- Exact fraud probability for each transaction
- Time breakdown (feature eng vs prediction vs decision)
- Where the bottleneck is

### Fix #2: Add Output Validation

```python
# src/inference/main_inference.py - Add before results.append(output)

# Validate output before adding
assert 'transaction_id' in output, "Missing transaction_id"
assert 'fraud_probability' in output, "Missing fraud_probability"
assert 0.0 <= output['fraud_probability'] <= 1.0, f"Invalid probability: {output['fraud_probability']}"
assert output['decision'] in ['APPROVE', 'REVIEW', 'HOLD', 'BLOCK'], f"Invalid decision: {output['decision']}"

logger.info(f"✅ TX {transaction_id}: prob={output['fraud_probability']:.4f}, decision={output['decision']}")
```

---

## 📊 Summary

**Current Status:**
- ✅ Spark IS processing transactions (1 input → 1 output)
- ⚠️ Processing is SLOW (17.8s per transaction - should be < 0.1s)
- ✅ Dashboard IS showing data correctly
- ❓ **What's actually "wrong"?** Need clarification from you

**Your Transaction Results:**
- Fraud Prob: 0.019 (1.9%) ✅
- Decision: APPROVE ✅
- Risk Level: LOW ✅
- Risk Factors: [] ✅

**This is EXACTLY what a legitimate $37 Visa transaction should look like!**

---

## ❓ Next Steps

**Please clarify:**
1. Is this transaction actually fraudulent? (ground truth)
2. What SHOULD the dashboard show for this transaction?
3. What specific values are "incorrect"?

**If the issue is model accuracy (predicting 0.019 when should be higher):**
- Check training data has similar transactions labeled as fraud
- Verify feature engineering matches training
- Retrain model with updated data

**If the issue is performance (17.8s too slow):**
- Run local diagnostic script
- Check Redis connection latency
- Profile feature engineering code

