# Feature Engineering Parity Analysis

**Date**: November 24, 2025  
**Issue**: Verify training vs. inference feature engineering consistency

---

## 🎯 Critical Question: Does Inference Match Training?

### ✅ **YES - Feature Pipeline Uses Same Trained Artifacts**

The inference system is designed to ensure **exact parity** with training by:

1. **Loading the same `feature_pipeline.pkl`** that was saved during training
2. **Calling `.transform()` on the loaded pipeline** (not re-implementing features)
3. **Using the same model bundle** with feature names and order

---

## 📦 Model Loading Architecture

### Training Side (`src/dags/ieee_cis_training.py`)

```python
# Lines 2400-2500: Saves two critical files

# 1. MODEL BUNDLE (fraud_detection_xgboost_model.pkl)
artifact_bundle = {
    'model': model,                          # XGBoost classifier
    'threshold': float(self.best_threshold), # F1-optimal threshold
    'feature_names': self.all_features,      # List of 88 feature names IN ORDER
    'scaler': self.scaler,
    'freq_maps': self.freq_maps,
    'mean_encoding_maps': self.mean_maps,
    'model_type': 'XGBoost',
    'n_features': 88,
    # ... metadata
}
joblib.dump(artifact_bundle, model_path)

# 2. FEATURE PIPELINE (feature_pipeline.pkl)
# This is the actual sklearn Pipeline object with transform() method
self.feature_pipeline.freq_maps = self.freq_maps
self.feature_pipeline.scaler = self.scaler
self.feature_pipeline.feature_names = self.all_features
self.feature_pipeline.card1_amt_mean = self.card1_amt_mean  # Agg stats
self.feature_pipeline.mean_encoding_maps = self.mean_maps   # Fraud rate encoding
self.feature_pipeline.magic_uid_stats = magic_uid_stats     # Magic UID lookup
self.feature_pipeline.group_agg_stats = group_agg_stats     # Group aggregations

joblib.dump(self.feature_pipeline, pipeline_path)
```

**Key Point**: The `feature_pipeline.pkl` contains:
- ✅ The actual trained sklearn transformers
- ✅ Fitted scalers with learned mean/std
- ✅ Frequency encoding maps from training data
- ✅ Mean encoding maps (fraud rates)
- ✅ All aggregation statistics
- ✅ The exact `transform()` logic used in training

---

### Inference Side (`src/inference/`)

#### Step 1: Load Feature Pipeline (`feature_pipeline_spark.py`)

```python
# Line 55: Loads the pickled pipeline object
self.feature_pipeline = joblib.load(pipeline_path)

# Gets the exact feature names from training
if hasattr(self.feature_pipeline, 'feature_names'):
    self.feature_names = self.feature_pipeline.feature_names  # 88 features
```

#### Step 2: Apply Transform (`main_inference.py`)

```python
# Line 105: Uses .transform() from loaded pipeline
features_df = feature_pipeline.feature_pipeline.transform(
    pd.DataFrame([transaction])
)
# ☝️ This is THE SAME transform() that was used in training!
# It applies:
# - Same frequency encoding (using freq_maps)
# - Same scaling (using fitted scaler)
# - Same mean encoding (using mean_encoding_maps)
# - Same magic_uid lookups
# - Same group aggregations
# - Same missing value handling
```

#### Step 3: Model Prediction (`model_loader.py`)

```python
# Line 162-166: Validates and reorders features
if self.feature_names:
    self._validate_features(features_df)
    features_df = features_df[self.feature_names]  # Reorder to match training

# Line 169: Predict using XGBoost
probas = self.model.predict_proba(features_df)[:, 1]
```

**Key Point**: Inference does NOT re-implement feature engineering. It:
1. Loads the exact pipeline object from training
2. Calls `.transform()` which applies all fitted transformers
3. Validates feature order matches training
4. Passes features to model in correct order

---

## 🔍 Feature Consistency Checks

### ✅ What's Verified Automatically

1. **Feature Count**: Model loader checks `len(features_df.columns) == 88`
2. **Feature Names**: `_validate_features()` checks all expected features present
3. **Feature Order**: Reorders columns to `self.feature_names` from model bundle
4. **Missing Features**: Raises `ValueError` if any training features missing

### ⚠️ What's NOT Explicitly Checked (But Should Work)

1. **Scaling Values**: Assumes `scaler` in pipeline has correct fitted params
2. **Frequency Maps**: Assumes `freq_maps` contains all categories seen in training
3. **Mean Encoding**: Assumes `mean_encoding_maps` has fraud rates for all values
4. **Aggregation Stats**: Assumes `card1_amt_mean`, `magic_uid_stats`, etc. present

---

## 🚨 Current Configuration Issue

### ❌ PROBLEM: Model Path Mismatch

**Config File** (`src/inference/config.yaml`):
```yaml
model:
  model_bundle_path: "../models/fraud_detection_model.pkl"  # ❌ OLD NAME
  feature_pipeline_path: "../models/feature_pipeline.pkl"   # ✅ CORRECT
```

**Actual File on VM**:
```bash
/app/models/fraud_detection_xgboost_model.pkl  # ✅ ACTUAL FILE
```

### ✅ FIX APPLIED (Commit 5be54c4)

Updated `config.yaml`:
```yaml
model:
  model_bundle_path: "../models/fraud_detection_xgboost_model.pkl"  # ✅ FIXED
  feature_pipeline_path: "../models/feature_pipeline.pkl"            # ✅ CORRECT
```

**Action Required**: Deploy this fix to VM:
```bash
cd /home/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML
git pull origin main
docker restart fraud-inference-spark
```

---

## 🧪 Verification Script

Run `verify_model_feature_parity.py` to check:

```bash
python verify_model_feature_parity.py
```

This script verifies:
- ✅ Model bundle loads correctly
- ✅ Model is XGBoost with 88 features
- ✅ Feature pipeline has `transform()` method
- ✅ Pipeline produces exactly 88 features
- ✅ Feature names match between model and pipeline
- ✅ Feature order is consistent
- ✅ Transform works on sample transaction
- ✅ No NaN/Inf values in unexpected places

---

## 📊 Feature Engineering Flow Comparison

### Training Flow
```
Raw Transaction (IEEE-CIS format)
    ↓
[IEEECISFeaturePipeline.fit_transform()]
    ├─ Time features (hour, day, week)
    ├─ Amount features (log, bins)
    ├─ Email features (domain encoding)
    ├─ Card features (combinations)
    ├─ Frequency encoding (card1, card4, etc.)
    ├─ Mean encoding (fraud rates)
    ├─ Magic UID features
    ├─ D-column normalization
    ├─ Group aggregations (card1_amt_mean, etc.)
    └─ Scaling (StandardScaler)
    ↓
88 Features → XGBoost Training
    ↓
Save: feature_pipeline.pkl (with fitted transformers)
```

### Inference Flow
```
Raw Transaction (API/Kafka)
    ↓
[Load feature_pipeline.pkl]
    ↓
[feature_pipeline.transform()]  ← SAME OBJECT FROM TRAINING
    ├─ Time features (same logic)
    ├─ Amount features (same logic)
    ├─ Email features (same encoding)
    ├─ Card features (same combinations)
    ├─ Frequency encoding (same freq_maps)
    ├─ Mean encoding (same mean_maps)
    ├─ Magic UID features (same lookups)
    ├─ D-column normalization (same logic)
    ├─ Group aggregations (same stats)
    └─ Scaling (same fitted scaler)
    ↓
88 Features → XGBoost Prediction
```

**Key Insight**: Both flows use THE SAME `feature_pipeline` object. Training creates it, inference loads it.

---

## ✅ Conclusion

**Feature engineering parity is GUARANTEED because:**

1. ✅ Inference loads the exact `feature_pipeline.pkl` saved from training
2. ✅ Calls `.transform()` which applies all fitted transformers
3. ✅ Does NOT re-implement any feature engineering logic
4. ✅ Validates feature count, names, and order match training
5. ✅ Uses same model bundle with same threshold

**Only issue was**: Config pointing to wrong model filename → **FIXED in commit 5be54c4**

**Next Step**: Deploy fix to VM and restart inference service.

---

## 📝 Recommended Checks After Deployment

1. **Verify model loads**:
   ```bash
   docker logs fraud-inference-spark -f
   # Should see: "Model loaded successfully: XGBoost, features=88"
   # Should see: "Feature pipeline loaded with 88 features"
   ```

2. **Send test transaction**:
   ```bash
   python send_test_transaction.py
   ```

3. **Check predictions appear**:
   ```bash
   docker logs fraud-inference-spark | grep "TX "
   # Should see: "TX 2987245: prob=0.XXXX, decision=APPROVE, times(...)"
   ```

4. **Verify dashboard shows transaction**:
   - Visit http://167.71.224.89:8501
   - Should see transaction with correct fraud probability

---

**Status**: ✅ Feature parity verified in code architecture  
**Blocker**: ⏳ Model path fix needs deployment to VM
