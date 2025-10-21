# Balanced Approach Implementation - Model Improvements

**Date**: October 21, 2025
**Goal**: Improve fraud detection recall from 33% to 60-70% while maintaining reasonable precision

## Changes Summary

### 1. SMOTE Sampling Strategy ✅
**File**: `src/dags/ieee_cis_training.py` (Line ~1115)

**Before**:
```python
sampling_strategy = 0.5  # Fraud class = 50% of majority class
```

**After**:
```python
sampling_strategy = 0.7  # Fraud class = 70% of majority class
```

**Impact**: More synthetic fraud samples for better learning of fraud patterns

---

### 2. XGBoost Hyperparameters ✅
**File**: `src/dags/ieee_cis_training.py` (Line ~757)

**Before**:
```python
max_depth=8
```

**After**:
```python
max_depth=6  # Reduced for better generalization
```

**Impact**: Less overfitting, better recall on unseen data

---

### 3. LightGBM - GPU Version ✅
**File**: `src/dags/ieee_cis_training.py` (Line ~776)

**Before**:
```python
num_leaves=63
# No class_weight parameter
```

**After**:
```python
num_leaves=63
class_weight='balanced'  # NEW - better fraud detection
is_unbalance=True        # NEW - handles imbalanced classes
```

**Impact**: Model pays more attention to minority (fraud) class

---

### 4. LightGBM - CPU Fallback ✅
**File**: `src/dags/ieee_cis_training.py` (Line ~799)

**Before**:
```python
num_leaves=63
# No class_weight parameter
```

**After**:
```python
num_leaves=63
class_weight='balanced'  # NEW
is_unbalance=True        # NEW
```

---

### 5. LightGBM - CPU Only ✅
**File**: `src/dags/ieee_cis_training.py` (Line ~816)

**Before**:
```python
num_leaves=95
# No class_weight parameter
```

**After**:
```python
num_leaves=63            # Reduced from 95
class_weight='balanced'  # NEW
is_unbalance=True        # NEW
```

**Impact**: Better generalization + fraud focus

---

### 6. CatBoost - GPU Version ✅
**File**: `src/dags/ieee_cis_training.py` (Line ~845)

**Before**:
```python
depth=10
class_weights=[1.0, scale_pos_weight]  # scale_pos_weight ≈ 28
```

**After**:
```python
depth=6  # Reduced from 10
class_weights=[1.0, 30.0]  # Increased to 30 for better fraud detection
```

**Impact**: Less overfitting + stronger fraud signal

---

### 7. CatBoost - CPU Fallback ✅
**File**: `src/dags/ieee_cis_training.py` (Line ~860)

**Before**:
```python
depth=10
class_weights=[1.0, scale_pos_weight]
```

**After**:
```python
depth=6
class_weights=[1.0, 30.0]
```

---

### 8. CatBoost - CPU Only ✅
**File**: `src/dags/ieee_cis_training.py` (Line ~875)

**Before**:
```python
depth=8
class_weights=[1.0, scale_pos_weight]
```

**After**:
```python
depth=6  # Reduced from 8
class_weights=[1.0, 30.0]  # Increased from scale_pos_weight
```

---

## Expected Results

### Before Changes:
- Accuracy: ~82%
- Recall: **33%** (catches only 1 in 3 fraud cases)
- Precision: ~28%
- F1-Score: 0.30
- AUC-PR: 0.23

### After Changes (Target):
- Accuracy: 85-90%
- Recall: **60-70%** (catches 6-7 in 10 fraud cases) ← **2x improvement!**
- Precision: 35-45%
- F1-Score: 0.45-0.55
- AUC-PR: 0.30-0.35

---

## Why This Works

1. **Higher SMOTE ratio** (0.7): More fraud patterns for model to learn
2. **Reduced tree depth** (6 vs 8-10): Prevents memorizing training data
3. **Class weighting**: Forces models to care more about fraud detection
4. **is_unbalance flag**: LightGBM optimizes for imbalanced data
5. **Higher CatBoost weights** (30 vs 28): Stronger signal for fraud class

---

## Key Trade-offs

✅ **Gains**:
- Catch **2x more fraud transactions** (33% → 65%)
- Better real-world performance
- More fraud patterns detected

⚠️ **Trade-offs**:
- Slightly more false positives (can be reviewed manually)
- Lower precision (but much better recall)
- Focus on **AUC-PR** (right metric for imbalanced data)

---

## Next Steps

1. **Commit changes** to repository
2. **Retrain model** on server:
   ```bash
   cd ~/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/src
   docker compose restart airflow-scheduler airflow-worker
   ```
3. **Monitor new metrics** in training logs
4. **Compare results** with previous training run
5. **Fine-tune class_weights** if needed (experiment with 25-35 range)

---

## Monitoring

Watch for these metrics in training logs:
- `Recall (class 1)` - Should increase to 0.60+
- `F1-score (class 1)` - Should increase to 0.45+
- `AUC-PR` - Should increase to 0.30+
- `Support (class 1)` - Number of actual fraud cases in test set

---

## References

Based on analysis of:
- `SMOTE.ipynb` from IEEE-CIS competition (98.4% accuracy, 24% recall)
- `baseline-model.ipynb` feature importance analysis
- IEEE-CIS Kaggle competition winning solutions

**Key Insight**: High accuracy (98%) with low recall (24%) is worse than balanced accuracy (85%) with good recall (65%) for fraud detection.
