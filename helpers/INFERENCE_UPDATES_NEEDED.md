# Inference Pipeline Updates Needed

## 🎯 Overview

Your training pipeline (`ieee_cis_training.py`) has many enhancements that need to be synchronized with inference (`main_enhanced.py`).

---

## 📊 Gap Analysis: Training vs Inference

### ✅ What's Already Synced

| Feature | Training | Inference | Status |
|---------|----------|-----------|--------|
| **75 Features** | ✅ Yes | ✅ Yes | ✅ **SYNCED** |
| **Feature Pipeline** | ✅ Saved | ✅ Loaded | ✅ **SYNCED** |
| **Calibrated Model** | ✅ Yes | ✅ Loaded | ✅ **SYNCED** |
| **Velocity Features** | ✅ Yes | ✅ Calculated | ✅ **SYNCED** |
| **Frequency Encoding** | ✅ Yes | ✅ Used | ✅ **SYNCED** |
| **Mean Encoding** | ✅ Yes | ✅ Used | ✅ **SYNCED** |
| **Scaler** | ✅ Saved | ✅ Loaded | ✅ **SYNCED** |

### ⚠️ What Needs Syncing

| Feature | Training | Inference | Status |
|---------|----------|-----------|--------|
| **Adaptive Threshold** | ✅ Trained | ⚠️ Loaded but not used | ⚠️ **PARTIAL** |
| **SMOTE (0.7)** | ✅ Applied | N/A (inference only) | ✅ **OK** |
| **Balanced Hyperparameters** | ✅ Yes | N/A (uses saved model) | ✅ **OK** |
| **MLflow Logging** | ✅ Comprehensive | ❌ No logging | ⚠️ **MISSING** |
| **Threshold Visualization** | ✅ 6 plots | ❌ None | ⚠️ **MISSING** |
| **Per-Transaction Threshold** | ❌ Static | ❌ Static | ❌ **NEEDS IMPLEMENTATION** |

---

## 🔧 Required Updates

### Update 1: Enable Adaptive Threshold ⭐⭐⭐⭐⭐

**Priority**: HIGH  
**Impact**: Improves detection of high-risk transactions  
**Effort**: 30 minutes

**Current Code** (lines 133-140):
```python
# Loads adaptive threshold but doesn't use it
if self.model and 'adaptive_threshold_system' in self.model:
    self.adaptive_threshold_system = self.model['adaptive_threshold_system']
    self.threshold = self.adaptive_threshold_system.current_threshold
else:
    self.threshold = self.config["inference"]["threshold"]
```

**Problem**: Threshold loaded but never adjusted per transaction

**Solution**: Add velocity-based threshold adjustment

```python
def get_transaction_threshold(self, velocity_risk, amount_risk=0.0, hour=12):
    """Get adaptive threshold for specific transaction"""
    
    # Start with base threshold
    if self.adaptive_threshold_system:
        base_threshold = self.adaptive_threshold_system.current_threshold
    else:
        base_threshold = self.threshold
    
    # Adjust based on velocity risk
    adjustment = 0.0
    if velocity_risk > 0.8:
        adjustment -= 0.10
    elif velocity_risk > 0.6:
        adjustment -= 0.05
    elif velocity_risk > 0.4:
        adjustment -= 0.02
    
    # Adjust based on amount risk
    if amount_risk > 0.8:
        adjustment -= 0.05
    elif amount_risk > 0.6:
        adjustment -= 0.03
    
    # Adjust based on time of day (night = higher risk)
    if hour >= 22 or hour <= 5:
        adjustment -= 0.03
    elif hour >= 18 or hour <= 8:
        adjustment -= 0.01
    
    # Apply adjustment with safety bounds
    adjusted_threshold = base_threshold + adjustment
    return np.clip(adjusted_threshold, 0.15, 0.85)
```

**Update Prediction Logic** (line ~576):
```python
# OLD: Fixed threshold
predictions = (adjusted_probabilities >= threshold).astype(int)

# NEW: Per-transaction thresholds
per_transaction_thresholds = []
for i in range(len(features_df)):
    velocity_risk = velocity_results[i].get('velocity_risk', 0.0)
    amount_risk = self._calculate_amount_risk(features_df.iloc[i])
    hour = features_df.iloc[i].get('dt_hour', 12)
    
    tx_threshold = self.get_transaction_threshold(
        velocity_risk=velocity_risk,
        amount_risk=amount_risk,
        hour=hour
    )
    per_transaction_thresholds.append(tx_threshold)

per_transaction_thresholds = np.array(per_transaction_thresholds)
predictions = (adjusted_probabilities >= per_transaction_thresholds).astype(int)
```

**Add to Results** (line ~604):
```python
results.append({
    'transaction_id': str(tx_id),
    'is_fraud': int(is_fraud),
    'fraud_probability': float(prob),
    'threshold_used': float(per_transaction_thresholds[i]),  # NEW
    'threshold_adjustment': float(per_transaction_thresholds[i] - base_threshold),  # NEW
    'velocity_risk': float(velocity_risk),
    'amount_risk': float(amount_risk),  # NEW
    # ... rest of fields
})
```

---

### Update 2: Add Amount Risk Calculation ⭐⭐⭐⭐

**Priority**: MEDIUM  
**Impact**: Better threshold adaptation  
**Effort**: 15 minutes

**Add New Method**:
```python
def _calculate_amount_risk(self, features):
    """Calculate amount-based risk score"""
    amount = features.get('TransactionAmt', 0)
    
    # High amounts are riskier
    if amount > 1000:
        amount_risk = 0.9
    elif amount > 500:
        amount_risk = 0.7
    elif amount > 200:
        amount_risk = 0.5
    elif amount > 100:
        amount_risk = 0.3
    else:
        amount_risk = 0.1
    
    # Check amount patterns from features
    amount_std = features.get('amount_std_1h', 0)
    amount_mean = features.get('amount_mean_1h', amount)
    
    # High deviation = suspicious
    if amount_mean > 0 and amount > 0:
        deviation_ratio = abs(amount - amount_mean) / (amount_mean + 1)
        if deviation_ratio > 2.0:
            amount_risk = min(amount_risk + 0.2, 1.0)
    
    return amount_risk
```

---

### Update 3: Add MLflow Logging to Inference ⭐⭐⭐

**Priority**: LOW (for thesis tracking)  
**Impact**: Track inference performance  
**Effort**: 20 minutes

**Add MLflow Tracking**:
```python
def log_inference_batch(self, predictions, probabilities, thresholds):
    """Log inference batch statistics to MLflow"""
    try:
        import mlflow
        
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment("fraud_detection_inference")
        
        with mlflow.start_run():
            # Log batch statistics
            mlflow.log_metric("batch_size", len(predictions))
            mlflow.log_metric("fraud_detected", int(np.sum(predictions)))
            mlflow.log_metric("fraud_rate", float(np.mean(predictions)))
            mlflow.log_metric("avg_probability", float(np.mean(probabilities)))
            mlflow.log_metric("avg_threshold", float(np.mean(thresholds)))
            mlflow.log_metric("threshold_std", float(np.std(thresholds)))
            
            # Log threshold distribution
            mlflow.log_metric("threshold_min", float(np.min(thresholds)))
            mlflow.log_metric("threshold_max", float(np.max(thresholds)))
            mlflow.log_metric("threshold_median", float(np.median(thresholds)))
            
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")
```

---

### Update 4: Copy AdaptiveThresholdSystem Class ⭐⭐⭐⭐

**Priority**: MEDIUM  
**Impact**: Enable future feedback loop  
**Effort**: 5 minutes

**Add to inference/main_enhanced.py** (after imports):
```python
from collections import deque

class AdaptiveThresholdSystem:
    """
    Adaptive threshold that adjusts based on recent fraud rates
    (Copied from training for compatibility)
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
```

---

### Update 5: Enhanced Result Metrics ⭐⭐⭐

**Priority**: MEDIUM  
**Impact**: Better monitoring and debugging  
**Effort**: 10 minutes

**Enhanced Result Dictionary** (line ~604):
```python
results.append({
    # Identifiers
    'transaction_id': str(tx_id),
    'timestamp': str(tx['timestamp']) if 'timestamp' in tx else None,
    
    # Prediction
    'is_fraud': int(is_fraud),
    'fraud_probability': float(prob),
    'threshold_used': float(tx_threshold),
    'threshold_adjustment': float(tx_threshold - base_threshold),
    
    # Risk Factors
    'velocity_risk': float(velocity_risk),
    'amount_risk': float(amount_risk),
    'velocity_1h': int(velocity_features.get('velocity_1h', 0)),
    'velocity_24h': int(velocity_features.get('velocity_24h', 0)),
    
    # Transaction Details
    'amount': float(tx.get('TransactionAmt', tx.get('amount', 0))),
    'card_type': str(tx.get('card4', 'unknown')),
    'email_domain': str(tx.get('P_emaildomain', 'unknown')),
    
    # Feature Importance (top 5)
    'top_features': self._get_top_feature_values(features_df.iloc[i]),
    
    # Decision Factors
    'rule_based_flag': bool(rule_based_flags[i]),
    'model_based_flag': bool(predictions[i]),
    'final_decision': 'FRAUD' if is_fraud else 'LEGIT',
    
    # Metadata
    'model_version': self.config.get('model', {}).get('version', 'unknown'),
    'inference_time_ms': float(inference_time * 1000)
})
```

---

### Update 6: Add Feature Importance Extraction ⭐⭐

**Priority**: LOW  
**Impact**: Explainability  
**Effort**: 15 minutes

```python
def _get_top_feature_values(self, feature_row, top_n=5):
    """Get top N feature values for this transaction"""
    
    # Get feature importances from model
    if hasattr(self.model_bundle['calibrated_model'].base_estimator, 'feature_importances_'):
        importances = self.model_bundle['calibrated_model'].base_estimator.feature_importances_
        feature_names = self.model_bundle.get('feature_names', self.feature_pipeline.feature_names)
        
        # Get top features
        top_indices = np.argsort(importances)[-top_n:][::-1]
        
        top_features = {}
        for idx in top_indices:
            if idx < len(feature_names):
                feature_name = feature_names[idx]
                feature_value = feature_row.get(feature_name, 'N/A')
                top_features[feature_name] = float(feature_value) if isinstance(feature_value, (int, float)) else str(feature_value)
        
        return top_features
    
    return {}
```

---

## 📋 Implementation Checklist

### Critical Updates (Do First)
- [ ] Add `get_transaction_threshold()` method
- [ ] Update prediction logic to use per-transaction thresholds
- [ ] Add `_calculate_amount_risk()` method
- [ ] Update result dictionary with threshold info
- [ ] Test with sample transactions

### Important Updates (Do Second)
- [ ] Copy `AdaptiveThresholdSystem` class to inference
- [ ] Add threshold adjustment logging
- [ ] Enhance result metrics
- [ ] Add feature importance extraction
- [ ] Test threshold adaptation behavior

### Optional Updates (Nice to Have)
- [ ] Add MLflow inference logging
- [ ] Create threshold monitoring dashboard
- [ ] Add A/B testing capability
- [ ] Build explainability API endpoint

---

## 🧪 Testing Plan

### Test 1: Threshold Adaptation
```python
# Test different velocity risks
test_cases = [
    {'velocity_risk': 0.1, 'expected_adjustment': 0.0},
    {'velocity_risk': 0.5, 'expected_adjustment': -0.02},
    {'velocity_risk': 0.7, 'expected_adjustment': -0.05},
    {'velocity_risk': 0.9, 'expected_adjustment': -0.10}
]

for case in test_cases:
    threshold = inference.get_transaction_threshold(
        velocity_risk=case['velocity_risk'],
        amount_risk=0.0,
        hour=12
    )
    print(f"Velocity Risk: {case['velocity_risk']:.1f} → Threshold: {threshold:.4f}")
```

### Test 2: Amount Risk Calculation
```python
test_amounts = [50, 100, 250, 600, 1500]
for amount in test_amounts:
    features = pd.Series({'TransactionAmt': amount})
    risk = inference._calculate_amount_risk(features)
    print(f"Amount: ${amount} → Risk: {risk:.2f}")
```

### Test 3: End-to-End
```python
# Create test transaction with high risk
test_tx = {
    'TransactionID': 'TEST123',
    'TransactionAmt': 1200,  # High amount
    'card1': 12345,
    'velocity_1h': 15,  # High velocity
    'P_emaildomain': 'tempmail.com'  # Risky domain
}

# Process and check threshold adjustment
result = inference.process_transaction(test_tx)
print(f"Base threshold: 0.42")
print(f"Adjusted threshold: {result['threshold_used']:.4f}")
print(f"Adjustment: {result['threshold_adjustment']:.4f}")
print(f"Decision: {result['final_decision']}")
```

---

## 📊 Expected Impact

### Before Updates
```
All transactions use: threshold = 0.42 (fixed)
- Low risk transaction: 0.42
- Medium risk transaction: 0.42
- High risk transaction: 0.42
```

### After Updates
```
Per-transaction adaptive thresholds:
- Low risk (velocity=0.1): 0.42 (no adjustment)
- Medium risk (velocity=0.5): 0.40 (-0.02 adjustment)
- High risk (velocity=0.8, amount=1500): 0.27 (-0.15 adjustment)
```

**Result**: High-risk transactions get stricter scrutiny (lower threshold = catch more fraud)

---

## 🎯 Summary

### What Needs Syncing

1. **Adaptive Threshold** ⭐⭐⭐⭐⭐
   - Currently: Loaded but not used
   - Need: Per-transaction threshold adjustment
   - Effort: 30 minutes
   - Impact: HIGH

2. **Amount Risk** ⭐⭐⭐⭐
   - Currently: Not calculated
   - Need: Risk score for threshold adjustment
   - Effort: 15 minutes
   - Impact: MEDIUM

3. **Enhanced Results** ⭐⭐⭐
   - Currently: Basic fields
   - Need: Threshold info, risk factors
   - Effort: 10 minutes
   - Impact: MEDIUM

4. **MLflow Logging** ⭐⭐⭐
   - Currently: None
   - Need: Inference metrics tracking
   - Effort: 20 minutes
   - Impact: LOW (for monitoring)

### Total Effort
- **Critical**: 45 minutes
- **Important**: 35 minutes
- **Optional**: 30 minutes
- **Total**: ~2 hours

### Deployment Plan
1. Make critical updates locally
2. Test with sample transactions
3. Commit and push to git
4. Pull on server
5. Restart inference container
6. Monitor threshold adjustments
7. Compare results with baseline

---

**Next Steps**: Implement Update 1 (Adaptive Threshold) first - it has the highest impact.
