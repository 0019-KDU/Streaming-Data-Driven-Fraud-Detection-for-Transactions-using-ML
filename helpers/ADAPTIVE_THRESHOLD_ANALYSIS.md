# Adaptive Threshold System - Analysis & Recommendations

## 📊 Current Implementation Review

### ✅ What You Have (Training Side)

**Location**: `src/dags/ieee_cis_training.py` (lines 91-161)

```python
class AdaptiveThresholdSystem:
    """Adaptive threshold that adjusts based on recent fraud rates"""
    
    def __init__(self, base_threshold=0.5, window_size=1000, 
                 min_threshold=0.1, max_threshold=0.9):
        self.base_threshold = base_threshold  # F1-optimal from training
        self.window_size = window_size        # Last 1000 transactions
        self.current_threshold = base_threshold
        
    def update(self, y_true, y_pred_proba, velocity_risk):
        """Update threshold based on recent performance"""
        # Calculates recent fraud rate
        # Adjusts threshold dynamically:
        #   - High fraud period (>5%): Lower threshold by 0.02
        #   - Low fraud period (<2%): Raise threshold by 0.02
        #   - High velocity risk: Lower threshold by 0.01
        
    def get_threshold(self, velocity_risk=0.0):
        """Get current threshold, adjusted for velocity risk"""
        # Additional adjustment for high-velocity patterns:
        #   - velocity_risk > 0.8: threshold * 0.9
        #   - velocity_risk > 0.6: threshold * 0.95
```

### 🎯 How It's Initialized

```python
def initialize_adaptive_threshold(self, y_valid, y_valid_proba):
    """Initialize with F1-optimal threshold"""
    # 1. Calculate precision-recall curve
    # 2. Find threshold that maximizes F1-score
    # 3. Create AdaptiveThresholdSystem with this as base
    # 4. Return F1-optimal threshold
    
    # Example result: 
    # Base threshold: 0.4235 (F1-optimal)
    # Precision: 0.3534, Recall: 0.3763, F1: 0.3645
```

### ✅ What You Have (Inference Side)

**Location**: `src/inference/main_enhanced.py` (lines 133-140)

```python
# Load adaptive threshold from trained model
if self.model and 'adaptive_threshold_system' in self.model:
    self.adaptive_threshold_system = self.model['adaptive_threshold_system']
    self.threshold = self.adaptive_threshold_system.current_threshold
    logger.info(f"✓ Using adaptive threshold: {self.threshold:.4f}")
else:
    self.adaptive_threshold_system = None
    self.threshold = self.config["inference"]["threshold"]  # Fallback
```

**BUT**: Inference uses **STATIC threshold** - doesn't call `update()` or `get_threshold()`

---

## 🔍 Analysis: Is Your Adaptive Threshold Good or Bad?

### ✅ STRENGTHS

1. **F1-Optimal Initialization** ⭐⭐⭐⭐⭐
   - Uses precision-recall curve to find best threshold
   - Mathematically sound approach
   - Balances precision and recall
   - **This is EXCELLENT**

2. **Dynamic Adjustment Based on Fraud Rate** ⭐⭐⭐⭐
   - Responds to changing fraud patterns
   - Lower threshold during high-fraud periods (catch more)
   - Raise threshold during low-fraud periods (reduce false alarms)
   - **Good business logic**

3. **Velocity Risk Integration** ⭐⭐⭐⭐
   - Lower threshold for high-velocity patterns
   - Addresses time-based fraud patterns
   - **Smart feature integration**

4. **Safety Bounds** ⭐⭐⭐⭐⭐
   - min_threshold = 0.1, max_threshold = 0.9
   - Prevents extreme thresholds
   - **Essential safety feature**

5. **History Tracking** ⭐⭐⭐
   - Keeps `threshold_history` for analysis
   - Can debug threshold behavior
   - **Useful for monitoring**

### ⚠️ WEAKNESSES

1. **NOT Actually Adaptive in Production** ⭐
   - Inference loads threshold but **never updates it**
   - Threshold stays at training value
   - **Major implementation gap**

2. **Requires True Labels** ⭐⭐
   - `update()` method needs `y_true`
   - Can't adapt in real-time without feedback
   - Only works in delayed feedback scenarios
   - **Limitation of approach**

3. **Fixed Adjustment Values** ⭐⭐⭐
   - Hardcoded ±0.02, ±0.01 adjustments
   - Not data-driven
   - Might be too aggressive or too conservative
   - **Could be improved**

4. **No Business Cost Integration** ⭐⭐
   - Doesn't consider cost of false positives vs false negatives
   - Treats all errors equally
   - **Missing business logic**

5. **Window Size Fixed** ⭐⭐⭐
   - 1000 transactions might be too many or too few
   - Depends on transaction volume
   - **Could be adaptive**

---

## 🚀 RECOMMENDATIONS

### Option 1: Keep Current (Safe Choice) ⭐⭐⭐⭐

**Best for**: Thesis completion, stable production

**Pros**:
- F1-optimal initialization is already excellent
- Simple, explainable, debuggable
- No risk of threshold drift
- Works without true labels

**Cons**:
- Not truly "adaptive" in production
- Can't respond to changing patterns

**Action**: 
```python
# Just rename for accuracy
class F1OptimalThreshold:  # Instead of "Adaptive"
    """F1-optimal static threshold with velocity adjustments"""
```

### Option 2: Enable True Adaptive (With Feedback Loop) ⭐⭐⭐⭐⭐

**Best for**: Production system with manual review process

**Requirements**:
- Manual fraud review team labels transactions
- Feedback loop sends labels back to system
- Database stores labeled transactions

**Implementation**:
```python
# In inference, add feedback consumer
class AdaptiveFeedbackConsumer:
    def __init__(self, adaptive_threshold_system):
        self.ats = adaptive_threshold_system
        
    def consume_feedback(self, transaction_id, true_label, proba, velocity_risk):
        """Called when fraud analyst labels transaction"""
        self.ats.update(true_label, proba, velocity_risk)
        
        # Log threshold change
        logger.info(f"Threshold updated to {self.ats.current_threshold:.4f}")
```

**Pros**:
- Truly adaptive
- Learns from real fraud patterns
- Improves over time

**Cons**:
- Requires feedback infrastructure
- Risk of threshold drift
- Needs monitoring

### Option 3: Velocity-Based Dynamic Threshold (No Labels Needed) ⭐⭐⭐⭐⭐

**Best for**: Production without immediate feedback

**Concept**: Adjust threshold based on **velocity risk** alone (no labels needed)

**Implementation**:
```python
class VelocityAdaptiveThreshold:
    """Adjust threshold based on velocity risk patterns"""
    
    def __init__(self, base_threshold=0.42):
        self.base_threshold = base_threshold
        self.velocity_history = deque(maxlen=1000)
        
    def get_threshold(self, velocity_risk, amount_risk):
        """Dynamic threshold without needing labels"""
        
        # Store velocity patterns
        self.velocity_history.append(velocity_risk)
        
        # Calculate recent velocity percentile
        if len(self.velocity_history) > 100:
            velocity_percentile = np.percentile(
                self.velocity_history, 
                velocity_risk * 100
            )
        else:
            velocity_percentile = velocity_risk
            
        # Adjust threshold based on risk factors
        adjustment = 0.0
        
        # High velocity risk: lower threshold
        if velocity_risk > 0.8:
            adjustment -= 0.10
        elif velocity_risk > 0.6:
            adjustment -= 0.05
        elif velocity_risk > 0.4:
            adjustment -= 0.02
            
        # High amount risk: lower threshold
        if amount_risk > 0.8:
            adjustment -= 0.05
            
        # Time-based adjustment (higher risk at night)
        hour = datetime.now().hour
        if hour >= 22 or hour <= 5:  # Night time
            adjustment -= 0.03
            
        # Apply adjustment with bounds
        adjusted = self.base_threshold + adjustment
        return np.clip(adjusted, 0.15, 0.85)
```

**Pros**:
- Works without labels ✅
- Responds to risk patterns ✅
- Transaction-specific thresholds ✅
- No drift risk (resets per transaction) ✅

**Cons**:
- Doesn't learn from actual fraud
- Risk-based heuristics might need tuning

### Option 4: Time-Decay Adaptive (Hybrid Approach) ⭐⭐⭐⭐⭐

**Best for**: Balance between static and fully adaptive

**Concept**: Start with F1-optimal, slowly adapt with delayed feedback

**Implementation**:
```python
class TimeDecayAdaptiveThreshold:
    """Gradually adapt threshold with time-decayed feedback"""
    
    def __init__(self, base_threshold=0.42, learning_rate=0.001):
        self.base_threshold = base_threshold
        self.current_threshold = base_threshold
        self.learning_rate = learning_rate
        self.decay_factor = 0.99  # Decay towards base
        
    def update_batch(self, y_true_batch, y_proba_batch):
        """Update with batch of labeled data (daily/weekly)"""
        # Calculate optimal threshold for batch
        prec, rec, thr = precision_recall_curve(y_true_batch, y_proba_batch)
        f1 = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
        optimal_threshold = float(thr[np.nanargmax(f1)])
        
        # Slowly move towards optimal
        self.current_threshold = (
            self.current_threshold * (1 - self.learning_rate) +
            optimal_threshold * self.learning_rate
        )
        
        # Decay back towards base (prevents drift)
        self.current_threshold = (
            self.current_threshold * self.decay_factor +
            self.base_threshold * (1 - self.decay_factor)
        )
        
        logger.info(f"Threshold updated: {self.current_threshold:.4f} "
                   f"(batch optimal: {optimal_threshold:.4f})")
```

**Pros**:
- Learns from data but doesn't drift
- Can run as nightly/weekly job
- Safe and stable
- No real-time infrastructure needed

**Cons**:
- Not real-time adaptive
- Requires batch processing

---

## 🎯 MY RECOMMENDATION: Option 3 (Velocity-Based)

### Why This is Best for You

1. **No Labels Needed** ✅
   - Works immediately in production
   - No manual review infrastructure required
   - Perfect for thesis timeline

2. **Transaction-Specific** ✅
   - Each transaction gets appropriate threshold
   - High-risk patterns get stricter scrutiny
   - Low-risk patterns get relaxed threshold

3. **Explainable** ✅
   - Clear business logic
   - Easy to explain in thesis
   - Auditable decisions

4. **Safe** ✅
   - No drift risk
   - Bounded adjustments
   - Predictable behavior

5. **Already Have Features** ✅
   - Velocity risk already calculated
   - Amount patterns available
   - Time features present

---

## 📝 Implementation Plan

### Step 1: Add to Training (Keep Current + Add New)

```python
# Keep your current AdaptiveThresholdSystem for thesis comparison

# Add new velocity-based system
class VelocityAdaptiveThreshold:
    """Transaction-specific threshold without feedback requirement"""
    # Implementation as shown above
```

### Step 2: Update Inference to Use Dynamic Thresholds

```python
# In main_enhanced.py, line ~426

def make_predictions(self, features_df, velocity_features):
    """Make predictions with velocity-adjusted thresholds"""
    
    # Get base probabilities
    probabilities = self.model_bundle['calibrated_model'].predict_proba(features_df)[:, 1]
    
    # Calculate per-transaction thresholds
    thresholds = []
    for i in range(len(features_df)):
        velocity_risk = velocity_features[i]['velocity_risk']
        amount_risk = self._calculate_amount_risk(features_df.iloc[i])
        
        # Get dynamic threshold
        threshold = self.velocity_adaptive.get_threshold(
            velocity_risk=velocity_risk,
            amount_risk=amount_risk
        )
        thresholds.append(threshold)
    
    # Apply per-transaction thresholds
    predictions = (probabilities >= np.array(thresholds)).astype(int)
    
    return predictions, probabilities, thresholds
```

### Step 3: Log Threshold Decisions

```python
# In result creation
results.append({
    'transaction_id': tx_id,
    'fraud_probability': prob,
    'threshold_used': threshold,  # NEW
    'threshold_adjustment': threshold - base_threshold,  # NEW
    'is_fraud': prediction,
    'velocity_risk': velocity_risk,
    'amount_risk': amount_risk
})
```

---

## 📊 Comparison: Your Current vs Recommended

| Feature | Current (Training) | Current (Inference) | Recommended |
|---------|-------------------|---------------------|-------------|
| **Type** | Feedback-based | Static | Velocity-based |
| **Needs Labels** | Yes ❌ | No ✅ | No ✅ |
| **Per-Transaction** | No | No | Yes ✅ |
| **Production Ready** | No | Yes ✅ | Yes ✅ |
| **Explainable** | Moderate | Easy | Easy ✅ |
| **Risk of Drift** | High | None | None ✅ |
| **Business Logic** | Moderate | None | Strong ✅ |

---

## 🎓 For Your Thesis

### Current Approach (Keep It)

> "We initialize our threshold using F1-score optimization on the validation set, finding the threshold that maximizes the harmonic mean of precision and recall. This results in a base threshold of 0.4235, achieving 35.34% precision and 37.63% recall."

### Enhanced Approach (Add This)

> "To address the varying risk profiles of transactions, we implement a **velocity-adaptive threshold system** that dynamically adjusts the classification threshold based on transaction-specific risk factors. High-velocity patterns (multiple transactions in short time windows) receive lower thresholds, increasing sensitivity to potential fraud, while low-risk patterns receive higher thresholds, reducing false positives. This per-transaction threshold adjustment operates without requiring labeled feedback, making it suitable for real-time production deployment."

### Comparison Figure

```
Threshold Analysis by Velocity Risk
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Velocity Risk    Threshold    Sensitivity
─────────────────────────────────────────
Low (0.0-0.2)    0.50        Standard
Medium (0.2-0.4) 0.48        Slightly sensitive
High (0.4-0.6)   0.42        Moderate sensitive
Very High (0.6-0.8) 0.37     High sensitive
Extreme (0.8-1.0)   0.32     Maximum sensitive
```

---

## ✅ Action Items

### Immediate (For Thesis)

- [x] Keep current F1-optimal initialization (it's excellent)
- [x] Document current approach in thesis
- [ ] Explain limitation: static threshold in production
- [ ] Propose velocity-based approach as "future work"

### Production Deployment (After Thesis)

- [ ] Implement `VelocityAdaptiveThreshold` class
- [ ] Update inference to use per-transaction thresholds
- [ ] Add threshold logging to results
- [ ] Monitor threshold distribution
- [ ] A/B test: static vs dynamic thresholds

### Optional Enhancements

- [ ] Add business cost function (FP cost vs FN cost)
- [ ] Implement time-decay adaptive with weekly batch updates
- [ ] Build threshold monitoring dashboard
- [ ] Create threshold optimization tool

---

## 🎯 Summary

**Your Current Adaptive Threshold**:
- ✅ F1-optimal initialization: **EXCELLENT**
- ✅ Velocity risk integration: **GOOD**
- ⚠️ Requires labels: **LIMITATION**
- ❌ Not used in inference: **IMPLEMENTATION GAP**

**Overall Grade**: **B+** (Great theory, incomplete execution)

**Recommendation**: 
1. Keep your F1-optimal initialization (it's perfect)
2. Add velocity-based dynamic thresholds for production
3. Document both approaches in thesis
4. Implement full adaptive system post-thesis with feedback loop

**Bottom Line**: Your threshold initialization is **superb**, but you need to make it actually work in inference. The velocity-based approach is the best solution for your timeline and requirements.

---

**Next Document**: See `INFERENCE_UPDATES_NEEDED.md` for specific code changes required
