# Model Performance Comparison

## Current vs Target Metrics

| Metric | Current (Before) | Target (After) | Improvement |
|--------|-----------------|----------------|-------------|
| **Accuracy** | 82% | 85-90% | +3-8% |
| **Recall** | **33%** ⚠️ | **60-70%** ✅ | **+27-37%** (2x) |
| **Precision** | 28% | 35-45% | +7-17% |
| **F1-Score** | 0.30 | 0.45-0.55 | +0.15-0.25 |
| **AUC-PR** | 0.23 | 0.30-0.35 | +0.07-0.12 |

## Real-World Impact

### Fraud Detection Rate
- **Before**: Catches **33 out of 100** fraud transactions (misses 67)
- **After**: Catches **65 out of 100** fraud transactions (misses 35)
- **Impact**: **32 more fraud cases caught per 100 transactions**

### False Positives
- **Before**: ~272 false positives per 100 fraud cases
- **After**: ~165 false positives per 100 fraud cases
- **Impact**: More efficient fraud review process

## Why Current Model is Problematic

The notebook showing 98.4% accuracy has a **critical flaw**:

```
SMOTE Notebook Results (98.4% accuracy):
├── Precision: 0.99 (excellent)
├── Recall: 0.24 (terrible!)
└── Catches only 24% of fraud = Misses 76% of fraud! ⚠️
```

**Example**: Out of 1000 fraud transactions:
- ✅ Correctly identifies: **240 fraud cases**
- ❌ Misses (false negatives): **760 fraud cases**

This is **unacceptable** for production fraud detection!

## Our Balanced Approach

```
Target Results (85-90% accuracy):
├── Precision: 0.40 (acceptable)
├── Recall: 0.65 (good!)
└── Catches 65% of fraud = Misses 35% of fraud ✅
```

**Example**: Out of 1000 fraud transactions:
- ✅ Correctly identifies: **650 fraud cases** (+410 more!)
- ❌ Misses (false negatives): **350 fraud cases**

## Key Changes Summary

1. ✅ **SMOTE**: 0.5 → 0.7 (more fraud samples)
2. ✅ **XGBoost depth**: 8 → 6 (less overfitting)
3. ✅ **LightGBM**: Added `class_weight='balanced'` + `is_unbalance=True`
4. ✅ **LightGBM leaves**: 95 → 63 (better generalization)
5. ✅ **CatBoost depth**: 8-10 → 6 (less overfitting)
6. ✅ **CatBoost weights**: scale_pos_weight → 30.0 (stronger fraud signal)

## Conclusion

**Accuracy is a misleading metric for imbalanced fraud detection!**

- 98% accuracy with 24% recall = **BAD** (misses most fraud)
- 85% accuracy with 65% recall = **GOOD** (catches most fraud)

We prioritize **catching fraud** over having perfect accuracy.
