# MLflow Tracking Dashboard - Quick Reference

## 📊 Complete Metrics & Visualizations Summary

---

## 🎯 METRICS (22+ Total)

### Core Performance Metrics (5)
```
✓ auc_pr              → Area Under Precision-Recall Curve (0.3271)
✓ auc_roc             → Area Under ROC Curve (0.8295)
✓ precision           → Positive Predictive Value (0.3534)
✓ recall              → Sensitivity / True Positive Rate (0.3763)
✓ f1_score            → Harmonic Mean of Precision & Recall (0.3645)
```

### Advanced Metrics for Imbalanced Data (3)
```
⭐ balanced_accuracy   → Avg of Sensitivity & Specificity [NEW]
⭐ matthews_corrcoef   → MCC - Best for Imbalanced Datasets [NEW]
⭐ cohen_kappa         → Agreement Adjusted for Chance [NEW]
```

### Confusion Matrix Counts (4)
```
✓ true_positives      → Fraud Correctly Identified (TP)
✓ false_positives     → Legit Wrongly Flagged as Fraud (FP)
✓ true_negatives      → Legit Correctly Identified (TN)
✓ false_negatives     → Fraud Wrongly Classified as Legit (FN)
```

### Confusion Matrix Rates (4)
```
⭐ true_positive_rate  → TP / (TP + FN) = Recall [NEW]
⭐ false_positive_rate → FP / (FP + TN) = Type I Error [NEW]
⭐ true_negative_rate  → TN / (TN + FP) = Specificity [NEW]
⭐ false_negative_rate → FN / (FN + TP) = Type II Error [NEW]
```

### Per-Class Metrics (6)
```
⭐ legit_precision     → Precision for Class 0 [NEW]
⭐ legit_recall        → Recall for Class 0 [NEW]
⭐ legit_f1           → F1-Score for Class 0 [NEW]
⭐ fraud_precision     → Precision for Class 1 [NEW]
⭐ fraud_recall        → Recall for Class 1 [NEW]
⭐ fraud_f1           → F1-Score for Class 1 [NEW]
```

---

## 📈 VISUALIZATIONS (6 Publication-Quality Plots)

### 1. Confusion Matrix Heatmap
```
File: confusion_matrix.png
Size: 8×6 inches

┌─────────────────────────────────┐
│     Confusion Matrix            │
│   (Threshold: 0.5000)           │
│                                 │
│           Predicted             │
│         Legit    Fraud          │
│  Legit  [TN]     [FP]           │
│  Fraud  [FN]     [TP]           │
│                                 │
│  Color: Blues (darker = more)   │
└─────────────────────────────────┘

Use: Quick visual of classification performance
```

### 2. Precision-Recall Curve
```
File: precision_recall_curve.png
Size: 10×6 inches

┌─────────────────────────────────┐
│  Precision-Recall Curve         │
│   1.0 ┐                         │
│       │    ╱─────────            │
│   P   │   ╱                      │
│   r   │  ╱  ● Operating Point    │
│   e   │ ╱                        │
│   c   │╱    --- Baseline         │
│   i   │                          │
│   s   │  AUC-PR: 0.3271         │
│   i   │                          │
│   o   │                          │
│   n   │                          │
│   0.0 └───────────────────       │
│       0.0    Recall    1.0       │
└─────────────────────────────────┘

Use: Most important for imbalanced data
     Shows precision/recall tradeoff
```

### 3. ROC Curve
```
File: roc_curve.png
Size: 10×6 inches

┌─────────────────────────────────┐
│       ROC Curve                 │
│   1.0 ┐                         │
│       │      ╱────               │
│   T   │     ╱                    │
│   P   │    ╱  ● Operating Point  │
│   R   │   ╱                      │
│       │  ╱   --- Random (0.5)    │
│       │ ╱                        │
│       │╱  AUC-ROC: 0.8295       │
│       │                          │
│   0.0 └───────────────────       │
│       0.0     FPR      1.0       │
└─────────────────────────────────┘

Use: Overall discrimination ability
     Compare with random baseline
```

### 4. Score Distribution
```
File: score_distribution.png
Size: 14×5 inches (2 subplots)

┌──────────────────┬──────────────┐
│ Histogram        │ Box Plot     │
│                  │              │
│ Density          │    ┌─┐       │
│   ▲              │ F  │ │       │
│   │  ▄▄          │ r  └─┘       │
│   │ █  █  ▄▄     │ a            │
│   │ █  █ █  █    │ u  ┌────┐    │
│   │ █  █ █  █    │ d  │    │    │
│   │ █  █ █  █    │    └────┘    │
│   │ Blue=Legit   │              │
│   │ Red=Fraud    │ L  ┌───────┐ │
│   │ |θ=Threshold │ e  │       │ │
│   └──────────────│ g  └───────┘ │
│     0.0  Score 1.0│  Score      │
└──────────────────┴──────────────┘

Use: Visual separation of classes
     Understand score distributions
```

### 5. Threshold Analysis
```
File: threshold_analysis.png
Size: 12×6 inches

┌─────────────────────────────────┐
│   Metrics vs Threshold          │
│   1.0 ┐                         │
│       │ ─── Precision           │
│       │ ─── Recall              │
│   S   │ ─── F1-Score            │
│   c   │      ╲                  │
│   o   │       ╲    ╱            │
│   r   │        ╲  ╱             │
│   e   │         ╲╱              │
│       │         ╱╲              │
│       │        ╱  ╲             │
│       │       ╱    ╲            │
│   0.0 └───────────────────      │
│       0.0  Threshold  1.0       │
│            |θ=0.5               │
└─────────────────────────────────┘

Use: Optimize threshold selection
     Understand metric tradeoffs
```

### 6. Feature Importance
```
File: feature_importance.png + feature_importance.csv
Size: 10×8 inches

┌─────────────────────────────────┐
│  Top 20 Feature Importances     │
│                                 │
│  velocity_1h       ████████████ │
│  amount_mean_1h    ██████████   │
│  freq_card_1h      ████████     │
│  TransactionAmt    ███████      │
│  amount_std_1h     ██████       │
│  velocity_24h      █████        │
│  card_freq         ████         │
│  ...               ...          │
│  mean_enc_email    ██           │
│                                 │
│  Color: Viridis gradient        │
└─────────────────────────────────┘

Use: Validate feature engineering
     Identify key fraud signals
     + CSV with all 75 features
```

---

## 🔧 PARAMETERS (9 Configuration Values)

```
✓ model_type           → "EnhancedEnsemble" (XGBoost + LightGBM + CatBoost)
✓ n_features           → 75 (11 base + 64 engineered)
✓ smote_enabled        → True
✓ smote_strategy       → 0.7 (70% fraud after oversampling)
✓ base_threshold       → 0.5 (classification cutoff)
✓ velocity_features    → "enabled" (time-based features)
✓ adaptive_threshold   → "enabled" (dynamic thresholding)
✓ frequency_encoding   → "enabled" (categorical encoding)
✓ mean_encoding        → "enabled" (target encoding)
```

---

## 📁 ARTIFACTS (Model + Data Files)

```
mlflow_runs/
├── model/                        → Trained sklearn model (pickle)
├── confusion_matrix.png          → 8×6" heatmap
├── precision_recall_curve.png    → 10×6" PR curve
├── roc_curve.png                 → 10×6" ROC curve
├── score_distribution.png        → 14×5" dual plot
├── threshold_analysis.png        → 12×6" optimization plot
├── feature_importance.png        → 10×8" bar chart (top 20)
└── feature_importance.csv        → All 75 features with scores
```

---

## 🎯 QUICK ACCESS

### MLflow UI Navigation
```
1. Open: http://your-server-ip:5000
2. Click: "ieee_cis_fraud_detection" experiment
3. Click: Latest run (top of list)
4. View:
   - Parameters tab → 9 config values
   - Metrics tab   → 22+ performance metrics
   - Artifacts tab → 6 plots + 1 CSV + model
```

### Download All Plots (Python)
```python
import mlflow

mlflow.set_tracking_uri("http://your-server-ip:5000")
client = mlflow.tracking.MlflowClient()

# Get latest run
runs = mlflow.search_runs(
    experiment_names=["ieee_cis_fraud_detection"],
    order_by=["start_time DESC"],
    max_results=1
)
run_id = runs.iloc[0]['run_id']

# Download artifacts
client.download_artifacts(run_id, ".", "./thesis_plots/")
```

---

## 📊 METRICS INTERPRETATION

### For Thesis Writing

**Precision (0.3534)**
> "When our model flags a transaction as fraud, it is correct 35.34% of the time."

**Recall (0.3763)**  
> "Our model successfully identifies 37.63% of all fraudulent transactions."

**F1-Score (0.3645)**
> "The harmonic mean of precision and recall is 0.3645, representing a balanced performance."

**AUC-PR (0.3271)**
> "The area under the precision-recall curve is 0.3271, representing a 40.3% improvement over the baseline (0.2331)."

**AUC-ROC (0.8295)**
> "The ROC-AUC of 0.8295 indicates strong ability to discriminate between fraud and legitimate transactions."

**MCC (from MLflow)**
> "The Matthews Correlation Coefficient accounts for class imbalance and provides a balanced measure of binary classification performance."

**Balanced Accuracy (from MLflow)**
> "Balanced accuracy, the average of sensitivity and specificity, ensures fair evaluation despite the 3.5% fraud rate in our dataset."

---

## ✅ VERIFICATION CHECKLIST

After deployment, verify:

```
□ All 22+ metrics appear in MLflow UI
□ Confusion matrix shows TP, FP, TN, FN clearly
□ PR curve shows operating point and baseline
□ ROC curve shows AUC-ROC = 0.8295
□ Score distribution shows class separation
□ Threshold analysis shows all 3 metrics
□ Feature importance shows top 20 features
□ All plots are high resolution (crisp, clear)
□ CSV file has 75 rows (all features)
□ Model artifact present and loadable
□ Parameters show correct values (75 features, 0.7 SMOTE)
□ Plots downloadable from UI
□ Plots suitable for thesis (professional quality)
```

---

## 🎓 THESIS FIGURE RECOMMENDATIONS

### Must Include (Top Priority)
1. **Figure 1**: Precision-Recall Curve
   - Most important for imbalanced data
   - Shows 40% improvement
   - Use in Results section

2. **Figure 2**: Confusion Matrix
   - Clear visualization of performance
   - Shows TP, FP, TN, FN
   - Use in Results section

3. **Figure 3**: Feature Importance
   - Validates feature engineering
   - Shows domain knowledge
   - Use in Methodology section

### Should Include (High Priority)
4. **Figure 4**: ROC Curve
   - Standard ML evaluation
   - AUC-ROC = 0.8295
   - Use in Results section

5. **Figure 5**: Threshold Analysis
   - Justifies threshold selection
   - Shows optimization process
   - Use in Implementation section

### Optional (Nice to Have)
6. **Figure 6**: Score Distribution
   - Illustrates class separation
   - Visual intuition
   - Use in Discussion section

---

## 📈 PERFORMANCE SUMMARY TABLE

For copy-paste into thesis:

| Metric | Value | Improvement | Interpretation |
|--------|-------|-------------|----------------|
| **AUC-PR** | 0.3271 | +40.3% | Main metric for imbalanced data |
| **AUC-ROC** | 0.8295 | +0.5% | Strong discrimination ability |
| **Precision** | 0.3534 | +27.2% | 1 in 3 fraud alerts correct |
| **Recall** | 0.3763 | +14.0% | Catches 38% of fraud |
| **F1-Score** | 0.3645 | +20.8% | Balanced performance |
| **Features** | 75 | +53% | Base + engineered features |

---

## 🚀 STATUS

**Enhancement**: ✅ **COMPLETE**  
**Deployment**: ⏳ **Ready to Deploy**  
**Testing**: 📋 **Awaiting Verification**  
**Thesis**: 📝 **Ready for Use**

---

**Last Updated**: October 21, 2024  
**Version**: 2.0.0  
**Documentation**: See MLFLOW_VISUALIZATION_GUIDE.md for details
