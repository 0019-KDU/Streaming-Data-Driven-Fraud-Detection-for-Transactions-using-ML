# MLflow Visualization and Metrics Guide

## Overview
This document describes the comprehensive MLflow tracking capabilities added to the fraud detection training pipeline for thesis documentation and model analysis.

**Last Updated**: October 21, 2024  
**Training File**: `src/dags/ieee_cis_training.py`  
**MLflow Version**: 2.x+

---

## 📊 What's Been Added

### 1. Core Metrics (5)
Standard classification metrics for fraud detection:

| Metric | Description | Best Value | Latest Result |
|--------|-------------|------------|---------------|
| **AUC-PR** | Area Under Precision-Recall Curve | 1.0 | 0.3271 |
| **AUC-ROC** | Area Under ROC Curve | 1.0 | 0.8295 |
| **Precision** | True Fraud / (True Fraud + False Fraud) | 1.0 | 0.3534 |
| **Recall** | True Fraud / Total Fraud | 1.0 | 0.3763 |
| **F1-Score** | Harmonic mean of Precision & Recall | 1.0 | 0.3645 |

### 2. Additional Metrics (3)
Advanced metrics specifically for imbalanced datasets:

| Metric | Description | Interpretation | Use Case |
|--------|-------------|----------------|----------|
| **Balanced Accuracy** | Average of sensitivity and specificity | 0.5 = random, 1.0 = perfect | Accounts for class imbalance |
| **Matthews Correlation Coefficient (MCC)** | Correlation between predictions and reality | -1 to +1, 0 = random | Best for imbalanced datasets |
| **Cohen's Kappa** | Agreement adjusted for chance | 0 = random, 1 = perfect | Inter-rater reliability |

### 3. Confusion Matrix Breakdown (8)
Detailed breakdown of prediction outcomes:

| Metric | Formula | Meaning |
|--------|---------|---------|
| True Positives (TP) | Count | Correctly identified fraud |
| False Positives (FP) | Count | Legit flagged as fraud |
| True Negatives (TN) | Count | Correctly identified legit |
| False Negatives (FN) | Count | Fraud missed |
| True Positive Rate | TP / (TP + FN) | Recall (sensitivity) |
| False Positive Rate | FP / (FP + TN) | Type I error rate |
| True Negative Rate | TN / (TN + FP) | Specificity |
| False Negative Rate | FN / (FN + TP) | Type II error rate |

### 4. Per-Class Metrics (6)
Separate performance metrics for each class:

**Legit Transactions (Class 0):**
- `legit_precision`: Precision for legitimate transactions
- `legit_recall`: Recall for legitimate transactions  
- `legit_f1`: F1-score for legitimate transactions

**Fraud Transactions (Class 1):**
- `fraud_precision`: Precision for fraud transactions
- `fraud_recall`: Recall for fraud transactions
- `fraud_f1`: F1-score for fraud transactions

### 5. Training Parameters (9)
Configuration parameters logged for reproducibility:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model_type` | "EnhancedEnsemble" | Ensemble of XGBoost, LightGBM, CatBoost |
| `n_features` | 75 | Total features used |
| `smote_enabled` | True | SMOTE oversampling enabled |
| `smote_strategy` | 0.7 | 70% fraud representation after SMOTE |
| `base_threshold` | 0.5 (default) | Classification threshold |
| `velocity_features` | "enabled" | Time-based features |
| `adaptive_threshold` | "enabled" | Dynamic threshold |
| `frequency_encoding` | "enabled" | Categorical encoding |
| `mean_encoding` | "enabled" | Target encoding |

---

## 📈 Visualizations

### 1. Confusion Matrix Heatmap
**Filename**: `confusion_matrix.png`  
**Size**: 8×6 inches  
**Description**: Heatmap showing TP, FP, TN, FN with annotations

**Key Features:**
- Color-coded intensity (Blues colormap)
- Actual vs Predicted labels
- Threshold value in title
- Exact counts annotated in cells

**Interpretation:**
- Diagonal cells (top-left, bottom-right) = correct predictions
- Off-diagonal cells = errors (FP, FN)
- Darker blue = higher count

---

### 2. Precision-Recall Curve
**Filename**: `precision_recall_curve.png`  
**Size**: 10×6 inches  
**Description**: PR curve showing precision vs recall tradeoff

**Key Features:**
- Blue curve with AUC-PR score
- Red dashed line showing baseline (fraud rate)
- Red dot marking operating point (selected threshold)
- Grid for easy reading

**Interpretation:**
- Higher curve = better model
- Curve above baseline = better than random
- Operating point shows current precision/recall
- Top-right corner = ideal (high precision, high recall)

**Why Important for Imbalanced Data:**
- More informative than ROC for rare events
- Focuses on positive class (fraud)
- Sensitive to changes in precision

---

### 3. ROC Curve
**Filename**: `roc_curve.png`  
**Size**: 10×6 inches  
**Description**: ROC curve showing TPR vs FPR tradeoff

**Key Features:**
- Blue curve with AUC-ROC score
- Red diagonal = random classifier
- Red dot = operating point
- Grid background

**Interpretation:**
- Higher curve = better separation
- AUC-ROC 0.5 = random, 1.0 = perfect
- Top-left corner = ideal (high TPR, low FPR)
- Operating point shows current TPR/FPR

**Use Cases:**
- Compare models
- Assess overall discrimination ability
- Understand tradeoff between TPR and FPR

---

### 4. Score Distribution
**Filename**: `score_distribution.png`  
**Size**: 14×5 inches (2 subplots)  
**Description**: Two views of prediction score distributions

**Left Plot - Histogram:**
- Blue histogram = legit transaction scores
- Red histogram = fraud transaction scores
- Green dashed line = threshold
- Density normalized for comparison

**Right Plot - Box Plot:**
- Side-by-side box plots by class
- Shows median, quartiles, outliers
- Green threshold line

**Interpretation:**
- Good separation = minimal overlap between distributions
- Threshold position affects precision/recall balance
- Outliers indicate edge cases

---

### 5. Threshold Analysis
**Filename**: `threshold_analysis.png`  
**Size**: 12×6 inches  
**Description**: How metrics change with different thresholds

**Key Features:**
- Blue line = Precision
- Red line = Recall  
- Green line = F1-Score
- Purple dashed line = selected threshold
- Tests 100 thresholds from 0.01 to 0.99

**Interpretation:**
- Precision increases with threshold (stricter)
- Recall decreases with threshold (miss more fraud)
- F1 peak shows optimal balance
- Use for threshold tuning

**Business Application:**
- Low threshold = catch more fraud (high recall), more false alarms
- High threshold = fewer false alarms (high precision), miss some fraud
- Choose based on business cost of FP vs FN

---

### 6. Feature Importance
**Filename**: `feature_importance.png`  
**Size**: 10×8 inches  
**Description**: Top 20 most important features

**Key Features:**
- Horizontal bar chart (easy to read feature names)
- Viridis colormap (perceptually uniform)
- Sorted by importance (descending)
- Shows only top 20 for clarity

**Interpretation:**
- Longer bars = more important features
- Identifies key fraud signals
- Validates feature engineering

**Additional Artifact:**
- Full feature importance CSV saved: `feature_importance.csv`
- Contains all 75 features with scores

---

## 🎯 How to Use for Thesis

### 1. Model Comparison
```python
# MLflow UI allows comparing runs side-by-side
# Navigate to: http://localhost:5000
# Select multiple runs → Compare
```

**What to Compare:**
- AUC-PR trends across experiments
- Effect of SMOTE strategy on recall
- Impact of hyperparameters on precision
- Feature importance consistency

### 2. Visualizations for Thesis

**Recommended Figures:**

1. **Figure 1**: Confusion Matrix
   - Shows model performance at a glance
   - Include in "Results" section

2. **Figure 2**: Precision-Recall Curve
   - Demonstrates handling of class imbalance
   - Compare before/after SMOTE
   - Include in "Methodology" or "Results"

3. **Figure 3**: Threshold Analysis
   - Shows decision-making process
   - Justifies threshold selection
   - Include in "Implementation" section

4. **Figure 4**: Feature Importance
   - Validates feature engineering
   - Shows domain knowledge application
   - Include in "Feature Engineering" section

5. **Figure 5**: Score Distribution
   - Illustrates class separation
   - Demonstrates model effectiveness
   - Include in "Results" section

### 3. Metrics Table for Thesis

**Table 1: Model Performance Metrics**

| Metric | Before Enhancement | After Enhancement | Improvement |
|--------|-------------------|-------------------|-------------|
| AUC-PR | 0.2331 | 0.3271 | +40.3% |
| AUC-ROC | 0.8257 | 0.8295 | +0.5% |
| Precision | 0.2778 | 0.3534 | +27.2% |
| Recall | 0.3300 | 0.3763 | +14.0% |
| F1-Score | 0.3018 | 0.3645 | +20.8% |
| MCC | [New] | [From MLflow] | - |
| Balanced Accuracy | [New] | [From MLflow] | - |

### 4. Accessing MLflow Data

**Via UI:**
```bash
# On server
cd ~/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/src
docker compose ps  # Check MLflow container
# Access: http://your-server-ip:5000
```

**Via Python API:**
```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Get experiment
experiment = mlflow.get_experiment_by_name("ieee_cis_fraud_detection")

# Get runs
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Download artifacts
for run_id in runs['run_id']:
    client = mlflow.tracking.MlflowClient()
    client.download_artifacts(run_id, "confusion_matrix.png", ".")
```

---

## 🔧 Technical Implementation

### Visualization Generation
All plots are generated using:
- **matplotlib** with 'Agg' backend (headless server)
- **seaborn** for statistical visualizations
- High-resolution PNG format (300 DPI)

### Error Handling
All visualization code wrapped in try-except:
```python
try:
    # Generate plots
    mlflow.log_figure(fig, "plot.png")
    logger.info("Successfully logged visualization")
except Exception as e:
    logger.warning(f"Failed to create visualization: {e}")
    # Training continues even if plots fail
```

### Memory Management
- Each plot explicitly closed: `plt.close(fig)`
- Prevents memory leaks in long-running Airflow tasks
- Uses temporary files for CSV artifacts

---

## 📝 Checklist for Next Training Run

- [ ] Verify MLflow UI accessible
- [ ] Check matplotlib/seaborn installed in Airflow container
- [ ] Ensure /tmp directory writable
- [ ] Trigger training DAG
- [ ] Monitor logs for visualization creation
- [ ] Verify all 6 plots appear in MLflow UI
- [ ] Check all 22+ metrics logged
- [ ] Download plots for thesis
- [ ] Compare with baseline run

---

## 🚀 Expected Output Structure

```
MLflow Run
├── Parameters (9)
│   ├── model_type: "EnhancedEnsemble"
│   ├── n_features: 75
│   ├── smote_strategy: 0.7
│   └── ...
├── Metrics (22+)
│   ├── auc_pr: 0.3271
│   ├── balanced_accuracy: [value]
│   ├── matthews_corrcoef: [value]
│   └── ...
├── Artifacts
│   ├── model/
│   │   └── [sklearn model]
│   ├── confusion_matrix.png
│   ├── precision_recall_curve.png
│   ├── roc_curve.png
│   ├── score_distribution.png
│   ├── threshold_analysis.png
│   ├── feature_importance.png
│   └── feature_importance.csv
└── Tags
    └── registered_model_name: "fraud_detection_model"
```

---

## 📚 References

1. **Imbalanced Learning**: Davis & Goadrich (2006) - "The Relationship Between Precision-Recall and ROC Curves"
2. **MCC for Imbalanced Data**: Chicco & Jurman (2020) - "The advantages of the Matthews correlation coefficient"
3. **MLflow Best Practices**: Official MLflow documentation
4. **Visualization Design**: Tufte (1983) - "The Visual Display of Quantitative Information"

---

## 🎓 Thesis Writing Tips

### Abstract/Summary
> "We implemented comprehensive MLflow tracking with 22+ metrics and 6 visualizations, achieving 40% improvement in AUC-PR (0.2331→0.3271) through balanced ensemble approach with SMOTE oversampling."

### Methodology Section
- Describe each visualization and its purpose
- Explain metric selection (why MCC for imbalanced data)
- Detail MLflow integration architecture

### Results Section
- Include before/after comparison table
- Present 2-3 key visualizations
- Discuss threshold selection process

### Discussion
- Interpret feature importance findings
- Analyze precision/recall tradeoff
- Compare with baseline models

---

**Status**: ✅ Ready for production use  
**Version**: 2.0 - Comprehensive MLflow Integration  
**Author**: Fraud Detection ML Pipeline Team
