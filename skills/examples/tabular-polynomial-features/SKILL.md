---
name: tabular-polynomial-features-breakthrough
description: |
  Polynomial features breakthrough for small tabular datasets. Use when: (1) Baseline
  model performance plateaus after hyperparameter tuning, (2) Dataset has <10K samples
  with <50 features, (3) Model optimization shows diminishing returns (<0.005 improvement),
  (4) Tree-based models (XGBoost/LightGBM) used but still underfitting. Covers degree-2
  expansion capturing feature interactions that tree models miss, verified on ISEC 2026
  competition (+0.017 LB improvement).
---

# Tabular Polynomial Features Breakthrough

## Problem

After exhausting standard optimization techniques (hyperparameter tuning, ensembling),
small tabular datasets often hit a performance plateau. Conventional wisdom suggests
tree-based models like XGBoost should capture non-linear patterns natively, but this
isn't always true for **feature interactions**.

## Context / Trigger Conditions

**Use this skill when**:
- **Small dataset**: <10K samples, <50 original features
- **Baseline plateau**: After hyperparameter tuning, improvements are <0.005
- **Tree models underperform**: XGBoost/LightGBM scores lower than expected
- **Feature interactions suspected**: Domain knowledge suggests features combine non-linearly
- **Model optimization exhausted**: Grid search, Optuna yield minimal gains

**Red flags that polynomial features may help**:
```
Baseline XGBoost:           F1 = 0.687, LB = 0.805
Grid Search (depth=8):      F1 = 0.686, LB = 0.810  (+0.005) ← Diminishing returns
Ensemble (3 models):        F1 = 0.688, LB = 0.815  (+0.010) ← Minimal gain
Feature Selection (Top 10): F1 = 0.692, LB = 0.819  (+0.014) ← But high gap!
```

**What this solves**:
- Captures **explicit feature interactions** (x₁×x₂, x₁×x₃, etc.)
- Provides **non-linear decision boundaries** without complex model architectures
- Outperforms model tuning on small datasets where data > model complexity

## Solution

### Step 1: Standardize Features First

**Critical**: Polynomial features are scale-sensitive. Always standardize before expansion.

```python
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import pandas as pd

# Load data
X_train = train_df.drop(columns=['target', 'id'])
X_test = test_df.drop(columns=['id'])
y_train = train_df['target']

print(f"Original features: {X_train.shape[1]}")  # e.g., 16

# Step 1: Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Step 2: Generate Degree-2 Polynomial Features

**Why degree=2?**
- Degree 1: Linear (no improvement over baseline)
- **Degree 2: Captures pairwise interactions** ← Sweet spot
- Degree 3+: Often overfits on small datasets

```python
# Step 2: Polynomial expansion (degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

print(f"Polynomial features: {X_train_poly.shape[1]}")
# Example: 16 → 152 features (9.5x expansion)
```

**Feature expansion formula**:
```
Original features: n
Degree-2 features: n + n(n-1)/2 = n(n+1)/2

For n=16:
  Linear terms: 16 (original features)
  Quadratic terms: 16 (x₁², x₂², ..., x₁₆²)
  Interaction terms: 120 (x₁×x₂, x₁×x₃, ..., x₁₅×x₁₆)
  Total: 16 + 16 + 120 = 152
```

### Step 3: Train with Conservative Parameters

**Reduce model complexity** to prevent overfitting on expanded feature space:

```python
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Use simpler parameters than baseline
model = XGBClassifier(
    n_estimators=50,      # ↓ from 100 (fewer trees)
    max_depth=4,          # ↓ from 6 (shallower trees)
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1
)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train_poly, y_train, cv=cv, scoring='f1')
print(f"CV F1: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Step 4: Monitor Public-Private Gap

**Warning sign**: Large Public-Private gap indicates overfitting.

```python
# After submission
print("Results:")
print(f"CV F1:       {cv_score:.4f}")
print(f"Public LB:   {public_lb:.4f}")
print(f"Private LB:  {private_lb:.4f}")
print(f"Pub-Priv Gap: {public_lb - private_lb:.4f}")

# Good: Gap < 0.010
# Warning: Gap > 0.020 (overfitting risk)
```

## Why This Works

### Tree Models Miss Some Interactions

**Common myth**: "XGBoost captures all non-linear patterns"

**Reality**:
```python
# XGBoost splits on individual features:
if loc > 100:
    if nosi > 3:
        predict_fault()

# Polynomial features explicitly capture interactions:
# loc × nosi = 100 × 3 = 300
# This combined threshold may be more predictive!
```

Tree models create **axis-aligned** decision boundaries. Polynomial features create
**diagonal/curved** boundaries that trees can only approximate with many splits.

### Feature Space Expansion > Model Complexity

On small datasets, expanding the feature space works better than complex models:

| Approach | ISEC 2026 Result | Improvement |
|----------|------------------|-------------|
| Baseline (16 features) | LB 0.805 | - |
| Grid Search (tuned depth=8) | LB 0.810 | +0.005 |
| Voting Ensemble | LB 0.815 | +0.010 |
| **Polynomial (152 features)** | **LB 0.822** | **+0.017** ✅ |

**Why**:
- More features = more expressive power without deeper trees
- Simpler models on expanded features < Complex models on raw features
- Reduced risk of overfitting compared to deep trees

### Regularization Effect of Feature Expansion

Polynomial expansion has a built-in regularization effect:

```python
# Original: model learns coefficients for 16 features
# weights = [w₁, w₂, ..., w₁₆]

# Polynomial: model learns coefficients for 152 features
# weights = [w₁, w₂, ..., w₁₆, w₁², w₂², ..., w₁₅×w₁₆]

# Each weight carries less information → distributed learning
# → Smoother decision boundary → Better generalization
```

## Verification

**Success indicators**:
1. **CV score improves** (not just LB): CV F1 0.687 → 0.710 (+0.023)
2. **Public-Private gap remains small**: 0.822 - 0.812 = 0.010 ✅
3. **Prediction distribution stable**: Class balance preserved in test set

**Example output from ISEC 2026**:
```python
# Baseline
CV F1: 0.6867 (+/- 0.0154)
Public LB: 0.80500
Private LB: 0.80121
Gap: 0.00379

# Polynomial Features
CV F1: 0.7097 (+/- 0.0183)  ← +0.023 CV improvement!
Public LB: 0.82242          ← +0.017 LB improvement!
Private LB: 0.81238
Gap: 0.01004               ← Still acceptable (<0.015)
```

**Check for overfitting**:
```python
# Bad: Feature selection increased gap
# CV F1: 0.6915
# Public LB: 0.81947
# Private LB: 0.78634
# Gap: 0.03313  ← 8x worse than baseline!

# Root cause: Top-10 features overfit to training patterns
# Lesson: More features (152) < Fewer "selected" features (10)
```

## Example

**Complete workflow** (ISEC 2026 competition):

```python
#!/usr/bin/env python3
"""
ISEC Data Challenge 2026 - Polynomial Features
Software Defect Prediction with Static Code Metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Load data
DATA_DIR = Path('/path/to/data')
train_df = pd.read_excel(DATA_DIR / 'train_Fault.xlsx')
test_df = pd.read_excel(DATA_DIR / 'test.xlsx')

# Prepare features
drop_cols = ['Issue-id', 'Generated Postmortem', 'Fault']
X_train = train_df.drop(columns=drop_cols)
y_train = train_df['Fault']
X_test = test_df.drop(columns=[col for col in drop_cols if col in test_df.columns])

print(f"🔧 Baseline Features: {X_train.shape[1]}")

# Step 1: Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Polynomial Features (degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

print(f"✅ Polynomial Features: {X_train_poly.shape[1]}")
# Output: 16 → 152 features

# Step 3: Train model (conservative parameters)
model = XGBClassifier(
    n_estimators=50,      # Reduced from 100
    max_depth=4,          # Reduced from 6
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1
)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train_poly, y_train, cv=cv, scoring='f1', n_jobs=-1)
print(f"CV F1-Score: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Step 4: Train and predict
model.fit(X_train_poly, y_train)
y_pred = model.predict(X_test_poly)

# Create submission
submission = pd.DataFrame({
    'Issue-id': test_df['Issue-id'],
    'Fault': y_pred
})
submission.to_csv('submission_polynomial.csv', index=False)

print(f"✅ Prediction distribution: {(y_pred == 1).sum()} / {(y_pred == 0).sum()}")
```

**Results**:
```
🔧 Baseline Features: 16
✅ Polynomial Features: 152
CV F1-Score: 0.7097 (+/- 0.0183)
✅ Prediction distribution: 1019 / 980

Leaderboard:
  Public:  0.82242  (+0.017 vs baseline)
  Private: 0.81238  (+0.011 vs baseline)
  Gap:     0.01004  (acceptable)
```

## Notes

### When Polynomial Features Work Best

✅ **Good scenarios**:
- **Static code metrics** (ISEC 2026): Complexity × Size interactions matter
- **Financial data**: Price × Volume, Risk × Exposure
- **Medical data**: Age × Biomarker, Dose × Weight
- **Survey data**: Demographic × Response interactions
- **Any tabular data** with <50 features and suspected feature interactions

✅ **Why software defect prediction worked**:
```python
# Defects arise from interactions:
loc * nosi              # Large code with deep nesting = high complexity
totalMethods * totalFields  # Many methods + many fields = complex class
loopQty * comparisonsQty     # Many loops + many comparisons = high cyclomatic

# Individual metrics less predictive:
loc alone              # Correlation: 0.013
nosi alone             # Correlation: 0.154
loc × nosi             # Combined effect > sum of parts!
```

❌ **Avoid when**:
- **High-dimensional data** (>50 features): Explosion creates too many features
- **Sparse data**: Interactions amplify noise
- **Interpretability critical**: Polynomial terms hard to explain
- **Very small datasets** (<500 samples): Risk of overfitting

### Alternative Approaches

**If polynomial features overfit**:
1. **Feature selection after expansion**: Use SelectKBest on polynomial features
2. **Regularization**: Increase L1/L2 penalty in model
3. **Degree=1.5**: Use custom feature engineering (only select interactions)
4. **Tree-based models**: Try CatBoost (handles categorical better)

**If polynomial features don't help**:
- Check if **linear relationship dominates**: Polynomial adds noise
- Try **target encoding** for categorical features
- Consider **deep learning** (TabNet, TabPFN) for very small datasets
- Use **domain knowledge** to create specific interactions, not all

### Comparison: Model Tuning vs Feature Engineering

**ISEC 2026 empirical comparison**:

| Method | Features | Params | CV F1 | Public | Private | Gap |
|--------|----------|--------|-------|--------|---------|-----|
| Baseline | 16 | Default | 0.687 | 0.805 | 0.801 | 0.004 |
| Grid Search | 16 | Tuned | 0.686 | 0.810 | 0.790 | 0.020 |
| Stacking | 16 | 3 models | 0.717 | 0.792 | 0.782 | 0.010 |
| **Polynomial** | **152** | **Simple** | **0.710** | **0.822** | **0.812** | **0.010** |
| RFE Top-10 | 10 | Default | 0.692 | 0.819 | 0.786 | 0.033 |

**Key insights**:
1. **Stacking CV ≠ LB**: CV 0.717 (highest!) but LB 0.792 (lowest)
2. **Feature selection paradox**: Top-10 features raised Public but hurt Private
3. **Polynomial wins**: Better LB than tuning, with lower overfitting risk

### CV-LB Mismatch Warning

**Critical**: High CV doesn't guarantee high LB!

```
Stacking Ensemble:
  CV F1:  0.717  ← Highest CV score!
  LB:     0.792  ← But LB score is low
  Why:    Meta-learner overfits to OOF predictions

Polynomial Features:
  CV F1:  0.710  ← Lower CV
  LB:     0822  ← But LB score is highest!
  Why:    Better generalization to test set
```

**Always validate on LB, not just CV!**

### Feature Selection Danger

**RFE (Recursive Feature Elimination) can hurt**:

```python
# ISEC 2026: RFE Top-10 features
selector = RFE(XGBClassifier(), n_features_to_select=10)
X_train_selected = selector.fit_transform(X_train, y_train)

# Result: Public ↑ but Private ↓↓
Public:  0.819  (+0.014)
Private: 0.786  (-0.015)
Gap:     0.033  (8x worse than baseline!)

# Problem: Top-10 features overfit to training patterns
# Lesson: On small datasets, MORE features (with structure) < FEWER "selected" features
```

**Why**:
- RFE selects features that perform well on **training CV**
- Test set may have different feature importance distribution
- Polynomial expansion distributes importance → more robust

## References

**Competition Verification**:
- [ISEC Data Challenge 2026](https://www.kaggle.com/competitions/isec-data-challenge-2026) - Software Defect Prediction, verified improvement from 0.805 → 0.822 (+0.017) using degree-2 polynomial features on 16 static code metrics

**Best Practice Sources**:
- [Kaggle: Best Practices for Small Datasets](https://www.kaggle.com/questions-and-answers/575548) - Recommends repeated K-Fold CV for stability
- [Kaggle: Winning Strategies for Multi-Class Classification](https://www.kaggle.com/general/583830) - Mentions polynomial features as part of successful strategies
- [Kaggle: Advanced Feature Engineering Techniques](https://www.kaggle.com/questions-and-answers/566066) - Feature engineering with gradient boosting models

**Technical References**:
- [scikit-learn: PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) - Official documentation
- [Nature: TabPFN - Tabular Foundation Models](https://www.nature.com/articles/s41586-024-08328-6) - State-of-the-art for very small tabular datasets (<1000 rows)

**Related Skills**:
- [small-dataset-optimization-limits](../small-dataset-optimization-limits/): When to stop optimization on small datasets (Re-ID focus)
- [kaggle-auc-binary-submission-bug](../kaggle-auc-binary-submission-bug/): CV-LB gap debugging (AUC metric focus)
- [model-ensemble](../model-ensemble/): Ensemble pitfalls (correlation trap)

**See Also**:
- [smote-data-augmentation-classification](../smote-data-augmentation-classification/): For imbalanced datasets (not needed for balanced data like ISEC 2026)
- [adversarial-validation-implementation](../adversarial-validation-implementation/): Check train/test distribution before feature engineering
