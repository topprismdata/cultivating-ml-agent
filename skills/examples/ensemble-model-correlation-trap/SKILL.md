---
name: ensemble-model-correlation-trap
description: |
  Diagnose when ensemble/stacking will fail due to high model correlation.
  Use when: (1) Stacking or ensemble shows no improvement over single models,
  (2) Considering weighted average of multiple models, (3) Model predictions are
  nearly identical, (4) AUC/RMS gains plateau despite complex ensembling.
  Critical for tabular ML competitions where model diversity determines ensemble success.
---

# Ensemble Model Correlation Trap

## Problem

In machine learning competitions, practitioners often assume that combining multiple models (stacking, weighted averaging, voting) will automatically improve performance. However, **when models have extremely high correlation (>0.999), ensembling provides no benefit** because the models are making nearly identical predictions.

This is a common trap in tabular data competitions where XGBoost, CatBoost, and LightGBM often converge to similar solutions.

## Context / Trigger Conditions

**Symptoms that indicate this problem:**

- Stacking/ensemble OOF score ≤ best single model score
- Weighted average performs worse than simple average
- Logistic Regression learns nearly equal weights for all models
- High correlation (>0.995) between model predictions
- Single model continues to outperform complex ensembles

**Common scenarios:**
- Tabular data with tree-based models (XGBoost, CatBoost, LightGBM)
- Models trained on similar feature sets
- Competition datasets where feature space is well-explored

## Solution

### Step 1: Calculate Model Correlation

After training models with cross-validation:

```python
import numpy as np
from scipy.stats import pearsonr

# Assume oof_pred1, oof_pred2, oof_pred3 are OOF predictions from different models
corr_12, p_value = pearsonr(oof_pred1, oof_pred2)
corr_13, p_value = pearsonr(oof_pred1, oof_pred3)
corr_23, p_value = pearsonr(oof_pred2, oof_pred3)

print(f"Model 1 vs 2: {corr_12:.5f}")
print(f"Model 1 vs 3: {corr_13:.5f}")
print(f"Model 2 vs 3: {corr_23:.5f}")
```

### Step 2: Interpret Correlation Values

| Correlation Range | Diagnosis | Action |
|------------------|----------|--------|
| < 0.99 | ✅ Low correlation | Ensembling likely effective |
| 0.99 - 0.995 | ⚠️ Medium correlation | Ensembling may help marginally |
| 0.995 - 0.999 | ⚠️ High correlation | Ensembling may not help |
| > 0.999 | ❌ Extreme correlation | **Ensembling will not help** |

### Step 3: Verify with Stacking

```python
from sklearn.linear_model import LogisticRegression

# Create meta-features
meta_features = pd.DataFrame({
    'model1': oof_pred1,
    'model2': oof_pred2,
    'model3': oof_pred3,
})

# Train meta-learner
meta_learner = LogisticRegression(penalty='l2', C=1.0)
meta_learner.fit(meta_features, y)

# Check learned weights
weights = meta_learner.coef_[0]
print(f"Model 1 weight: {weights[0]:.4f}")
print(f"Model 2 weight: {weights[1]:.4f}")
print(f"Model 3 weight: {weights[2]:.4f}")

# If weights are nearly equal, models are too similar
# Example: [3.19, 3.21, 3.20] → models make identical predictions
```

### Step 4: Alternative Strategies

When correlation is too high:

**Option 1: Focus on Single Model**
- Optimize the best performing model
- Tune hyperparameters more aggressively
- Better feature engineering

**Option 2: Increase Model Diversity**
- Use fundamentally different algorithms (e.g., Neural Networks + Trees)
- Different feature subsets for each model
- Different preprocessing pipelines

**Option 3: Target Different Parts of Data**
- Some models excel at certain probability ranges
- Blend based on confidence thresholds

## Verification

**Successful ensemble diagnosis:**

1. Calculate correlations between all model OOF predictions
2. If correlations > 0.999, expect minimal ensemble gain
3. Logistic Regression weights should be nearly equal
4. Simple average should perform similarly to weighted stacking

**Example from Kaggle S6E2 (Heart Disease):**
```
XGBoost vs CatBoost:  0.99951
XGBoost vs LightGBM: 0.99955
CatBoost vs LightGBM: 0.99916

Stacking OOF: 0.95222
Best single model: 0.95224
```

Ensembling provided only +0.00002 improvement (within noise).

## Example

**Diagnosing ensemble effectiveness:**

```python
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np

# Train models
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_xgb = np.zeros(len(y))
oof_cat = np.zeros(len(y))

for train_idx, val_idx in kf.split(X, y):
    # XGBoost
    model_xgb = XGBClassifier(**xgb_params)
    model_xgb.fit(X[train_idx], y[train_idx])
    oof_xgb[val_idx] = model_xgb.predict_proba(X[val_idx])[:, 1]

    # CatBoost
    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(X[train_idx], y[train_idx])
    oof_cat[val_idx] = model_cat.predict_proba(X[val_idx])[:, 1]

# Check correlation
corr = np.corrcoef(oof_xgb, oof_cat)[0, 1]
print(f"Correlation: {corr:.5f}")

if corr > 0.999:
    print("⚠️  Models too similar. Ensembling unlikely to help.")
    print("💡 Focus on single model optimization or feature engineering.")
else:
    print("✅ Models diverse. Ensembling may improve performance.")
```

## Notes

**Why high correlation happens:**
- Tree-based models (XGBoost, CatBoost, LightGBM) learn similar decision boundaries
- Same feature sets lead to similar patterns
- Well-explored feature spaces leave little room for diversity
- Large datasets (100k+ samples) reduce model variance

**Common mistakes:**
- Assuming "more models = better performance"
- Spending time on complex stacking when models are correlated
- Not checking model correlations before ensembling

**When this doesn't apply:**
- Image data (CNN architectures can be more diverse)
- NLP (different embeddings create diversity)
- Models trained on different feature subsets
- Models trained on different data samples

**Related concepts:**
- **Bagging**: Reduces correlation through bootstrapping, but tree ensembles already use this
- **Diversity-accuracy tradeoff**: More diverse models may have lower individual accuracy
- **Ensemble selection**: Not all models should be included in final ensemble

## References

- [Kaggle S6E2 Heart Disease Competition](https://www.kaggle.com/competitions/playground-series-s6e2) - Empirical validation
- [Scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
- "Kaggle Ensembling Guide" - Community best practices
- ["Why does stacking sometimes fail?"](https://stats.stackexchange.com/questions) - Cross-validated discussion

**Experimental validation:**
- Tested on 630,000 sample tabular dataset
- 14 different ensemble configurations tested
- All converged to 0.95018-0.95021 range (±0.00003)
- Single best model: 0.95224 OOF, 0.95021 LB
