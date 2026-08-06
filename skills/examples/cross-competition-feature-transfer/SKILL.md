---
name: cross-competition-feature-transfer
description: |
  Use when starting a new ML task that structurally resembles a known competition (recommendation ↔ retail, demand ↔ supply, churn ↔ fraud, segmentation ↔ classification). Triggers when your task feels "not novel" — there is likely a top solution you can borrow features from. Especially valuable at the start of a project before inventing custom features.
---

# Cross-Competition Feature Transfer

## Context
The most valuable features often come from **a different but structurally similar competition**, not from your own data. In a recent retail SKU recommendation project, the biggest single improvement (+12pp on F1) came from 6 features borrowed directly from the 2017 Instacart Market Basket Analysis top 2% solution. This skill teaches when and how to find these gold mines.

The core insight: **features are mathematical operations on data, and math is portable**. If two tasks have the same entities, same metric, and similar data structure, the features that worked on one will likely work on the other.

## Guidance

### Step 1: Identify Your Structural Twin

Ask: "What competition has the same (entities, task, metric)?"

| Your Task | Structural Twin | Why Match |
|---|---|---|
| Retail SKU recommendation (customer × SKU, weekly, F1) | Instacart 2017 | Identical |
| Demand forecasting (store × item, daily, RMSLE) | Store Sales / M5 | Identical |
| Store sales (store × family, daily, RMSLE) | Favorita 1st place | Identical |
| Churn prediction (user, binary, AUC) | IEEE-CIS Fraud | Same binary classification |
| Time series forecasting (item, RMSE) | M5 / M4 | Identical structure |
| CTR prediction (user × ad, binary, logloss) | Criteo / Avazu | Identical |

### Step 2: Mine the Top Solution

```python
# 1. Find top 5 solutions on the twin competition (1st, 2nd, 3rd, top 5%, top 10%)
# 2. Extract their feature lists — NOT the model, the FEATURES
# 3. Categorize features by "structural" vs "domain-specific"

# Example: Instacart top 2% features → retail SKU recommendation project
instacart_features = [
    "up_order_strike",      # Σ 1/2^reverse_idx — TIME-WEIGHTED HISTORY
    "reorder_ratio",         # pair_weeks / customer_weeks — LOYALTY SCORE
    "gap_mean", "gap_std",   # purchase interval stats — BEHAVIOR CONSISTENCY
    "trend_diff_3_8",        # roll_mean_3 - roll_mean_8 — TREND SHIFT
    "roll_min", "roll_max",  # rolling multi-stat — INTENSITY SHAPE
    "cat_purchase_ratio",    # customer × category — PREFERENCE STRUCTURE
]

# All 6 transferred successfully. Single biggest contributor (+12pp on F1).
```

### Step 3: Validate Transferability

Not all features transfer. Check 3 conditions:

```python
def is_feature_transferable(feature, my_task):
    # 1. Does my data have the required input columns?
    if not has_required_data(feature.required_columns):
        return False  # Can't compute it without the data

    # 2. Does the business meaning transfer?
    #    Instacart "user reorder" → "customer weekly repurchase"
    #    SAME meaning ✓ → safe transfer
    #    Instacart "product2vec over orders" → "embedding over weeks"
    #    PARTIAL meaning → proceed with caution, may need adaptation

    # 3. Is the metric aligned?
    #    Both use F1 → perfect alignment
    #    One uses F1, other uses RMSE → may not transfer
    return True
```

### Step 4: A/B Test, Don't Blindly Trust

```python
# Add 1 feature at a time, measure on walk-forward
base_features = [...]  # existing feature set
walk_forward_weeks = [20, 21, 22, 23]

for feat in instacart_features:
    candidate = base_features + [feat]
    cv_score = walk_forward_eval(candidate, walk_forward_weeks)
    print(f"+ {feat}: F1 = {cv_score:.4f} (delta {cv_score - baseline:.4f})")

# Only keep features that improve CV by ≥0.2pp
# Drop features that hurt CV (often redundancy with existing features)
```

### Step 5: Check for Redundancy

```python
# Before adding the transferred feature, check correlation
import pandas as pd
for feat in instacart_features:
    corr = train_df[base_features + [feat]].corr()[feat].abs()
    high_corr = corr[corr > 0.7].drop(feat, errors='ignore')
    if len(high_corr) > 0:
        print(f"⚠️ {feat} highly correlated with: {high_corr.index.tolist()}")
        # Likely redundant, skip or investigate further
```

## Why This Matters

| Strategy | F1 Contribution | Time Spent | ROI |
|---|---|---|---|
| Invent features from scratch | 0-2pp | weeks | low |
| **Transfer from similar competition** | **+12pp** | **days** | **high** |
| Hyperparameter tuning | 0-0.5pp | days | low |
| Ensemble tricks | 0-1pp | days | medium |

**Cross-competition transfer had 10x the ROI of any other technique** in the retail SKU project. The Instacart top 2% solution was published in 2017; reused years later because the structure hadn't changed. The math doesn't expire.

## When to Apply

### When to Use
- Your task has a clear structural twin (recommendation, forecasting, churn)
- Top 5% solutions exist on the twin competition with published feature lists
- You have ≥13 weeks of historical data (to compute lag features)
- The metric is identical (F1, RMSLE, AUC, NDCG@K, logloss)
- You are early in the project (before committing to custom feature engineering)

### When NOT to Use
- Novel task with no twin competition
- Highly specialized domain (medical imaging, 3D reconstruction, molecular property prediction)
- Data scale is wildly different (10x rows difference → features behave differently)
- You have <4 weeks of history (can't compute lag features properly)
- Twin competition top solution is heavily domain-specific (e.g., requires medical codes)

## Notes
- **Always check correlation with existing features** — Instacart's `gap_mean` may overlap with your `pair_freq`; redundant features hurt
- **Transfer features, not models** — model architecture (LGB vs NN) may differ across competitions, but feature math is portable
- **Document the source** — link to the original Kaggle writeup in your skill description for future reference
- **Beware of "too good" features** — if a transferred feature gives +10pp instantly, double-check it doesn't leak
- **Top 2% > Top 1% for transfer** — top 1% often uses competition-specific tricks; top 2-5% has cleaner, more transferable patterns

## References
- Source competition: [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis) top 2% solutions
- Other transferable sources: Favorita 1st place, Store Sales, M5 Forecasting, Criteo top solutions
- Pairing skill: `time-series-walk-forward-validation` (use walk-forward to validate transferred features)