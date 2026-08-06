---
name: time-series-walk-forward-validation
description: |
  Use when working on time series forecasting, sequence prediction, or any task with temporal data (weekly/daily orders, retail SKU recommendation, demand forecasting, user churn prediction). Triggers when you need to set up cross-validation, evaluate a temporal model, or audit temporal leakage. Always use instead of K-fold on time-indexed data.
---

# Time Series Walk-Forward Validation

## Context
Random K-fold cross-validation on time series data **silently destroys validity**. A model can score 0.85 on K-fold and 0.65 on live data, wasting weeks of effort. This skill encodes the validation discipline proven in a retail SKU recommendation project (F1 76.5%, walk-forward stable ±0.2pp over 4 weeks).

The core lesson: **time goes one direction, and your validation must respect that**. Any information from the future — even an "innocent" aggregate like mean purchase frequency — can leak and inflate your CV score.

## Guidance

### Step 1: Use Walk-Forward, Never K-fold

```python
# ❌ WRONG: K-fold on time series (future leaks into train)
from sklearn.model_selection import KFold
for train_idx, val_idx in KFold(5).split(X):
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[val_idx], y[val_idx])
    # X[val_idx] is randomly scattered through time!
    # Train contains rows AFTER validation rows.

# ✅ RIGHT: Walk-forward (each fold uses only past data)
train_weeks = [16, 17, 18, 19]
val_weeks = [20]
model.fit(X[train_weeks], y[train_weeks])
score = model.score(X[val_weeks], y[val_weeks])

# Slide forward
train_weeks = [16, 17, 18, 19, 20]
val_weeks = [21]
# ... at least 4 windows for variance estimate
```

### Step 2: Pre-Train Data Leakage 8-Item Checklist

Before ANY model training, verify:

```
[ ] Training time range strictly < Evaluation time range
[ ] All aggregate features (co-occurrence, statistics, embeddings)
    computed ONLY on train period
[ ] Candidate set generation rule matches what you'll have at prediction time
[ ] N-formula / threshold / quantile inputs contain NO label information
[ ] Eval set customer set ⊆ train-seen customers
    (cold-start customers evaluated separately)
[ ] No "improvement" can be explained as "future information leaked"
[ ] Walk-forward split has ≥4 evaluation windows (for variance estimate)
[ ] Customer/segment populations are stable across windows
    (no massive churn mid-evaluation)
```

### Step 3: Compute Theoretical Upper Bound

You cannot know if you're done without knowing the ceiling.

```python
def compute_f1_em_upper_bound(actual_sets, true_sets, n_range):
    """F1-EM: oracle ceiling where N is per-customer optimal
    (unusable in production but tells you the ceiling)"""
    best_f1 = 0
    for n in n_range:
        # For each customer, compute F1 with n recommendations
        f1_at_n = mean_f1_at_n(actual_sets, true_sets, n=n)
        if f1_at_n > best_f1:
            best_f1 = f1_at_n
    return best_f1

# Also compute: Historical coverage (what % of next-week purchases
# appear in train history?)
historical_coverage = compute_coverage(history, next_week_purchases)

# Now you have 3 numbers:
# - F1-EM (oracle ceiling): 90.7%
# - Historical coverage: 87.3%  ← the "data ceiling"
# - Your actual F1: 76.5%
# - Gap to historical coverage tells you how much room is left
```

### Step 4: Stability Check Across Windows

A single walk-forward window can be lucky. Use ≥4 windows.

```
W20: 76.1%
W21: 76.3%
W22: 76.0%
W23: 76.5%
Stability: ±0.2pp ✅ (model is stable)
```

If stability is ±5pp or worse → overfitting, reduce features or regularize.

## Why This Matters

| Practice | Risk | Real Cost |
|---|---|---|
| K-fold on time series | Future info leaks into train | F1 inflated 5-15pp, deployed model breaks |
| Skip leakage checklist | Aggregate features include target week | "Magic" 5pp boost in CV, 0pp in production |
| No upper bound analysis | Optimize forever without direction | Wasted weeks past saturation |
| Single window evaluation | Lucky/unlucky fold misleads | Wrong conclusions on model quality |

The retail SKU project: **without walk-forward validation, we couldn't trust our 76.5% F1**. K-fold would have shown ~85%, which would have been a comfortable lie. The 4-week walk-forward ±0.2pp stability gave us deployment confidence.

## When to Apply

### When to Use
- Any time series forecasting (sales, demand, traffic, energy)
- Sequence prediction (next-purchase, next-event, next-page)
- Retail SKU recommendation (weekly/daily)
- Anomaly detection with temporal data
- ANY temporal ML task where future is correlated with past

### When NOT to Use
- Static tabular data (no time component) — K-fold is fine
- Image classification (no temporal order)
- NLP without temporal context (sentence-level tasks)
- Tasks where train/test split is random by design (e.g., cross-sectional survey analysis)

## Notes
- **Multi-window evaluation is mandatory**: single walk-forward window can be lucky; 4+ windows give variance
- **Stability check**: ±0.2pp = stable; ±5pp = overfitting; investigate the gap
- **Train window size matters**: too short (4 weeks) misses patterns; too long (100 weeks) overfits old patterns. 13-22 weeks is the sweet spot for weekly retail data
- **Combine with adversarial validation**: AUC≈0.50 means train/eval are from same distribution (good); AUC>0.70 means distribution drift, revisit feature engineering
- See also: `ts-forecasting-stale-lag-methodology`, `ts-lag-out-of-sample-trap`, `adversarial-validation-implementation`

## References
- Source methodology: industry-standard time series validation (Hyndman, Athanasopoulos "Forecasting: Principles and Practice")
- Related Kaggle competitions: Store Sales, M5 Forecasting — both use walk-forward
- Pairing skill: `feature-engineering-saturation-detection` (uses upper bound as saturation criterion)