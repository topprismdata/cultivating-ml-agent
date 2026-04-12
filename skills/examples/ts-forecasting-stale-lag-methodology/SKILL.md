---
name: ts-forecasting-stale-lag-methodology
description: |
  Complete methodology for solving multi-step time series forecasting competitions where lag features
  cause systematic underprediction at test time. Documents the full journey from LB 1.859 → 0.399,
  including every failed approach and the key breakthrough. Use when: (1) Starting a new time series
  forecasting competition, (2) Model underpredicts at test time due to stale lag features,
  (3) CV-LB gap of 3-10x persists, (4) Need a systematic approach to diagnose and fix prediction
  magnitude issues. This is the MASTER methodology combining all ts-* skills.
---

# Multi-Step Time Series Forecasting: Stale Lag Methodology

## Context

This methodology was developed through the Kaggle Store Sales Time Series Forecasting competition.
The journey went from LB 1.859 to LB 0.399 — a **4.7x improvement** — through systematic
diagnosis and multiple breakthroughs.

## The Problem Pattern

In multi-step time series forecasting (predict N days ahead):
1. You build lag/rolling features (lag_1, rolling_mean_7, etc.)
2. CV score looks great (0.36)
3. LB score is 3-10x worse (1.86)
4. Predictions underpredict by 5-10x (mean=40 vs actual mean=467)

## Complete Journey with Failures and Successes

### Phase 1: Baseline (LB 1.859)

```
Approach: LightGBM with lag features, forward-fill for test
Result: CV=0.36, LB=1.859 (5x gap!)
Diagnosis: prediction mean = 40, training mean = 467
```

**Why it fails**: All test days get the same ffill'd lag values. The model sees constant
recent history and predicts conservatively (low).

### Phase 2: Failed Fixes (All Tried, All Failed)

| Attempt | Result | Why It Failed |
|---------|--------|---------------|
| Tweedie objective | LB=1.87 | Doesn't fix stale features |
| Remove short lags | LB=2.84 | Loses too much signal |
| Recursive prediction | mean=5.87, LB=2.89 | Error accumulates through lag features |
| TE-fill (replace ffill with TE) | mean=34 (WORSE!) | Model expects noisy lags, smooth TE is OOD |
| Linear blend model+TE | LB=0.83 | Better but geometric mean is optimal for RMSLE |

**Key anti-pattern**: Replacing ffill values with TE estimates makes underprediction WORSE.
The model learned on noisy real lags — smooth TE estimates are a different kind of OOD.

### Phase 3: Geometric Mean Blend Breakthrough (LB 0.670)

```
Approach: Blend model predictions with TE level using geometric mean in log1p space
Formula: final = expm1(alpha * log1p(model) + (1-alpha) * log1p(te_level))
Best alpha: 0.01 (1% model, 99% TE)
Result: LB = 0.670 (2.8x improvement!)
```

**Why it works**:
- RMSLE is a log-space metric → averaging in log-space is the natural operation
- Model provides "ranking signal" (which items sell more/less)
- TE level provides "magnitude signal" (what is the actual sales level)
- Even 1% model contribution adds meaningful ranking signal

### Phase 4: Day-Specific Models Breakthrough (LB 0.399)

```
Approach: Train 16 separate models, one per prediction day
Key: ALL models use features from the last training date (real data, no ffill)
Result: LB = 0.399 (1.7x improvement over geo blend!)
```

**Why it works**:
- All lag features reference REAL data — never stale
- No error accumulation — each day predicted independently
- Model itself produces correct magnitude — NO post-processing needed
- This is the 1st place approach from the original Favorita competition

### Phase 4.1: Critical Discovery — Geo Blend is HARMFUL for Day-Specific Models

| Variant | LB Score |
|---------|----------|
| **Raw day-specific** | **0.399** (BEST) |
| Geo blend 1/99 | 3.528 |
| Geo blend 10/90 | 3.233 |

**Why blending hurts**: Day-specific models already produce correct-magnitude predictions
(mean=432). Blending with TE level (also ~400) introduces numeric distortion in log-space.

## The Methodology (Step-by-Step)

### Step 1: Diagnose the Underprediction

```python
# If this ratio < 0.5, you have a stale lag problem
ratio = preds.mean() / train['sales'].mean()
print(f"Underprediction ratio: {ratio:.3f}")
# Expected: 0.05-0.15 for severe stale lag
```

Also check:
- Per-day prediction means: day 1 worst, day 16 best = stale lag signature
- Lag feature importance: if >40%, they dominate but are unreliable at test time

### Step 2: Check if Your Model Architecture Matches the Problem

| Model Type | When to Use | Expected Behavior |
|-----------|-------------|-------------------|
| Unified model + geo blend | When day-specific is too expensive | Underpredicts, needs blend fix |
| Day-specific models | When you have enough data per day | Correct magnitude, no blend needed |
| Recursive prediction | When lag features are the ONLY signal | Error accumulation, usually bad |

### Step 3: Implement Day-Specific Models (If Applicable)

```python
# For each prediction day d (1 to N):
for d in range(1, N+1):
    # Training: features from date t → target = sales on date t+d
    # Reference features from date t (ALWAYS from real, known data)
    # Target-date features: calendar, TE, promotions for date t+d

    # Test: ALL models use features from last training date
    # NO ffill, NO recursive prediction
```

### Step 4: Multi-Level Aggregation (1st Place Enhancement)

Compute features at multiple aggregation levels, not just the finest grain:

| Level | Group Key | What It Captures |
|-------|-----------|-----------------|
| Store-Item | store_nbr, family | Individual item behavior (current) |
| Item-Only | family | Category-wide trends across stores |
| Store-Cluster | store_nbr, cluster | Store-segment behavior |
| Store-Only | store_nbr | Overall store health |

The 1st place solution used `store x item`, `item only`, and `store x class` as the three
most impactful aggregation levels. Additional levels (`cluster x item`, `store x family`)
were tested and found useless.

### Step 5: Validation Strategy

- Use rolling time-series CV with multiple folds
- Validate on periods that mirror the test structure (16 days ahead)
- Don't trust a single CV split
- The 1st place solution used exactly 16 days as validation period

## Decision Tree

```
Multi-step time series forecasting
├── Diagnose: preds.mean() / train.mean()
│   ├── > 0.8: Model is fine, optimize features/hyperparams
│   ├── 0.1-0.8: Stale lag problem detected
│   │   ├── Can train N models? → Day-specific models (BEST)
│   │   │   ├── Check: do predictions have correct magnitude?
│   │   │   │   ├── Yes → Submit raw, NO blending
│   │   │   │   └── No → Debug feature engineering
│   │   │   └── Add multi-level aggregation features
│   │   └── Only 1 model? → Geometric mean blend with TE
│   │       ├── Alpha 0.01-0.05 for RMSLE
│   │       └── TE = per(store, item, day_of_week) mean
│   └── < 0.1: Severe bug (NaN cascade or wrong target)
```

## Key Principles

1. **Stale lag features are the #1 enemy** in multi-step forecasting
2. **Day-specific models eliminate stale lags** — the 1st place approach
3. **Geometric mean is for unified models only** — harmful for day-specific
4. **TE-fill is an anti-pattern** — worse than ffill
5. **Multi-level aggregation** captures trends at different granularity levels
6. **CV-LB gap is the diagnostic signal** — if >2x, you have a fundamental issue
7. **Prediction mean should match training mean** — the simplest check

## Competition Results Timeline

| Round | Approach | LB | Key Change |
|-------|----------|-----|-----------|
| R1 | Baseline LightGBM | 1.859 | Initial with ffill |
| R8 | Geo blend 1/99 | 0.670 | Geometric mean breakthrough |
| R10 | Day-specific raw | 0.399 | Eliminated stale lags entirely |

## What NOT to Do

1. Don't replace ffill with TE estimates (makes things worse)
2. Don't apply geo blend to day-specific models (destroys accuracy)
3. Don't use recursive prediction for multi-step (error accumulates)
4. Don't assume more data = better (1st place used only 50 days of 2017)
5. Don't add features that are unavailable at test time
6. Don't trust holidays blindly (1st place found them useless)

## See Also

- `ts-lag-nan-cascade-bug`: NaN cascade prerequisite fix
- `ts-lag-stale-underprediction`: Geometric mean blend workaround
- `ts-day-specific-forecasting`: Day-specific model implementation
- `kaggle-optimal-blending`: General blending strategy

## References

- 1st place solution (sjv/shixw125): 70% day-specific + 30% unified
- Journey: LB 1.859 → 0.670 → 0.399 (4.7x total improvement)
- Original discussion: https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/47582
