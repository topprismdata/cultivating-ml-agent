---
name: ts-day-specific-forecasting
description: |
  Day-specific (direct) multi-step time series forecasting. Trains N separate models,
  one per prediction horizon day, with all features computed from the last known date.
  Eliminates stale lag feature problem entirely — predictions at correct magnitude
  without post-processing. CRITICAL: Do NOT apply geometric mean blending to day-specific
  model outputs — raw predictions are already at correct magnitude; blending DESTROYS
  accuracy (0.40 vs 3.2+). Use when: (1) Multi-step time series prediction (e.g., 16-day
  forecast), (2) Lag features become stale/constant during test, (3) Model underpredicts
  by 5-10x despite good CV, (4) Post-processing blends (geometric mean) are needed to
  fix magnitude in unified model. This is the 1st place approach from Favorita grocery
  sales competition. Applies to any multi-step time series forecasting with GBM/neural
  network models.
---

# Day-Specific Direct Forecasting for Multi-Step Time Series

## Problem

In multi-step time series forecasting (predicting N days ahead), a single unified model
suffers from stale lag features at test time. All N days get the same ffill'd lag values
(constant), causing the model to underpredict by 5-10x. Even with geometric mean
post-processing (see `ts-lag-stale-underprediction`), the model only provides "ranking
signal" — the magnitude comes from target encoding, not the model itself.

## Context / Trigger Conditions

Use when:
- Predicting N days (N > 1) into the future with lag/rolling features
- A single unified model underpredicts at test time due to stale lag features
- Post-processing blends (geometric mean with TE) are needed but feel like a workaround
- CV score is good but LB score is 3-10x worse
- You want the model itself to produce correct-magnitude predictions

**Common in**: Kaggle time series competitions (Store Sales, M5, Web Traffic),
retail demand forecasting, any multi-step prediction with autoregressive features.

## Solution

### Core Idea: Train N Separate Models

Instead of 1 model that predicts all N days, train N separate models where:
- Model_d predicts "sales d days from the reference date"
- Features are ALWAYS computed from the last known training date
- At test time, ALL lag features reference real, known data — no ffill needed

### Implementation

```python
# For each prediction day d (1 to N):
for d in range(1, N + 1):
    # Training: features from date t → target = sales on date t+d
    ref_dates = [t for t in all_dates if t + d is still in training data]
    target_dates = [t + timedelta(days=d) for t in ref_dates]

    # Reference features (from date t, always known)
    ref_data = train_features[train_features["date"].isin(ref_dates)]

    # Target sales (d days ahead)
    target_data = train_features[train_features["date"].isin(target_dates)]

    # Merge on (store, family, target_date)
    merged = ref_data.merge(target_data, on=["store_nbr", "family", "target_date"])

    # Add TARGET-DATE features (calendar, holidays, promotions on target date)
    merged["target_day_of_week"] = merged["target_date"].dt.dayofweek
    merged["target_month"] = merged["target_date"].dt.month
    merged["target_is_weekend"] = (merged["target_day_of_week"] >= 5).astype(int)

    # Add TARGET-DATE target encoding
    # te_sf_dow_mean for the TARGET day_of_week, not the reference day
    merged = merged.merge(te_sf_dow, on=["store_nbr", "family", "target_day_of_week"])

    # Train model_d on this data
    model_d = lgb.LGBMRegressor(...)
    model_d.fit(X_train, np.log1p(y_train), eval_set=[(X_val, np.log1p(y_val))])
```

### Test-Time Prediction

```python
# ALL models use features from the SAME last training date
last_date_features = train_features[train_features["date"] == last_train_date]

for d in range(1, N + 1):
    target_date = last_train_date + timedelta(days=d)
    test_data = last_date_features.copy()

    # Add target-date features
    test_data["target_day_of_week"] = target_date.dayofweek
    test_data["target_month"] = target_date.month
    test_data = test_data.merge(te_sf_dow, on=["store_nbr", "family", "target_day_of_week"])

    # Predict — ALL lag features are from real data!
    preds_d = np.expm1(model_d.predict(test_data[features]))
```

### Key Insight: Two Types of Features

1. **Reference-date features** (from the last training date):
   - Lag features: `sales_lag_1`, `sales_lag_7`, etc.
   - Rolling features: `sales_roll_mean_7`, `sales_roll_std_14`
   - EWM features: `sales_ewm_7`
   - All computed from REAL training data — never stale!

2. **Target-date features** (from the day being predicted):
   - Calendar: `day_of_week`, `month`, `is_weekend`, `is_holiday`
   - Target encoding: `te_sf_dow_mean` for target's day_of_week
   - Promotions: `onpromotion` for the target date
   - Oil price: interpolated for the target date

## Results (Verified)

**Kaggle Store Sales Competition**:

| Approach | Prediction Mean | LB Score (RMSLE) |
|----------|----------------|------------------|
| Unified model (ffill) | 40 | 1.859 |
| Unified + geo blend 1/99 | ~343 | 0.670 |
| **Day-specific models (raw)** | **432** | **0.399 (BEST)** |
| Day-specific + geo blend 1/99 | — | 3.528 |
| Day-specific + geo blend 10/90 | — | 3.233 |

Day-specific model predictions (raw, no blending):
- Day 1 (Wed): mean=439 | Day 4 (Sat): mean=530 | Day 5 (Sun): mean=558
- Weekend predictions correctly higher — model learned the seasonal pattern
- Raw predictions at correct magnitude — NO post-processing needed

### CRITICAL: Geo Blend is HARMFUL for Day-Specific Models

Unlike unified models (where geo blend with TE improves LB from 1.8→0.67), applying
the same geometric mean blending to day-specific models **DESTROYS** accuracy:

| Day-specific variant | LB Score |
|---------------------|----------|
| **Raw predictions** | **0.39880** (BEST) |
| Geo blend alpha=0.01 (1/99) | 3.528 |
| Geo blend alpha=0.02 (2/98) | 3.495 |
| Geo blend alpha=0.05 (5/95) | 3.396 |
| Geo blend alpha=0.10 (10/90) | 3.233 |

**Why**: Day-specific models produce predictions at the correct magnitude (mean=432 vs
training mean=358). Blending with TE level (which is also ~400) and then applying log1p
averaging introduces numeric instability — both components are similar magnitude, so the
blend doesn't help and actually distorts the predictions.

**Rule**: Geo blend is for UNIFIED models that underpredict (mean=40). Day-specific
models (mean=432) should be used RAW.

## Why This Works

1. **All lag features reference real data**: No ffill, no recursive prediction, no guessing
2. **No error accumulation**: Each day is predicted independently
3. **Model produces correct magnitude**: No post-processing blend needed
4. **Target-date TE features**: Provide seasonality information specific to each prediction day
5. **First place approach**: Used by sjv/shixw125 in the original Favorita competition
   (70% weight on day-specific models in the winning ensemble)

## Enhancements

### Multi-Level Aggregation (from 1st place solution)
Compute features at multiple aggregation levels, not just store-item:
- Store-item level (standard)
- Item-only level (across all stores)
- Store-class level (items in same category at same store)

### Ensemble with Unified Model
- 70% day-specific models + 30% unified model (1st place pattern)
- The unified model captures inter-day relationships
- The day-specific models provide clean, accurate predictions

### Multiple Validation Periods
- Don't trust a single CV split in time-series
- Validate on multiple periods across different years/seasons

## Verification

1. Check prediction mean: should be close to training mean (ratio ~0.8-1.2)
2. Check per-day means: should vary realistically (weekends higher for retail)
3. Compare CV score with LB score: should be similar (no 3-10x gap)
4. Diagnostic: `preds.mean() / train['sales'].mean()` should be > 0.5

## Notes

- **CV is typically higher** than unified model (0.407 vs 0.362 in our case) because
  predicting "d days ahead" is inherently harder than "predicting today"
- **LB should be close to CV** — the key advantage is that test predictions match
  the CV distribution (no stale feature gap)
- **Training cost**: N models instead of 1, so Nx training time
- **Do NOT replace ffill with TE estimates** — this makes underprediction worse
  (see `ts-lag-stale-underprediction` for the TE-fill anti-pattern)
- The 1st place Favorita solution also added promotion-specific statistics
  (mean and exponentially weighted sum of promotional vs non-promotional days)

### See Also
- `ts-lag-stale-underprediction`: The geometric mean blend workaround (fallback)
- `ts-lag-nan-cascade-bug`: The NaN cascade issue (prerequisite fix)
- `ts-forecasting-stale-lag-methodology`: MASTER methodology — complete journey from LB 1.859→0.399
- `kaggle-top-solution-replication`: Methodology for studying winning solutions

## References

- Kaggle Favorita 1st place solution (sjv/shixw125): 70% day-specific + 30% unified
- Ensemble weights: 0.42 * day_specific_LGB + 0.28 * day_specific_NN + 0.18 * unified_LGB + 0.12 * unified_NN
- Code: https://github.com/sjvasquez/web-traffic-forecasting/tree/master/cf
- Discussion: https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/47582
