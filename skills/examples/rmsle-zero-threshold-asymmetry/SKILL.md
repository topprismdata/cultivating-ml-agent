---
name: rmsle-zero-threshold-asymmetry
description: |
  Use when: (1) optimizing post-processing thresholds for RMSLE-evaluated competitions,
  (2) considering zeroing out small predictions, (3) implementing min-sales or adaptive
  thresholds for time series forecasting, (4) CV improves but LB degrades after changing
  post-processing, (5) comparing "smart" vs "simple" zeroing strategies.
  CRITICAL: Never assume that a "smarter" post-processing threshold is better for RMSLE
  without controlled experiment verification.
---

# RMSLE Zero-Threshold Asymmetry

## Problem

When optimizing post-processing for RMSLE metrics, "smarter" adaptive zero-thresholds
that use historical min-sales or store-family-level statistics can WORSE leaderboard
scores compared to simple fixed thresholds like `< 0.1 → 0`.

## Symptoms

- Model improvements (better features, better CV) appear to have NO effect or NEGATIVE
  effect on LB after changing post-processing
- Zero ratio in submission increases significantly (e.g., 8% → 12%) after "improving"
  the zeroing logic
- Items with ~60-70% historical zero rate and ~30-40% non-zero rate are most affected
- Controlled experiment (same model + different post-processing) reveals large LB gap

## Root Cause

RMSLE has a fundamental asymmetry:

```
Predicting small positive when actual = 0:  log1p(0.27)^2 = 0.057  (small error)
Predicting 0 when actual = 2.88:            log1p(2.88)^2 = 1.84   (huge error)
```

For items with ~68% zero rate and ~32% non-zero rate (mean ~2.9 when non-zero):
- Expected error of predicting 0.27:  0.68 * 0.057 + 0.32 * 1.26 = 0.44
- Expected error of predicting 0:      0.68 * 0    + 0.32 * 1.84 = 0.59

**Predicting a small positive value is 25% better than predicting 0**, even though
the item is zero 68% of the time. The penalty for missing a non-zero sale (log1p)
far exceeds the penalty for over-predicting a zero sale.

## Solution

### Rule 1: Use simple fixed thresholds for RMSLE

```python
# GOOD: Simple, proven threshold
predictions[predictions < 0.1] = 0

# BAD: Complex adaptive threshold — can aggressively zero legitimate predictions
min_threshold = historical_min_nonzero_sales * 0.5
mask = (predictions > 0) & (predictions < min_threshold) & (zero_rate > 0.5)
predictions[mask] = 0  # This zeros 1,172 legitimate predictions!
```

### Rule 2: Verify post-processing changes with controlled experiments

When changing post-processing, create a controlled submission:
1. Take the SAME model predictions
2. Apply ONLY the post-processing change
3. Submit both versions to compare LB impact

This isolates the post-processing effect from model changes.

```python
# Controlled experiment
r10 = model_A_predictions + postprocess_A  # baseline
r11b = model_B_predictions + postprocess_B  # new
r11c = model_B_predictions + postprocess_A  # controlled: new model, old postproc

# If R11c > R10: model_B is better, postproc_B is neutral/better
# If R11c < R10: model_B is worse
# If R11c ≈ R11b: postproc doesn't matter
# If R11c >> R11b and R11c > R10: model_B is better BUT postproc_B is worse
```

### Rule 3: For RMSLE, err on the side of keeping small predictions

```python
# For RMSLE: prefer false small positive over false zero
# Safe threshold: 0.05-0.10 (keeps predictions that might be real)
# Aggressive threshold: 0.5+ (risks zeroing real sales, HUGE RMSLE penalty)

# If you MUST use adaptive thresholds, use VERY conservative multipliers
threshold = historical_min_nonzero_sales * 0.1  # NOT 0.5!
```

## Prevention

1. **Never change model AND post-processing in the same submission** — always isolate
   variables with controlled experiments
2. **Compare zero ratios** between submissions — if zero ratio jumps > 2%, investigate
3. **Test threshold sensitivity** — submit with 0.05, 0.10, 0.15, 0.20 thresholds
4. **Remember the asymmetry** — RMSLE punishes false zeros far more than false small positives

## Evidence (Real Experiment)

Kaggle Store Sales (Favorita) competition, April 2026:

| Version | Model | Post-processing | Zeros | LB |
|---------|-------|----------------|-------|--------|
| R10 | baseline | `< 0.1 → 0` (simple) | 8.36% | 0.39880 |
| R11b | improved | min_sales adaptive | 12.41% | 0.40073 |
| R11c | improved | `< 0.1 → 0` (simple) | 8.30% | **0.39824** |

- R11b model was actually BETTER (+0.00056 improvement)
- But R11b post-processing added 1,172 wrong zeros, costing -0.00249
- Net: -0.00193 (model improvement masked by post-processing regression)
- R11c (controlled experiment) proved: model improvement is real, post-processing is the problem

## Notes

- This applies specifically to RMSLE (root mean squared log error). For MAE or MSE,
  the asymmetry may differ — always test.
- For MAPE, the asymmetry is REVERSED: zero predictions for zero actuals are perfect,
  but small positive predictions for zero actuals create infinite percentage errors.
- See also: `kaggle-optimal-blending` (blending strategies), `ts-day-specific-forecasting`
  (day-specific model approach)
