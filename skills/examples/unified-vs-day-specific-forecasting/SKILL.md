---
name: unified-vs-day-specific-forecasting
description: |
  Use when: (1) deciding between unified vs day-specific models for multi-step forecasting,
  (2) considering blending day-specific with unified predictions,
  (3) CV shows day-specific worse than unified but unsure if LB will agree,
  (4) building multi-step time series models with >7 day horizons.
  KEY FINDING: Unified model with target_day_offset feature can significantly outperform
  day-specific models, especially when features are rich enough to encode temporal patterns.
---

# Unified Model vs Day-Specific in Multi-Step Forecasting

## The Assumption

Day-specific models (one per horizon day) should outperform a single unified model because:
- Each model specializes in its specific horizon
- No need for the model to "figure out" how day offset affects predictions
- Each model has its full capacity for one task

## The Finding

In Kaggle Favorita Store Sales (16-day horizon, 1782 store-family pairs):

| Approach | CV RMSLE | LB |
|----------|----------|--------|
| Day-specific (16 models) | 0.42567 | 0.39779 |
| Unified (1 model + target_day_offset) | 0.38206 | **0.38850** |
| Blend 70/30 | — | 0.39393 |
| Blend 50/50 | — | ~0.391 |

**Unified model is 0.00929 LB better than day-specific.**

## Why Unified Wins Here

1. **More training data**: Unified sees 16× more samples (46M vs 2.87M per model), giving
   LightGBM more statistical power to learn rare patterns.

2. **Cross-day generalization**: The unified model learns that "sales patterns on Monday"
   are similar whether predicting day 1 (if target is Monday) or day 8 (if target is Monday).
   Day-specific models can't share this knowledge.

3. **Rich features encode temporal structure**: With YoY, TE, and lag features, the model
   already has enough information to distinguish between horizons. The `target_day_offset`
   feature is sufficient for the model to specialize internally.

4. **Day-specific overfits the fold structure**: Each day-specific model trains on the same
   store-family pairs with very similar time splits, potentially overfitting to the CV fold
   boundaries.

## When Day-Specific Might Still Win

- Very long horizons (>30 days) where temporal patterns change dramatically
- When features are minimal (model needs explicit specialization)
- When different horizons have fundamentally different data availability

## Rule of Thumb

- **Rich features + moderate horizon (7-30 days)**: Try unified first
- **Sparse features + long horizon**: Day-specific may be better
- **Always test both** and use controlled comparison (not just CV)

## Evidence

Kaggle Store Sales (Favorita), April 2026:
- R13 (day-specific + YoY): LB=0.39779
- R14 (unified + YoY): LB=0.38850 (NEW BEST, -0.00929)
- Blend actually WORSE than unified alone

## Related

- `yoy-364day-features` — the features used in both approaches
- `ts-day-specific-forecasting` — the day-specific approach (still valid in some contexts)
- `controlled-submission-experiment` — methodology for comparing approaches
