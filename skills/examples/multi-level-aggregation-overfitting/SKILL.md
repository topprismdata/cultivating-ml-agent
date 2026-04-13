---
name: multi-level-aggregation-overfitting
description: |
  Use when: (1) considering adding family-level or store-level aggregation features to
  day-specific models, (2) tempted to copy hierarchical features from 1st place solutions,
  (3) CV improves but LB degrades after adding aggregated features.
  WARN: Multi-level aggregation features that work in unified models may OVERFIT in
  day-specific (direct) forecasting frameworks.
---

# Multi-Level Aggregation Overfitting in Day-Specific Models

## Problem

1st place solutions often use family-level and store-level aggregation features
(family lag, store rolling mean, store-family ratio features). However, when
applied to **day-specific models** (separate model per horizon day), these features
can cause overfitting: CV improves but LB degrades.

## Why It Overfits

In day-specific models, each model trains on ~2.87M samples covering all (store, family)
pairs. The model already sees sufficient examples per pair. Family/store level features
add redundancy:

1. **Leakage via aggregation**: Family-level lags are highly correlated with individual
   store-family lags (especially for families with few stores). The ratio features
   (`sf_to_fam_ratio`) may capture noise rather than signal.

2. **Cross-pair interference**: Day-specific models learn patterns across all pairs
   simultaneously. Adding aggregated features increases the feature space without adding
   truly independent information.

3. **CV overfitting**: The expanding-window CV may not penalize these features enough
   because the family/store patterns are stable across time, but they don't generalize
   to the test period's specific dynamics.

## Evidence

| Version | Features | CV RMSLE | LB |
|---------|----------|----------|--------|
| R11b (base) | 82 features | 0.42041 | 0.40073 |
| R12 multilevel | 82 + 18 agg | 0.41668 (better) | 0.39874 (worse vs R11c=0.39824) |

CV improved by 0.00373 but LB got worse by 0.00050 relative to R11c.

In contrast, YoY features (only 4 features) improved LB by 0.00045 with less CV improvement.

## Rule of Thumb

- **Unified model**: Multi-level aggregation likely helps (model needs hints about hierarchy)
- **Day-specific model**: Multi-level aggregation may overfit (model already sees all pairs)
- **Safer alternative**: Use target encoding at family/store level instead of raw aggregation
- **Feature budget**: Prefer fewer high-signal features (like YoY) over many correlated ones

## Related

- `yoy-364day-features` — the feature that actually worked instead
- `controlled-submission-experiment` — how we isolated this finding
- `ts-day-specific-forecasting` — the framework where this applies
