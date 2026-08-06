---
name: autogluon-preset-strategy
description: |
  AutoGluon preset selection strategy: when to use medium/good/high/best_quality,
  when EDA is unnecessary, and how to tune within a preset. Validated on s6e7
  (AG high_quality 600s → OOF=0.8739, LB=0.87458, gap=0.0007 — perfect alignment).
  Use when: (1) Deciding which AutoGluon preset to start with, (2) Wondering
  whether to do manual EDA before fitting, (3) Wanting to know which hyperparameters
  to tune first, (4) Deciding time budget for tabular training. Differs from
  `autogluon-first` (which says "just use best_quality"): this skill explains
  the preset tradeoffs and where manual intervention pays off vs not.
---

# AutoGluon Preset Strategy

## Problem

`autogluon-first` says "use best_quality preset always". But:
- `best_quality` takes 1-4 hours — wasted if you only need a quick baseline
- `medium_quality` is the default but **doesn't enable bagging/stacking** → no OOF predictions
- The choice between presets has a 5x time × accuracy tradeoff that the existing skill doesn't quantify

## Decision Tree

```
N = training rows
N < 10K   → presets='best_quality' (single GBDT plateau is real concern)
N 10K-100K → presets='good_quality' first (10-30 min), upgrade to high if time allows
N > 100K  → presets='good_quality' is enough (data variance dominates model choice)

GPU available? 
  No → presets='high_quality' max (TabPFNv2 needs GPU)
  Yes → consider 'extreme_quality' (uses TabPFNv2, TabICL, Mitra)

Metric?
  balanced_accuracy / F1 → add calibrate_decision_threshold=True
  accuracy → default ('auto')
  log_loss → use predict_proba, no calibration
```

## Preset Reference (from AG 1.4.0 docstring)

| Preset | Train Time | OOF Available? | Models | When |
|--------|-----------|----------------|--------|------|
| `medium_quality` (default) | ~5min | ❌ No (auto_stack=False) | 3 | NEVER for serious use |
| `good_quality` | 10-30min | ✅ Yes | 12 | **Default starting point** |
| `high_quality` | 30-60min | ✅ Yes | 14 | When good isn't enough |
| `best_quality` | 1-4h | ✅ Yes | ~100 | Kaggle competition |
| `extreme_quality` | 2-8h | ✅ Yes | 22+TabFM | GPU only, 30K rows ideal |

**Why medium_quality has no OOF**: AG's `predict_oof()` requires bag mode (`auto_stack=False` means no bag). The "val score" reported is in-sample on the holdout — not real OOF. Don't trust it.

## Empirical Validation (s6e7, N=690K, 3-class)

| Preset | Time | Models | Best Val | OOF | OOF/Val gap |
|--------|------|--------|----------|-----|-------------|
| medium_quality | 182s | 3 | 0.8825 | N/A | N/A |
| good_quality | 376s | 12 | 0.8730 | 0.8730 | 0 |
| high_quality | 947s | 14 | 0.8739 | 0.8739 | 0 |

**Surprising finding**: medium_quality's val score (0.8825) is **higher** than good/high (0.873). This is **fake**: medium has no bag/stack, val is in-sample on the internal holdout. The true OOF (good=0.8730, high=0.8739) is what you should trust.

**Practical implication**: good→high gives only +0.0009 OOF improvement at 2.5x time cost. **good_quality is the sweet spot** unless you have hours to burn.

## The "Do I Need EDA?" Question

**Short answer**: For standard tabular (numeric + categorical features, no time series, no text), you do NOT need EDA before fitting AG.

| Data Type | EDA Before AG? | Why |
|-----------|---------------|-----|
| Numeric + categorical (standard) | ❌ No | AG handles missing/encoding automatically |
| Heavy missingness (>30%) | ❌ No | AG's NN_TORCH handles it natively |
| Class imbalance (>10:1) | ❌ No | Set eval_metric='balanced_accuracy' |
| Time series | ✅ Yes | AG default doesn't create lag/rolling features |
| Text features | ⚠️ Optional | AG has NGRAM auto-generation; manual helps sometimes |
| Image features | ✅ Yes | Need AG_AUTOMM, different setup |
| Business rules | ✅ Yes | Encode rules into target/feature |

**Validation**: s6e7 (690K rows, 14 features, ~50% missing) — fit AG directly with no manual EDA. Result: OOF=0.8739, LB=0.87458, gap=0.0007. **No EDA was needed**.

## Tuning Within a Preset

After baseline (`good_quality`, 30 min), try in order:

### Priority 1: Increase time_limit (cheapest)
```python
predictor.fit(tr, time_limit=3600, presets='good_quality')  # 6x longer
```

### Priority 2: Customize key models
```python
predictor.fit(tr, presets='good_quality',
              hyperparameters={'GBM': {'num_leaves': 64, 'learning_rate': 0.03},
                              'CAT': {'depth': 8}})
```

### Priority 3: Use refit_full (more data = better)
```python
predictor.fit(tr, presets='high_quality', refit_full=True)
# refit trains on ALL data (no holdout) — typically +0.001-0.005 OOF
```

### Priority 4: Decision threshold calibration
```python
predictor.fit(tr, eval_metric='balanced_accuracy',
              calibrate_decision_threshold=True)
# Helps when metric is balanced_accuracy or F1
```

## Standard Workflow

```python
# Step 1: Baseline (10-30 min)
predictor = TabularPredictor(label='target', eval_metric='accuracy').fit(
    tr, time_limit=600, presets='good_quality')

# Step 2: Inspect (1 min)
print(predictor.leaderboard(silent=True))
print(predictor.feature_importance(tr))

# Step 3: If you need more (decide based on Step 2)
# - Models are similar → refit_full=True
# - One model dominates → tune hyperparameters
# - Need more time → longer time_limit
predictor = TabularPredictor(label='target', eval_metric='accuracy').fit(
    tr, time_limit=3600, presets='good_quality',
    hyperparameters={'GBM': {'num_leaves': 64}})

# Step 4: Predict and submit
pred = predictor.predict(te)
```

## When AG Wins and Loses

**AG Wins**:
- Standard tabular with 1K-1M rows
- No time budget constraints
- You don't need to understand model decisions
- Multiple data types (numeric + categorical + text) — AG handles all

**AG Loses**:
- Real-time inference (<10ms latency) — AG ensembles are slow
- Custom loss functions
- Tiny data (<100 rows) — GBDT overfits, AG can't help
- Adversarial / non-stationary targets — AG assumes IID
- You need SHAP/explainability for **each** model decision

## Anti-Patterns

| Anti-pattern | Why wrong | Fix |
|--------------|----------|-----|
| Use medium_quality because it's default | No bag/stack, no OOF | Use good_quality |
| Skip AG and start manual LightGBM | Hours vs minutes | Run AG baseline first |
| Spend days tuning GBDT hyperparameters | AG already HPO-tunes | Trust AG's defaults |
| Do extensive EDA before fitting AG | AG auto-handles standard cases | Just fit, inspect, then iterate |
| Hand-craft 30+ features when AG already does | Diminishing returns | Add only domain-specific signals |

## Key API Methods

```python
# Inspect
predictor.leaderboard(silent=True)              # Model rankings + val scores
predictor.feature_importance(tr)                 # Permutation importance
predictor.model_names()                         # List of trained models
predictor.info()                                # Full info

# Predict
predictor.predict(te)                           # Class labels
predictor.predict_proba(te)                     # Class probabilities

# Inspect best model
best_model = predictor.model_best
predictor.predict_oof()                          # Only if bag mode enabled
predictor.refit_full()                          # Retrain best on all data
```

## When to Use This Skill vs `autogluon-first`

| Scenario | Use this skill | Use `autogluon-first` |
|----------|---------------|----------------------|
| Choose first preset | ✅ | ❌ (says use best_quality always) |
| Decide if EDA needed | ✅ | ❌ |
| Optimize within preset | ✅ | ❌ |
| Strategic "why use AG" reasoning | ❌ | ✅ |
| Quick reference of AG principles | ❌ | ✅ |

## References

- AG 1.4.0 docstring (inspect.getsource on TabularPredictor.fit)
- [AutoGluon 1.4.0 release notes](https://github.com/autogluon/autogluon/releases)
- Erickson et al. (2020) "AutoGluon-Tabular" arXiv:2003.06505
- Salinas & Erickson (2025) "TabRepo" arXiv:2507.07829 (zeroshot hyperparameter portfolio)