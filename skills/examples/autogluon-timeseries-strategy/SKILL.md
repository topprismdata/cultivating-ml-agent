---
name: autogluon-timeseries-strategy
description: |
  AutoGluon TimeSeriesPredictor: special API and presets for time series forecasting
  (different from TabularPredictor). Validated on Store Sales (N=3M, 33 families × 54
  stores × 1684 days): AG TimeSeriesPredictor medium_quality 300s → LB 0.41852 RMSLE.
  Use when: (1) Working on time series competitions (forecasting), (2) Have multi-series
  data (multiple stores/products/regions), (3) Need to handle lag features automatically,
  (4) Comparing manual lag engineering vs AG's built-in features. Differs from
  `autogluon-preset-strategy` (which covers TabularPredictor) — this skill covers
  the completely separate TimeSeriesPredictor API, data format, presets, and
  frequency specification.
---

# AutoGluon TimeSeriesPredictor Strategy

## The Critical Difference

**TabularPredictor ≠ TimeSeriesPredictor**. They are separate APIs with different:
- Data formats (TimeSeriesDataFrame vs DataFrame)
- Cross-validation (multi-window backtesting vs k-fold)
- Evaluation metrics (RMSLE/MAE/MAPE vs RMSE/accuracy)
- Models (DeepAR/TFT/Chronos vs LightGBM/XGBoost)

```python
# Tabular (independent rows):
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label='target').fit(df)

# Time series (sequential, grouped by entity):
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
predictor = TimeSeriesPredictor(target='target', prediction_length=16, freq='D').fit(ts_df)
```

## Data Format (Critical)

TimeSeriesPredictor requires **TimeSeriesDataFrame**:

```python
from autogluon.timeseries import TimeSeriesDataFrame
import pandas as pd

# Required columns: item_id, timestamp, target
# Each item_id is ONE time series (e.g., one store-product combination)
df = pd.DataFrame({
    'item_id': ['store_1', 'store_1', 'store_1', 'store_2', 'store_2'],
    'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03',
                                  '2024-01-01', '2024-01-02']),
    'target': [100, 110, 105, 50, 55],
    'onpromotion': [0, 1, 0, 0, 1]  # optional covariates
})

ts_df = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column='item_id',
    timestamp_column='timestamp'
)
```

**Common mistake**: passing `pd.DataFrame` directly to `predictor.fit()` with a date column. AG will reject it.

## Core Parameters

| Parameter | Required | Default | Purpose |
|-----------|----------|---------|---------|
| `target` | Yes | None | Column name to forecast |
| `prediction_length` | Yes | 1 | How many steps ahead to forecast |
| `freq` | Yes (or inferred) | None | 'D', 'H', 'M', 'Q', etc. |
| `eval_metric` | No | 'WQL' | RMSLE, MAE, MAPE, WQL, MASE |
| `known_covariates_names` | No | None | Future-known variables (e.g., holidays) |
| `quantile_levels` | No | [0.1, ..., 0.9] | For probabilistic forecasts |
| `path` | No | None | Model save directory |

## Presets — AutoGluon 1.4 vs 1.5 (Critical Update)

**IMPORTANT: AG 1.5 (released 2025-12-19) made significant changes to TimeSeries presets.** This skill is based on AG 1.4 source code. If you're on 1.5+, see the "AG 1.5 Differences" section below.

### AG 1.4 Presets (what this skill is built on)

**IMPORTANT: TimeSeriesPredictor has NO `extreme_quality` preset.** That preset only exists in TabularPredictor.

| Preset | Models | Time | Best For |
|--------|--------|------|----------|
| `fast_training` | Naive, SeasonalNaive, ETS, Theta, RecursiveTabular, DirectTabular | 1-2 min | Quick baseline |
| **`medium_quality`** | Above + TemporalFusionTransformer + Chronos-Bolt small | 5-15 min | **Recommended starting point** |
| `high_quality` | DL + ML + statistical mix + Chronos (zero-shot + fine-tuned) + TiDE | 30-60 min | Strong forecast |
| `best_quality` | **Same models as high_quality** + `num_val_windows=2` (multi-backtest validation) | 1-4 hours | Maximum accuracy via robust validation |
| `bolt_tiny` | Chronos-Bolt tiny only | <1 min | Zero-shot baseline |
| `bolt_mini` | Chronos-Bolt mini only | <2 min | Zero-shot |
| `bolt_small` | Chronos-Bolt small only | <5 min | Zero-shot |
| `bolt_base` | Chronos-Bolt base only | <10 min | Zero-shot (most accurate Chronos) |

**Hidden presets (verified from source `presets_configs.py`)**:

| Preset | Equivalent | Use |
|--------|-----------|-----|
| `chronos` | `chronos_small` | Alias |
| `chronos_tiny` | Original Chronos tiny | (rarely useful now) |
| `chronos_mini` | Original Chronos mini | (rarely useful now) |
| `chronos_small` | Original Chronos small | (rarely useful now) |
| `chronos_base` | Original Chronos base | (rarely useful now) |
| `chronos_large` | Original Chronos large (batch_size=8) | (rarely useful now) |
| `chronos_ensemble` | Chronos small + 4 statistical models | **Hidden gem — better than chronos_small alone** |
| `chronos_large_ensemble` | Chronos large + 4 statistical models | Best of original Chronos line |
| `best` / `high` / `medium` | `best_quality` / etc. | Shorthand aliases |
| `bq` / `hq` / `mq` | `best_quality` / etc. | Even shorter aliases |

**Key difference vs Tabular presets**:
- TimeSeries has **NO `good_quality`** (Tabular does)
- TimeSeries has **NO `extreme_quality`** (Tabular does — uses GPU foundation models)
- TimeSeries `best_quality` ≠ Tabular `best_quality`:
  - Tabular: ~100 models via zeroshot hyperparameter portfolio
  - TimeSeries: same models as `high_quality`, but uses `num_val_windows=2` (2 backtest windows for robust model selection)

### AG 1.5 Differences (Breaking Changes!)

**Released 2025-12-19. Source: GitHub release notes.**

| Preset | AG 1.4 | AG 1.5 |
|--------|--------|--------|
| `fast_training` | ✅ | ✅ |
| `medium_quality` | ✅ | ✅ |
| `high_quality` | ✅ | ✅ |
| `best_quality` | ✅ | ✅ |
| `bolt_*` (Chronos-Bolt) | ✅ | ✅ |
| `chronos` (alias for chronos_small) | ✅ | ❌ **REMOVED** |
| `chronos_tiny/mini/small/base/large` | ✅ | ❌ **REMOVED** |
| `chronos_ensemble` | ✅ (hidden) | ❌ **REMOVED** |
| `chronos2` | ❌ | ✅ **NEW** |
| `chronos2_small` | ❌ | ✅ **NEW** |
| `chronos2_ensemble` | ❌ | ✅ **NEW** |

**Migration 1.4 → 1.5**:
```python
# ❌ 1.4 → 1.5 breaks
predictor.fit(data, presets='chronos_small')
predictor.fit(data, presets='chronos_ensemble')

# ✅ 1.5 replacement
predictor.fit(data, presets='chronos2')
predictor.fit(data, presets='chronos2_ensemble')
```

**New models in 1.5** (verified from release notes):
- **Chronos-2** (zero-shot, LoRA fine-tune, full fine-tune)
- **Toto** (new time series model)

**Other 1.5 changes**:
- `num_val_windows="auto"` — automatic backtesting configuration
- `refit_every_n_windows="auto"` — automatic refit scheduling
- New methods: `predictor.backtest_predictions()`, `predictor.backtest_targets()`
- Multi-layer stack ensembles for time series
- 80% win rate vs 1.4 (with same time budget)
- 10min of 1.5 > 2hr of 1.4

**Other breaking changes in 1.5**:
- **Python 3.10+ required** (1.4 worked on 3.9)
- **Models trained on 1.4 cannot be loaded in 1.5** — must retrain
- Major dependency upgrades (torch 2.6-2.10, ray 2.43-2.53, etc.)

## Empirical Validation (Store Sales)

| Setup | Time | RMSLE val | LB |
|-------|------|-----------|-----|
| AG TimeSeries `medium_quality` 300s | 240s | 0.4813 | **0.41852** |
| Manual LightGBM with lag features | 5+ min | 0.5-0.6 | 0.4-0.5 (varied) |
| Manual stacking ensemble (prior best) | 1+ hour | — | ~0.4-0.5 |

**Key insight**: AG TimeSeries achieves competitive LB in **4 minutes** without manual lag engineering, time series CV, or feature engineering. The models (DirectTabular, RecursiveTabular, Naive, SeasonalNaive) automatically handle:
- Lag features
- Rolling statistics
- Calendar features (day of week, month)
- Holiday effects (if known_covariates_names provided)

## Validation: Multi-Window Backtesting

Unlike Tabular's KFold, TimeSeriesPredictor uses **multi-window backtesting**:

```python
predictor.fit(
    ts_df,
    time_limit=600,
    presets='medium_quality',
    num_val_windows=3,        # Use 3 backtest windows
    val_step_size=prediction_length  # Step size between windows
)
```

This is AG's equivalent of TimeSeriesSplit, but automated across multiple windows. **You do NOT need to manually implement TimeSeriesSplit**.

## Critical Pitfalls

### Pitfall 1: Wrong data format
```python
# ❌ WRONG: passing regular DataFrame
predictor.fit(df_with_date_column)

# ✓ CORRECT: convert first
ts_df = TimeSeriesDataFrame.from_data_frame(df, id_column='item_id', timestamp_column='timestamp')
predictor.fit(ts_df)
```

### Pitfall 2: Missing freq
```python
# ❌ AutoGluon may guess wrong freq (IRREG → resample to D)
# ✓ Explicit
predictor = TimeSeriesPredictor(target='target', prediction_length=16, freq='D')
```

### Pitfall 3: Trying to use lag features manually
```python
# ❌ AG TimeSeries handles lag internally
df['lag_1'] = df.groupby('item_id')['target'].shift(1)

# ✓ Don't pre-compute lag features
# AG's models (DirectTabular, RecursiveTabular) generate them automatically
```

### Pitfall 4: Using Tabular predictor on time series
```python
# ❌ WRONG: treats each row as independent, ignores time order
from autogluon.tabular import TabularPredictor
TabularPredictor(label='target').fit(df)

# ✓ Use TimeSeriesPredictor for sequential forecasting
from autogluon.timeseries import TimeSeriesPredictor
TimeSeriesPredictor(target='target', prediction_length=16, freq='D').fit(ts_df)
```

## Models Trained by AG TimeSeries (1.4 default preset)

| Model | Type | Speed | Use case |
|-------|------|-------|----------|
| Naive | Last value | ~1s | Baseline |
| SeasonalNaive | Same as last season | ~1s | Strong seasonal data |
| RecursiveTabular | LGB with recursive prediction | ~30s | Many series |
| DirectTabular | LGB with direct multi-step | ~30s | Multi-horizon |
| TemporalFusionTransformer | Attention-based DL | ~5min | Complex patterns |
| DeepAR | RNN-based | ~5min | Many related series |
| Chronos-Bolt | Pretrained foundation model | ~1min | Zero-shot |
| PatchTST | Patch-based transformer | ~5min | Long horizons |
| AutoETS | Automatic Exponential Smoothing | ~10s | Statistical baseline |
| NPTS | Non-Parametric Time Series | ~10s | Statistical baseline |
| DynamicOptimizedTheta | Optimized Theta method | ~10s | Statistical baseline |
| TiDE | Long-term Time-series Dense Encoder | ~5min | Long horizons |

**Note (1.4 only)**: `high_quality` / `best_quality` presets include **automatic fine-tuning of Chronos-Bolt small** with a CatBoost covariate regressor. This is hidden in source `get_default_hps('default')` — not visible in the docstring.

```python
# This is what 1.4 high_quality actually does (hidden in source):
"Chronos": [
    {"ag_args": {"name_suffix": "ZeroShot"}, "model_path": "bolt_base"},
    {"ag_args": {"name_suffix": "FineTuned"}, "model_path": "bolt_small",
     "fine_tune": True, "target_scaler": "standard",
     "covariate_regressor": {"model_name": "CAT", "model_hyperparameters": {"iterations": 1000}}},
],
```

## Standard Workflow

```python
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
import pandas as pd

# Step 1: Prepare TimeSeriesDataFrame
df = ... # your long-format DataFrame
ts_df = TimeSeriesDataFrame.from_data_frame(df, id_column='item_id', timestamp_column='timestamp')

# Step 2: Define predictor
predictor = TimeSeriesPredictor(
    target='target',
    prediction_length=16,        # Match LB forecast horizon
    freq='D',                    # Match your data frequency
    eval_metric='RMSLE',         # Match Kaggle metric
    path='ag_ts_model'
)

# Step 3: Fit (5-15 min for medium_quality)
predictor.fit(ts_df, time_limit=600, presets='medium_quality')

# Step 4: Predict
pred = predictor.predict(ts_df)  # Returns mean + quantile forecasts

# Step 5: Convert to Kaggle submission format
pred_df = pred.reset_index()  # MultiIndex (item_id, timestamp)
pred_df['sales'] = pred_df['mean'].clip(0)  # clip negatives
```

## When to Use TimeSeriesPredictor vs TabularPredictor

| Scenario | Use TimeSeriesPredictor | Use TabularPredictor |
|----------|------------------------|---------------------|
| Sales forecasting | ✅ | ❌ |
| Inventory/demand | ✅ | ❌ |
| Time-series with clear temporal structure | ✅ | ❌ |
| Mixed categorical + numeric, no clear time | ❌ | ✅ |
| Want SHAP per model | ❌ | ✅ |
| Multi-series data | ✅ | ❌ |

## Empirical Recommendation

For most Kaggle time series competitions:

```python
predictor = TimeSeriesPredictor(
    target=target_col,
    prediction_length=forecast_horizon,  # match test set length
    freq='D',  # or 'H', 'M'
    eval_metric='RMSLE'  # or MAE, MAPE, WQL
).fit(ts_df, time_limit=600, presets='medium_quality')
```

This is the **simplest possible starting point**. Custom lag features are usually unnecessary — AG handles them. Stacking/ensembling are usually unnecessary — AG does it. **For time series, AG is even more hands-off than tabular.**

## References

- [AG Timeseries Quick Start](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-quick-start.html)
- [AG Timeseries Models](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-models.html)
- AG 1.4.0 TimeSeriesPredictor docstring (inspect.getsource)
- [Chronos-Bolt paper](https://arxiv.org/abs/2504.05291)
- Store Sales competition: empirical validation (RMSLE LB 0.41852)