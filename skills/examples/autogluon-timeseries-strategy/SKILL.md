---
name: autogluon-timeseries-strategy
description: |
  AutoGluon TimeSeriesPredictor: special API and presets for time series forecasting
  (different from TabularPredictor). Validated on Store Sales (N=3M, 33 families × 54
  stores × 1684 days): AG 1.5 Chronos-2 + Chronos + onpromotion covariates →
  **LB RMSLE 0.39525** (best historical, vs AG 1.4 0.41852, vs manual 3.0+).
  Use when: (1) Working on time series competitions (forecasting), (2) Have multi-series
  data with optional covariates (promotions, holidays, prices), (3) Want AG 1.5 Chronos-2
  zero-shot OR fine-tuned, (4) Need to bypass HF download errors. Differs from
  `autogluon-preset-strategy` (which covers TabularPredictor). Key breakthrough:
  using `model_path=LOCAL_PATH` to bypass `hf-mirror.com` download errors.
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

---

## Chronos-2 + Covariates: The 0.39525 Breakthrough (Store Sales)

**Date: 2026-07-10. Validated on Kaggle Store Sales competition.**

The combination of three AG 1.5 features beats all previous attempts:

| Approach | OOF (RMSLE) | LB (public) | Date |
|----------|-------------|-------------|------|
| Manual LightGBM + lag features + stacking | — | 3.0+ | 2026-07-07/08 |
| AG 1.4 medium_quality | -0.4813 | 0.41852 | 2026-07-10 |
| AG 1.5 best_quality (no Chronos, no covariates) | -0.4381 | 0.40053 | 2026-07-10 |
| **AG 1.5 Chronos-2 + Chronos + onpromotion covariate** | **NaN (val)** | **0.39525** | 2026-07-10 |

**Net improvement**: 0.41852 → 0.39525 = **-0.023 (-5.6%)** vs AG 1.4 baseline.

### What AG 1.5 adds that 1.4 didn't

1. **Chronos-2 model** (`autogluon/chronos-2`, 120M params)
   - Zero-shot forecasting with state-of-the-art accuracy (fev-bench, GIFT-Eval)
   - Native support for **known covariates** (Chronos-Bolt does NOT)
   - Cross-learning: makes joint predictions across time series in a batch
   - Default `presets="chronos2"` for zero-shot; `presets="chronos2_ensemble"` for ensemble

2. **Multi-window backtesting** (`num_val_windows="auto"`)
   - Default: 1 window. Best_quality uses multiple windows for robust selection

3. **New method** `predictor.make_future_data_frame(train_data)` for covariates
   - Returns DataFrame with item_id + timestamp for the next `prediction_length` steps
   - You then fill in known covariate values and pass to `predict(known_covariates=...)`

### Chronos-2 Correct Usage (from official tutorial)

```python
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# Data MUST include future-known covariates (not just past)
ts_full = TimeSeriesDataFrame.from_data_frame(
    pd.concat([train_df, test_df]),  # test has known onpromotion for 16 days
    id_column='item_id',
    timestamp_column='timestamp',
)

predictor = TimeSeriesPredictor(
    target='target',
    prediction_length=16,
    known_covariates_names=['onpromotion', 'dcoilwtico', ...],  # NEW in 1.5
).fit(
    ts_full,
    hyperparameters={
        "Chronos": {"model_path": "<bolt path>"},
        "Chronos2": {"model_path": "<c2 path>"},
    },
    enable_ensemble=False,
    time_limit=600,
)

# Predict with covariates
train_only = ts_full.loc[ts_full.index.get_level_values('timestamp') < test_start]
future_cov = predictor.make_future_data_frame(train_only)
# Fill in known values (e.g., merge with test.csv's onpromotion)
future_cov = future_cov.merge(test_df[['item_id', 'date', 'onpromotion']], ...)
pred = predictor.predict(train_only, known_covariates=TimeSeriesDataFrame(future_cov))
```

### The HF-Mirror Bug: How to Work Around It

When running AG 1.5 in environments where `huggingface.co` is blocked, AG 1.5 defaults to `hf-mirror.com` (which is also unreachable). Result: `OSError: We couldn't connect to 'https://hf-mirror.com'`.

**Solution**: Pre-download models manually + use `model_path=LOCAL_PATH`.

```bash
# Step 1: Pre-download (any machine with internet)
HF_HUB_OFFLINE=0 HF_ENDPOINT=https://huggingface.co python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='autogluon/chronos-2', cache_dir='~/.cache/huggingface/hub')
snapshot_download(repo_id='amazon/chronos-bolt-base', cache_dir='~/.cache/huggingface/hub')
"

# Find local paths
ls ~/.cache/huggingface/hub/models--autogluon--chronos-2/snapshots/
# Pick the actual hash directory, e.g., 60088152a34e...

# Step 2: Use LOCAL paths in AG (no download needed)
```

```python
LOCAL_C2 = '/Users/mac/.cache/huggingface/hub/models--autogluon--chronos-2/snapshots/60088152a34e242427b44c3100014473a0157d53/'
LOCAL_BOLT = '/Users/mac/.cache/huggingface/hub/models--amazon--chronos-bolt-base/snapshots/5d9f166d69f47aef3401367a7b842e78fe97b121/'

# Critical: also unset HF_ENDPOINT
import os
os.environ.pop('HF_ENDPOINT', None)

predictor.fit(
    ts_full,
    hyperparameters={
        "Chronos": {"model_path": LOCAL_BOLT},
        "Chronos2": {"model_path": LOCAL_C2},
    },
)
```

### Fine-Tuning Chronos-2 (LoRA by default)

```python
predictor.fit(
    ts_full,
    hyperparameters={
        "Chronos2": [
            {"ag_args": {"name_suffix": "ZeroShot"}},          # zero-shot
            {"fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},  # LoRA
        ],
    },
    time_limit=1800,  # Fine-tuning needs more time
)
```

**Caveat**: On Store Sales (large data), Chronos-2 fine-tuning timed out at 30min. Use `fine_tune_mode="lora"`, `fine_tune_steps=1000` for faster fine-tuning.

### Known Covariates vs Past Covariates

| Type | Example | AG 1.5 Support |
|------|---------|----------------|
| **Known covariates** (future known) | promotions, holidays, planned prices | ✅ Chronos-2 native |
| **Past covariates** (only historical) | weather observations, lagged indicators | ✅ Chronos2, TFT, DeepAR |

```python
predictor = TimeSeriesPredictor(
    known_covariates_names=['onpromotion', 'holiday', 'price'],  # future known
    # past_covariates_names=[...],  # only historical
)
```

### Common Pitfalls

1. **Forgetting `enable_ensemble=False`**: AG 1.5 will try to train all Chronos variants together, taking very long. For fastest experiments, set False.
2. **Not providing test data in TimeSeriesDataFrame**: Chronos-2 needs to know future covariate values during training/fitting.
3. **Wrong column name in merge**: TimeSeriesDataFrame uses `timestamp` not `date` after `reset_index()`.

---

## References

- [AG Timeseries Quick Start](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-quick-start.html)
- [AG Timeseries Models](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-models.html)
- [AG Forecasting with Chronos-2 (official tutorial)](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html)
- AG 1.4.0 TimeSeriesPredictor docstring (inspect.getsource)
- [Chronos-Bolt paper](https://arxiv.org/abs/2504.05291)
- [Chronos-2 paper](https://arxiv.org/abs/2510.15821)
- Store Sales competition: empirical validation (RMSLE LB 0.39525, 2026-07-10)
- AG GitHub: `autogluon/autogluon/forecasting-chronos.ipynb` (canonical Chronos-2 examples)