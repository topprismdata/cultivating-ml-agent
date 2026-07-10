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

## Presets (Different from Tabular!)

| Preset | Models | Time | Best For |
|--------|--------|------|----------|
| `fast_training` | Statistical + tree-based (no DL) | 1-2 min | Quick baseline |
| **`medium_quality`** | Above + TemporalFusionTransformer + Chronos-Bolt small | 5-15 min | **Recommended starting point** |
| `high_quality` | DL + ML + statistical mix | 30-60 min | Strong forecast |
| `best_quality` | All + multi-window backtests | 1-4 hours | Maximum accuracy |
| `bolt_tiny/mini/small/base` | Chronos-Bolt pretrained only | 1-5 min | Zero-shot |

**Note**: Timeseries presets are NOT the same as Tabular presets! Tabular's `good_quality` ≠ Timeseries' `good_quality`.

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

## Models Trained by AG TimeSeries

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
| AutoCES | Complex Exponential Smoothing | ~10s | Statistical baseline |

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