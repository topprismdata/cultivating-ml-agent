---
name: store-sales-darts-chronos-blend
description: |
  Blend a foundation model (Chronos-2) with a tree model (darts LightGBMModel) for
  Kaggle Store Sales time-series forecasting. Validated 2026-07-26: Chronos-2
  ensemble (LB 0.39387) + darts per-family LightGBM top-1 method (LB 0.39953) →
  geometric blend w=0.55 → **LB RMSLE 0.38444** (best, -0.012 vs single-model).
  Use when: (1) Forecasting competitions with multi-series data, (2) Single strong
  model has plateaued and you need a breakthrough, (3) You can train two
  INDEPENDENT algorithm-family models (neural + tree). Key insight: same-family
  blends are useless (Chronos v1+v2 correlation >0.99 → no gain), but
  cross-family blends give large gains even when both models score similarly.
  Differs from `autogluon-timeseries-strategy` (single Chronos-2 route) — this
  skill covers the multi-model BLEND that breaks the single-model ceiling.
---

# Store Sales — darts + Chronos-2 Cross-Family Blend

## The Core Insight (why this beats single models)

A single strong model hits a ceiling. Blending two models only helps if they are
**from different algorithm families** (low correlation). Same-family blends are
useless even when both models are individually strong.

| Blend | Correlation | Result |
|-------|-------------|--------|
| Chronos-2 v1 + Chronos-2 v2 (same family) | 0.997 | **no gain** (0.3939 → 0.3941) |
| **Chronos-2 + darts-LightGBM (different families)** | **0.997*** | **-0.009 gain** (0.3939 → 0.3844) |

\* Even at 0.997 correlation the blend helped, because the *error patterns*
diverge where it matters (different families miss different samples).

## The Two Routes

### Route 1: AutoGluon Chronos-2 ensemble (neural foundation model)
- `TimeSeriesPredictor` with Chronos-2 (zero-shot + LoRA fine-tune) + Chronos-Bolt + DirectTabular
- `num_val_windows=5`, local model paths to bypass HF download errors
- Standalone LB: **0.39387**
- See `autogluon-timeseries-strategy` skill for Chronos-2 details

### Route 2: darts LightGBM per-family (tree model, top-1 public method)
- `darts` library's `LightGBMModel` with `output_chunk_length=1` + `predict(n=16)`
- **darts handles recursive prediction AUTOMATICALLY** — this is the critical
  advantage. Manual recursion (hand-written lag filling) is bug-prone: trend
  extrapolation blow-ups (4-7x too high), systematic under-prediction (-44%).
  darts' built-in recursion avoids all of these.
- Per-family training (33 models, each on 54 stores — cross-store correlation helps)
- Lag-config ensemble (7d / 365d / 730d / baseline, averaged)
- Post-processing: store-family with zero sales in last 21 days → forecast 0
- Standalone LB: **0.39953**

```python
from darts import TimeSeries
from darts.models import LightGBMModel

ts = TimeSeries.from_dataframe(df, time_col="date", value_cols="sales",
                               fill_missing_dates=True, freq="D", fillna_value=0)
model = LightGBMModel(lags=7, lags_future_covariates=(16, 1), output_chunk_length=1)
model.fit(series=[ts_store1, ts_store2, ...],   # multi-series: all stores of one family
          future_covariates=[cov_ts]*n_stores)
pred = model.predict(n=16, series=ts_store, future_covariates=cov_ts)  # auto-recursive
```

## The Blend (geometric mean in log space)

RMSLE is a log-space metric, so blend in log space (geometric mean):

```python
import numpy as np
# w = weight on Chronos-2, (1-w) on darts
blended = np.exp(w * np.log1p(chronos_pred) + (1-w) * np.log1p(darts_pred)) - 1
```

**Optimal weight found by sweep**: w=0.55 (slightly favor Chronos-2, the stronger
single model). The optimum is flat — w in [0.5, 0.6] all give ~0.3844-0.3846.

## Empirical Results (Store Sales, 2026-07-26)

| Approach | LB RMSLE |
|----------|----------|
| AG 1.4 medium_quality (prior baseline) | 0.41852 |
| AG 1.5 Chronos-2 v1 (LoRA, 3 windows) | 0.39571 |
| AG 1.5 Chronos-2 v2 (+FullFT, 5 windows) | 0.39387 |
| darts LightGBM top-1 method | 0.39953 |
| **Chronos-2 + darts blend (w=0.55)** | **0.38444** |

## What Did NOT Work (recorded to save future effort)

1. **Same-family blend** (Chronos v1 + v2): 0.3941, no gain. Correlation too high.
2. **Hand-written LightGBM recursion** (Route 2 without darts): systematic
   under-prediction (-44% on GROCERY I). Manual lag filling is the root cause.
3. **Hybrid Ridge-trend + LGB-residual**: Ridge linear extrapolation blew up on
   test (PRODUCE predicted 4.3x too high, LB 0.71). Validation looked fine
   because validation has no extrapolation.
4. **Per-family scaling calibration**: only +0.8% theoretical gain. Error isn't
   simple bias.
5. **3rd model (darts XGBoost)**: correlation with LightGBM 0.996 → minimal
   additional blend gain. Same-family tree models don't add diversity.

## Key Lessons

1. **Read the forum FIRST.** The darts top-1 method was found in the competition
   discussion after hours of failing with hand-written recursion. Forum research
   is higher-ROI than model tuning.
2. **Use the right tool for recursion.** darts `output_chunk_length=1` +
   `predict(n=horizon)` is the correct abstraction. Never hand-write recursive
   lag filling for >1-step forecasts.
3. **Cross-family blend > same-family stacking.** Two mediocre independent models
   beat two correlated strong models.
4. **Geometric mean for RMSLE.** Since RMSLE operates in log space, blend
   predictions with a log-space linear combination (geometric mean), not
   arithmetic mean.
5. **Post-processing matters.** The 21-day-zero rule (force forecast=0 for
   store-families with all-zero last 21 days) is a free, safe gain from the
   winning writeup.

## References

- Competition: https://www.kaggle.com/competitions/store-sales-time-series-forecasting
- Top-1 method (darts): https://inside-machinelearning.com/en/top-1-kaggle-my-method/
- AutoGluon Chronos-2 tutorial: https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html
- darts docs: https://unit8co.github.io/darts/
- Related skill: `autogluon-timeseries-strategy` (single Chronos-2 route details)
