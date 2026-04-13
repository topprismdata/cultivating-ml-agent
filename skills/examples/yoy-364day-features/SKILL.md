---
name: yoy-364day-features
description: |
  Use when: (1) building time series features for data with annual seasonality,
  (2) working with retail/sales forecasting where year-over-year patterns matter,
  (3) looking for features that capture long-term seasonal patterns beyond short lags,
  (4) the 1st place solution mentions "364-day" or "YoY" features.
  NOT for: data without annual seasonality, very short history (< 1 year).
---

# YoY 364-Day Features for Time Series

## Core Idea

Use sales/values from exactly 364 days ago (52 weeks) as features. This aligns
day-of-week perfectly (364 = 52 × 7), capturing annual seasonality while maintaining
weekly structure.

## Feature Set (4 features)

```python
# For each (store, family) on reference date t:
yoy_sales_364 = sales[t - 364]                          # Same day-of-week last year
yoy_sales_364_7d_avg = mean(sales[t-370 : t-364])       # 7-day avg around same week last year
yoy_ratio_1y = sales_lag_1 / (yoy_sales_364 + 1)        # Current vs last year momentum
yoy_ratio_7d = rolling_mean_7 / (yoy_sales_364_7d_avg + 1)  # Recent trend vs last year
```

## Implementation

```python
# Build YoY lookup: map each date to sales from 364 days ago
sales_df = train_raw[["store_nbr", "family", "date", "sales"]].copy()
sales_df["yoy_date"] = sales_df["date"] + pd.Timedelta(days=364)

yoy_lookup = sales_df.rename(columns={
    "date": "yoy_ref_date", "sales": "yoy_sales_364"
})[["store_nbr", "family", "yoy_ref_date", "yoy_sales_364"]]

# Merge into training data
merged = merged.merge(
    yoy_lookup, left_on=["store_nbr", "family", "date"],
    right_on=["store_nbr", "family", "yoy_ref_date"], how="left"
)

# 7-day average around same week last year
yoy_7d = sales_df.groupby(["store_nbr", "family"]).apply(
    lambda g: g.set_index("date")["sales"].rolling(7, min_periods=1).mean().shift(364)
).reset_index()
```

## Evidence

Kaggle Store Sales (Favorita), April 2026:

| Version | Features | LB |
|---------|----------|--------|
| R11c (baseline) | 82 base | 0.39824 |
| R12 multilevel | +18 family/store aggregation | 0.39874 (+0.00050 worse) |
| **R13 YoY** | **+4 YoY 364-day** | **0.39779 (-0.00045 better)** |

Feature importance: `yoy_sales_364_7d_avg` = 2577 (high), `yoy_sales_364` = 1204.

## Key Insight

Why 364 and not 365? Because 364 = 52 × 7, so day_of_week is guaranteed to match.
This matters for retail data where weekday/weekend sales differ dramatically.
Using 365 would shift day-of-week by 1 (or 2 for leap years), introducing noise.

## Related Skills

- `rmsle-zero-threshold-asymmetry` — post-processing for RMSLE metrics
- `controlled-submission-experiment` — isolate variable effects
- `ts-day-specific-forecasting` — the framework these features are used in
