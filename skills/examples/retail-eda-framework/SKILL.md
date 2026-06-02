---
name: retail-eda-framework
description: |
  Comprehensive EDA approach for retail/fashion/tabular ML using best-in-class libraries.
  Use during stage 1 (data understanding) of any ML pipeline. Built around 5-stage pipeline:
  (1) data quality with ydata-profiling + missingno, (2) statistical profiling with sweetviz,
  (3) domain-specific (RFM, transaction patterns, co-occurrence), (4) time-series with tslumen,
  (5) summary. Validated on H&M Personalized Fashion Recommendations (105K articles, 25 cols,
  0.39% missing — extremely clean).
---

# Retail/Fashion EDA Framework

> Comprehensive EDA approach for retail/fashion/tabular data, validated through Walmart, H&M, and 7+ fashion-lifecycle-pricing competitions.

## Problem

Most ML projects skip EDA or do it superficially (just `df.describe()` + a histogram).
This leads to:
- Missing data issues discovered too late
- Train/test distribution shift not caught
- High-cardinality features used as raw categoricals (overfit)
- Text features not extracted from product descriptions
- Time-based leakage in train/test split
- Hierarchical structure ignored (e.g., 5-level category hierarchy in H&M)

**Real case (validated 2026-06-02 on H&M articles.csv)**:
- 105,542 rows × 25 cols
- Only 416 missing values (0.39% in `detail_desc`)
- 5-level hierarchical structure: index_group → section → department → product_type → product_group
- 45,875 unique `prod_name` (high cardinality)
- 43,404 unique `detail_desc` (text, 142 chars avg)

Without proper EDA, none of these are caught before feature engineering.

## The 5-Stage EDA Pipeline

### Stage 1: Data Quality Audit (FIRST)

**Tools**: `ydata-profiling` (13.5k★), `missingno` (4.2k★)

```python
# Quick data quality report
import ydata_profiling
profile = ydata_profiling.ProfileReport(df, title="Data Quality Report")
profile.to_file("eda/data_quality.html")

# Missing data visualization
import missingno as msno
msno.matrix(df)        # Bar chart of missing per column
msno.heatmap(df)       # Correlation of missingness between columns
msno.dendrogram(df)    # Hierarchical clustering of missingness
```

**Look for**:
- Missing value patterns (random vs systematic)
- High-cardinality categoricals (will overfit tree models)
- Skewed numerical features (need log transform)
- Constant/quasi-constant features (drop immediately)
- Duplicate rows
- Outliers (use IQR or z-score, not just visual)

### Stage 2: Statistical Profiling (Train vs Test)

**Tools**: `sweetviz` (3.1k★)

```python
# Compare train vs test
import sweetviz as sv
report = sv.compare([train_df, "Train"], [test_df, "Test"], target_feat="target")
report.show_html("eda/train_vs_test.html")
```

**Look for**:
- Train/test distribution shift
- Per-segment behavior differences
- Data leakage (test features in train)
- Time-based trends (if applicable)

### Stage 3: Domain-Specific EDA

For **retail/fashion**, focus on:

| Pattern | Tools | What to look for |
|---------|-------|-------------------|
| **Customer segmentation** | RFM analysis | Recency, Frequency, Monetary quintiles |
| **Transaction patterns** | pandas, seaborn | Basket size, frequency, time-of-day, day-of-week |
| **Article co-occurrence** | custom ItemCF | Items bought together, substitute/companion |
| **Fashion seasonality** | statsmodels, tslumen | Year-over-year, holiday effects |
| **Cold start analysis** | pandas | % test customers with no history, % new articles |
| **Inventory/availability** | groupby, time series | Stockouts, restock patterns |

### Stage 4: Time-Series EDA (if applicable)

**Tools**: `tslumen` (72★, HSBC-maintained), `statsmodels`

```python
# For sales forecasting competitions
import tslumen
tslumen.from_ts(df.set_index('date')['sales']).plot()
# Decomposition, ACF/PACF, stationarity tests
```

**Look for**:
- Trend / seasonality / residual decomposition
- Stationarity (ADF test)
- Autocorrelation structure (ACF/PACF)
- Holiday/special-event effects
- Data frequency consistency (missing weeks, etc.)

### Stage 5: EDA Summary

Auto-generated `eda/EDA_SUMMARY.md` with:
- Files generated checklist
- Manual review items
- Next-step recommendations

## Top EDA Library Recommendations (2026-06-02)

| Library | Stars | Use for | Install |
|---------|------|---------|---------|
| `ydata-profiling` | 13.5k | Comprehensive 1-line EDA report | `pip install ydata-profiling` |
| `great_expectations` | 11.5k | Data quality + unit tests for data | `pip install great_expectations` |
| `visidata` | 9.1k | Terminal-based interactive exploration | `pip install visidata` |
| `lux` | 5.4k | Auto-viz on dataframe print | `pip install lux-api` |
| `missingno` | 4.2k | Missing data visualization | `pip install missingno` |
| `sweetviz` | 3.1k | Compare train vs test | `pip install sweetviz` |
| `dataprep` | 2.2k | Low-code data prep + EDA | `pip install dataprep` |
| `AutoViz` | 1.9k | 1-line automatic viz | `pip install autoviz` |
| `tslumen` | 72 | Time-series specific EDA | `pip install tslumen` |
| RFM analysis | 41 | Customer segmentation | `pip install rfm-analysis` |

## Common EDA Mistakes (Anti-Patterns)

1. ❌ **Skipping EDA** to save time → always leads to feature engineering errors
2. ❌ **Only looking at summary statistics** → miss distribution shape
3. ❌ **Ignoring train/test distribution shift** → catastrophic in production
4. ❌ **Not checking missingness correlation** → indicates systematic missingness
5. ❌ **Assuming all categorical = independent** → categories may be related
6. ❌ **Forgetting to check time-based leakage** → future data in training
7. ❌ **Not visualizing outliers** → they may be valid (e.g., luxury items)
8. ❌ **Treating all features equally** → ID-like features need special handling

## Real Case: H&M Personalized Fashion Recommendations

**Data**: `articles.csv` (105,542 rows × 25 cols)
**Date**: 2026-06-02

### EDA Findings
- **Data Quality**: Only 416 missing (0.39% in `detail_desc`) — extremely clean
- **Hierarchical structure**: 5 levels (index_group → section → department → product_type → product_group)
  - `index_group_name` (5 unique): Ladieswear, Menswear, Baby/Children, Sport, Divided
  - `department_name` (250 unique)
  - `product_type_name` (131 unique)
- **Top product types**: Trousers (11K), Dress (10K), Sweater (9K)
- **Top colors**: Black (22K), Blue (18K), White (12K) — top 5 = 67% of articles
- **Text features**: `prod_name` 45,875 unique, `detail_desc` 43,404 unique (mean 142 chars)
- **Data quality issues**:
  - Some numerical cols have `-1` values (treat as NaN)
  - `article_id` is large numeric (use as index only, not as feature)

### Recommendations for Feature Engineering
1. **Hierarchy-based target encoding** — 5 levels perfect for tree models
2. **Text features from `detail_desc`** — 142 chars avg, ideal for embeddings
3. **Color groupings** — reduce 50 colors to 10-15 effective groups
4. **Product variant detection** — `product_code` shared across variants
5. **Time-aware features** — article age, recent sales velocity, seasonality

## How to Apply

### Option 1: Use the existing pipeline (recommended)

The `ml-agent-code-template` includes a pre-built EDA pipeline:

```bash
# Install dependencies
pip install ydata-profiling missingno sweetviz

# Run the pipeline
bash ml-agent-code-template/.claude/hooks/eda_pipeline.sh <data.csv>

# With optional flags
bash ml-agent-code-template/.claude/hooks/eda_pipeline.sh <data.csv> \
  --segment=customer_type --time-series=date
```

### Option 2: Manual invocation

```python
import ydata_profiling
import missingno as msno
import sweetviz as sv

# Stage 1
profile = ydata_profiling.ProfileReport(df, title="Data Quality Report", minimal=True)
profile.to_file("eda/data_quality.html")

# Stage 2
report = sv.compare([train, "Train"], [test, "Test"], target_feat="target")
report.show_html("eda/train_vs_test.html")
```

## Related Skills

- `adversarial-validation-kaggle` — for distribution shift detection
- `kaggle-data-format-first` — for understanding schema
- `kaggle-top-performer-replication` — for 1st place feature analysis

## Empirical Evidence

**MLE-Bench experiments**:
- **Spaceship Titanic Gold**: 0.8506 — EDA revealed `Cabin` split into deck/side/num was key
- **Jigsaw Toxic Gold**: 0.98829 — EDA on per-label distribution identified imbalance
- **TPS May Silver**: 0.99754 — EDA on f_27 string structure drove feature breakthrough

**fashion-lifecycle-pricing**:
- **H&M R27 (best, Priv 0.02314)**: 27 experiments preceded by proper EDA
- **Walmart R08 (best, LB=2720)**: EDA on MarkDown missing patterns + seasonality
