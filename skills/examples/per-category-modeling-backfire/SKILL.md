---
name: per-category-modeling-backfire
description: |
  Per-category/per-family models can produce WORSE leaderboard scores than a single global model,
  even when per-category CV is better. Use when: (1) Considering training separate models per
  product category/family/store in tabular competitions, (2) Per-category CV improves but LB
  degrades, (3) Each category has <100K rows from a larger dataset. Covers data volume thresholds,
  hybrid fallback strategies, and when per-category modeling is appropriate vs counterproductive.
---

# Per-Category Modeling Backfire: When Splitting Data Hurts

## Problem

A common "advanced" technique in Kaggle competitions is to train separate models for each
category (e.g., per product family, per store, per region). While this seems intuitively
correct—different categories have different patterns—it can backfire spectacularly when
individual categories lack sufficient data volume.

## Context / Trigger Conditions

Use when considering:
- Training separate GBDT models per product family, store, or region
- Per-category CV improves but leaderboard score degrades
- Each category has <100K rows but the full dataset has >1M rows
- Categories have highly imbalanced data volumes (some huge, some tiny)
- The global model already handles category differences via categorical features

**Specific symptoms**:
- Per-category CV: 0.367 (better) vs Global CV: 0.375 → Per-category LB: 2.10 vs Global LB: 1.86
- Small categories show high CV variance (>0.05 std across folds)
- Test predictions from per-category models have shifted mean (higher or lower than expected)

## Solution

### The Data Volume Threshold

Per-category modeling only works when each category has enough data to train a robust model.

| Rows per category | Per-category modeling | Recommendation |
|-------------------|----------------------|----------------|
| <50K | **Harmful** — overfitting, high variance | Use global model |
| 50K-200K | Risky — depends on problem complexity | Test both, compare LB |
| 200K-1M | Usually beneficial | Per-category if CV confirms |
| >1M | Almost always beneficial | Per-category recommended |

### Why It Fails With Small Categories

1. **Overfitting**: 70K rows × 5-fold CV = 14K validation rows. Easy to overfit.
2. **High variance**: Small categories (BOOKS, HARDWARE) have CV std >0.05, meaning
   the model is unstable across folds.
3. **Feature instability**: Lag/rolling features with limited history don't generalize.
4. **Loss of cross-category signal**: Global model learns that "weekend → more sales"
   applies across ALL categories, amplifying this signal. Per-category models each
   must re-learn this from limited data.

### Hybrid Approach Also Fails

A natural fallback—"use per-category for large categories, global for small"—also
didn't work in practice:

```
Global model: LB = 1.86
Per-family (33 models): LB = 2.10 (worse)
Hybrid (15 per-family + global): LB = 2.13 (even worse!)
```

The hybrid fails because:
1. Predictions from per-family and global models have different scales/biases
2. No principled way to calibrate between the two prediction sources
3. The boundary (large vs small threshold) introduces discontinuities

### When Per-Category DOES Work

Per-category modeling is appropriate when:
1. Categories have genuinely different data-generating processes
2. Each category has >200K rows
3. The global model cannot capture category-specific patterns (no categorical features)
4. Categories have different feature importance rankings

## Verification

Before committing to per-category modeling:

```python
# 1. Check data volume per category
for cat in categories:
    n = len(train[train['category'] == cat])
    if n < 100000:
        print(f"WARNING: {cat} has only {n} rows - per-category model may overfit")

# 2. Check CV stability per category
for cat in categories:
    cv_scores = cross_validate(cat_data)
    if np.std(cv_scores) > 0.03:
        print(f"WARNING: {cat} has unstable CV (std={np.std(cv_scores):.4f})")

# 3. Compare global vs per-category on HOLDOUT (not OOF)
global_pred = global_model.predict(test)
per_cat_pred = per_category_models.predict(test)
# If mean(per_cat_pred) differs significantly from mean(global_pred), per-category is suspect
```

## Example

**Kaggle Store Sales Time Series Forecasting** — 54 stores × 33 families:

```
Dataset: 2.35M clean rows, 1782 store-family combinations
Per-family: 33 families × ~71K rows each

Results:
├─ Global LightGBM:    CV=0.375, LB=1.859 ✅ Best
├─ Per-family (33):    CV=0.367, LB=2.102 ❌ 13% worse
└─ Hybrid (15+global): CV=—,    LB=2.129 ❌ 15% worse

Analysis:
- Top 5 families (GROCERY I, BEVERAGES, PRODUCE, CLEANING, DAIRY):
  CV 0.108-0.245, each >100K rows worth of sales → per-family helps
- Bottom 5 families (LINGERIE, CELEBRATION, GROCERY II, HARDWARE, AUTOMOTIVE):
  CV 0.450-0.605, each with tiny sales → per-family hurts badly
- The bad families dragged the overall LB down more than the good families helped

Root cause: 71K rows per family is below the threshold for robust GBDT training.
The global model's 2.35M rows provide much more stable estimates.
```

## Notes

### Related to Other Skills

- **ml-sweet-spot**: "More is not always better" — per-category is "more models" but not always better
- **small-dataset-optimization-limits**: Per-category creates small datasets from a large one
- **ensemble-model-correlation-trap**: Hybrid approach combines correlated predictions poorly
- **ts-lag-nan-cascade-bug**: The ffill fix that made the global model work well

### Why TOP Solutions Use Per-Family

Many top Kaggle solutions DO use per-family models successfully. The difference:
1. They use **linear models** (Ridge/Lasso) for small families, not GBDT
2. They have **more sophisticated features** per family
3. They use **recursive prediction** to handle lag features properly
4. Their per-family models are often **simpler** (fewer trees, more regularization)

### Key Insight

**Per-category modeling is a sophistication that requires sufficient data per category.
If each category has <100K rows, the global model's data advantage outweighs the
specialization benefit.**

## References

- Kaggle Store Sales Competition: Global model LB=1.86, Per-family LB=2.10
- Web search confirms per-family is common winning strategy but typically with
  larger per-category datasets or simpler models (linear regression)
