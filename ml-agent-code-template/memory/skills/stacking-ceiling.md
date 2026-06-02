# Stacking Ceiling

> Recognize when stacking stops helping.

## The Rule

**When all base models have correlation > 0.93 AND AUC > 0.99, stacking adds < 0.001 AUC.**

The mathematical reason: the meta-learner is just averaging highly correlated predictors. No new information is extracted.

## Diagnostic

```python
def check_stacking_ceiling(base_models, target_metric=0.99, corr_threshold=0.93):
    """Returns True if you've hit the stacking ceiling."""
    preds = np.array([m.oof_preds for m in base_models])
    n = len(base_models)

    # Pairwise correlation
    corr_matrix = np.corrcoef(preds)
    upper_triangle = corr_matrix[np.triu_indices(n, k=1)]
    avg_corr = upper_triangle.mean()

    # All models above threshold
    all_strong = all(m.score > target_metric for m in base_models)

    return avg_corr > corr_threshold and all_strong
```

If this returns True, you're at the ceiling.

## What Breaks the Ceiling

Only structurally new information:
1. **New base model with correlation < 0.85** (new data, new features, new architecture)
2. **External data** that wasn't in the original feature set
3. **Different problem framing** (different target, different loss, different metric)

What does NOT break the ceiling:
- ❌ More meta-learner parameter tuning
- ❌ Multi-seed averaging
- ❌ Cross meta-learner blending
- ❌ Stacking more rounds

## Empirical Evidence

**TPS May 2022** (2026-05-17):
- 14 base models, all AUC > 0.99, average correlation 0.95
- Greedy forward selection: 4 models → 0.9959, 14 models → 0.9952
- 20-config comprehensive search → 0.9952
- 9-config × 3-seed averaging → 0.9952 (identical)
- **Conclusion**: confirmed at ceiling

**Solution that worked**:
- f_27 character features → new base model with correlation 0.92
- New stack: 0.997540 (+0.00158)

## Strategy

```
1. Build initial 4-6 base models
2. Check correlation matrix
3. If avg corr > 0.93:
   a. Stop tuning meta-learner
   b. Engineer features OR add external data
   c. Build new base model from new info
4. Add to stack
5. Re-check ceiling
```

## Anti-Patterns

- ❌ Tuning meta-learner for hours when ceiling is reached
- ❌ Adding more correlated base models hoping for averaging
- ❌ Multi-seed averaging to "stabilize" — it's already stable, just limited
- ❌ Trying different meta-learner types (LR, LGB, NN) — all converge

## Related Skills

- `feature-engineering-roi` — what to do at the ceiling
- `external-data-fusion` — alternative to feature engineering
- `local-optimum-trap` — broader principle