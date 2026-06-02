# Feature Engineering ROI

> When feature engineering beats hyperparameter tuning.

## The Rule

**When base models correlate > 0.93, only new features can break the stacking ceiling.**

This is empirically validated. Meta-learner parameter tuning gives diminishing returns at this point because all base models extract similar information from the data.

## When to Apply

Use this skill when:
- All base models have correlation > 0.93
- Multi-seed averaging gives no improvement
- Different meta-learner configs converge to same value
- 3+ tuning experiments yielded < 0.0001 gain

## The Procedure

```
1. Identify the "weakest" feature in your model
2. Analyze it domain-agnostically:
   - String features → character-level ordinal, hash, n-gram
   - Numeric features → log, square, ratio with siblings
   - Categorical → target encoding, frequency, pairwise interactions
3. Create a new base model with these engineered features
4. Verify: correlation with existing base models < 0.95
5. Add to stack — expect meaningful gain
```

## Worked Example: TPS May 2022 f_27

**Original situation**: All 14 base models correlated > 0.95. Stacking ceiling at 0.996.

**f_27 analysis**:
- String feature, 10 characters, domain A-T (base-20)
- All base models were using it identically (treated as categorical)

**Feature engineering**:
```python
# 10 ordinal features
f27_pos = [ord(c) - ord('A') for c in f27_str]

# Prefix hash (base-16)
f27_prefix6 = sum(ord(c) * (16**i) for i, c in enumerate(f27_str[:6]))

# Suffix hash (base-20)
f27_suffix4 = sum(ord(c) * (20**i) for i, c in enumerate(f27_str[6:]))

# Total
f27_sum = sum(f27_pos)
```

**Result**: New base model `lgb_f27` with correlation 0.92-0.94 to existing models. Adding to stack → +0.00158 AUC.

## Diagnostic: When Feature Engineering Will Help

```python
# Check current state
def should_engineer_features(base_models, threshold=0.93):
    corr_matrix = np.corrcoef([m.oof_preds for m in base_models])
    avg_corr = (corr_matrix.sum() - len(base_models)) / (len(base_models) * (len(base_models) - 1))
    return avg_corr > threshold
```

If True → time to engineer features, not tune models.

## When NOT to Apply

- Base models are diverse (correlation < 0.85)
- Data is already well-engineered
- Problem is image/audio (use architecture changes instead)
- No clear feature weakness to exploit

## Related Principles

- `local-optimum-trap` — recognize when you're stuck
- `stacking-ceiling` — formalize the limit
- `external-data-fusion` — sometimes the right move is to add new data, not engineer existing features