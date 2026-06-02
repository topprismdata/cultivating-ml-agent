# Cross-Validation Strategy

> Choosing the right CV splits for your problem.

## The Rule

**Match your CV strategy to your data structure. Wrong CV = wrong model selection.**

## Strategies by Data Type

### IID Tabular (random split OK)
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```
Use for: classification with random splits, no temporal or group structure.

### Time Series (temporal split)
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
# Train: t, t+1, ..., t+k
# Val:   t+k+1
```
Use for: any time-dependent data, no future information leakage.

### Group Structure (molecules, users, sessions)
```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
# Group column must be specified
for fold, (trn, val) in enumerate(gkf.split(X, y, groups=groups)):
    ...
```
Use for: CHAMPS (molecules), user behavior, anything where one entity appears multiple times.

### Stratified Group (rare)
```python
# When you need both group separation AND class balance
from sklearn.model_selection import StratifiedGroupKFold

# Only in sklearn >= 1.1
```

## Diagnostic: Did I use the right CV?

```python
def check_leakage(train, test, group_col=None, time_col=None):
    # Check 1: Group overlap
    if group_col:
        train_groups = set(train[group_col])
        test_groups = set(test[group_col])
        overlap = train_groups & test_groups
        if overlap:
            print(f"⚠️ {len(overlap)} groups in both train and test")
            return "USE_GROUP_KFOLD"

    # Check 2: Time gap
    if time_col:
        train_max = train[time_col].max()
        test_min = test[time_col].min()
        if test_min < train_max:
            print("⚠️ Test contains dates from training period")
            return "USE_TIME_SERIES_SPLIT"

    return "IID_OK"
```

## Empirical Evidence

**CHAMPS Scalar Coupling** (2026-05-04):
- Wrong: regular KFold on molecules → 1JHC OOF MAE = 5.55, Priv MAE = 18.43
- Right: GroupKFold by molecule_name → OOF/Priv aligned
- **Lesson**: without proper grouping, OOF/Priv gap explodes

**Spaceship Titanic** (2026-04-28):
- Standard StratifiedKFold worked fine
- No group structure (each passenger is unique)

## Common Mistakes

1. **Random KFold on time series** → trains on future, leaks
2. **Regular KFold on grouped data** → same group in train and val
3. **Stratified without considering target distribution shift** → wrong calibration
4. **Too many folds** → slower, sometimes worse (high variance per fold)

## When to Use Repeated CV

```python
from sklearn.model_selection import RepeatedStratifiedKFold

# For small datasets where 5-fold is too noisy
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
```

Use when:
- Dataset < 10K rows
- 5-fold OOF score is unstable
- You have time to wait (3x slower)

Don't use when:
- Dataset is large (variance is already small)
- CV time is the bottleneck

## Quick Reference

| Data Type | CV Strategy |
|-----------|-------------|
| Tabular IID | StratifiedKFold(5) |
| Time series | TimeSeriesSplit(5) |
| Groups | GroupKFold(5) |
| Small dataset | RepeatedStratifiedKFold(5, 3) |
| Multi-class | StratifiedKFold(5) |
| Regression | KFold(5) |
| Imbalanced binary | StratifiedKFold(5) + class weights |