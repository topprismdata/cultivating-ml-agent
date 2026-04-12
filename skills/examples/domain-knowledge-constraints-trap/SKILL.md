---
name: domain-knowledge-constraints-trap
description: |
  Avoid applying domain knowledge constraints that hurt ML model performance. Use when:
  (1) Considering medical/physical/logical constraints on training data, (2) Adversarial
  validation AUC changes significantly after applying constraints (>0.10 shift),
  (3) CV score drops after adding "reasonable" domain rules, (4) Train/test
  distributions become misaligned after filtering.
---

# Domain Knowledge Constraints Trap

## Problem

Applying domain knowledge "constraints" or "rules" to filter training data can
seem like a good idea for improving model quality, but often **destroys model
performance** by changing the data distribution and removing valuable information.

**Real Case**: Adding 4 medical constraints to heart disease prediction:
- Expected: Improve model with medical domain knowledge
- Actual: CV AUC dropped from 0.970 to 0.943 (-2.8%)
- Root cause: Adversarial AUC changed from 0.501 → 0.661 (distribution shift)

## Context / Trigger Conditions

**Use this skill when**:
- Working on ML projects with domain experts providing "rules"
- Considering filtering data based on physical/medical/logical constraints
- Adversarial validation shows train/test distribution change after filtering
- CV score decreases after adding "reasonable" constraints

**Symptoms**:
- Domain experts say "this data point is impossible"
- Filtering "anomalies" or "outliers" based on domain rules
- CV score drops after adding constraints
- Train/test distributions become different (adversarial AUC ≠ 0.5)

**Common Trap Examples**:
- **Medical**: "Heart rate can't exceed 220 - age" → Removes valid extreme cases
- **Physical**: "Temperature can't be negative" → Removes sensor errors AND valid extremes
- **Business**: "Customer can't spend >$10K/month" → Removes high-value outliers
- **Temporal**: "Events can't happen in the future" → Removes data entry errors AND valid edge cases

## Solution

### Step 1: Quantify Distribution Impact

**Before applying constraints**, check adversarial validation:

```python
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

def check_adversarial_auc(train_df, test_df, features):
    """Check if train/test distributions are similar"""
    adv_train = train_df[features].copy()
    adv_train['is_test'] = 0

    adv_test = test_df[features].copy()
    adv_test['is_test'] = 1

    adv_combined = pd.concat([adv_train, adv_test], axis=0)

    # Train classifier to distinguish train vs test
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(adv_combined))

    for tr, val in skf.split(adv_combined, adv_combined['is_test']):
        train_data = lgb.Dataset(adv_combined.iloc[tr][features],
                                  label=adv_combined.iloc[tr]['is_test'])
        val_data = lgb.Dataset(adv_combined.iloc[val][features],
                                label=adv_combined.iloc[val]['is_test'])

        model = lgb.train({'objective': 'binary', 'verbosity': -1},
                          train_data, num_boost_round=100,
                          valid_sets=[val_data],
                          callbacks=[lgb.early_stopping(stopping_rounds=10)])

        oof_pred[val] = model.predict(adv_combined.iloc[val][features])

    auc = roc_auc_score(adv_combined['is_test'], oof_pred)
    return auc

# Before constraints
auc_before = check_adversarial_auc(train, test, features)
print(f"Adversarial AUC (before): {auc_before:.5f}")
# Output: 0.50111 ≈ 0.5 (distributions are similar) ✅
```

### Step 2: Apply Constraints Cautiously

If you MUST apply constraints:

```python
def apply_constraints_with_validation(df, constraints):
    """Apply constraints and measure impact"""
    df_filtered = df.copy()
    removed = []

    for constraint_name, constraint_func in constraints:
        before = len(df_filtered)
        df_filtered = constraint_func(df_filtered)
        after = len(df_filtered)

        removed_count = before - after
        removed.append((constraint_name, removed_count))
        print(f"{constraint_name}: removed {removed_count} samples")

    total_removed = len(df) - len(df_filtered)
    print(f"Total removed: {total_removed} ({100*total_removed/len(df):.1f}%)")

    # Check adversarial AUC after filtering
    auc_after = check_adversarial_auc(df_filtered, test, features)
    print(f"Adversarial AUC (after): {auc_after:.5f}")

    # Warn if distribution changed significantly
    if abs(auc_after - 0.5) > 0.10:
        print("⚠️ WARNING: Distribution changed significantly!")
        print("   Constraints may be removing test-like samples.")
        print("   Consider NOT applying these constraints.")

    return df_filtered, removed
```

### Step 3: Validate Impact on Model Performance

**Always train with and without constraints to compare**:

```python
# Train WITHOUT constraints (baseline)
model_baseline = train_model(train, features)
cv_baseline = cross_val_score(model_baseline, X, y, cv=5, scoring='roc_auc')
print(f"CV (no constraints): {cv_baseline.mean():.5f}")

# Apply constraints
train_constrained, removal_stats = apply_constraints_with_validation(train, constraints)

# Train WITH constraints
model_constrained = train_model(train_constrained, features)
cv_constrained = cross_val_score(model_constrained,
                                  X_constrained, y_constrained,
                                  cv=5, scoring='roc_auc')
print(f"CV (with constraints): {cv_constrained.mean():.5f}")

# Compare
if cv_constrained.mean() < cv_baseline.mean():
    print("❌ Constraints HURT performance!")
    print(f"   Drop: {cv_baseline.mean() - cv_constrained.mean():.5f}")
else:
    print("✅ Constraints HELP performance")
```

### Step 4: Decision Framework

**Should you apply domain constraints?**

| Condition | Action |
|-----------|--------|
| Adversarial AUC change < 0.05 | ✅ Safe to apply |
| Adversarial AUC change 0.05-0.10 | ⚠️ Proceed with caution |
| Adversarial AUC change > 0.10 | ❌ DO NOT apply |
| CV score drops | ❌ DO NOT apply |
| Removes >20% of data | ❌ DO NOT apply |
| Removes >5% of data | ⚠️ Validate carefully |

**Real Example**:
```
Adversarial AUC before constraints: 0.501 ✅
Adversarial AUC after constraints:  0.661 ❌ (Distribution changed!)
CV before constraints:              0.970
CV after constraints:               0.943 (-2.8%)

Conclusion: DO NOT apply these medical constraints
```

## Verification

**Good Signs** (constraints are safe):
- Adversarial AUC stays close to 0.5 (±0.05)
- CV score improves or stays same
- Removed samples are true errors (verified by domain expert)
- Train/test distributions remain similar

**Bad Signs** (constraints are harmful):
- Adversarial AUC moves away from 0.5 (>0.10)
- CV score drops significantly
- Many "edge case" samples are removed
- Train/test distributions become different

## Example

**Medical Constraints for Heart Disease**:

```python
def apply_medical_constraints(df):
    """Apply 4 medical domain constraints"""
    df_filtered = df.copy()

    # Constraint 1: Max HR ≤ 220 - Age
    valid_hr = df_filtered['Max HR'] <= (220 - df_filtered['Age'])
    removed_1 = (~valid_hr).sum()
    df_filtered = df_filtered[valid_hr]
    print(f"Heart rate constraint: removed {removed_1} samples")

    # Constraint 2: Young low-risk patients don't have high vessel count
    young_high_risk = (
        (df_filtered['Age'] < 40) &
        (df_filtered['Vessels'] >= 3) &
        (df_filtered['Cholesterol'] < 160)
    )
    removed_2 = young_high_risk.sum()
    df_filtered = df_filtered[~young_high_risk]
    print(f"Young high-risk constraint: removed {removed_2} samples")

    # Constraint 3: ST depression consistency
    st_inconsistent = (
        (df_filtered['ST_depression'] > 2) &
        (df_filtered['ST_slope'] == 1)
    )
    removed_3 = st_inconsistent.sum()
    df_filtered = df_filtered[~st_inconsistent]
    print(f"ST consistency constraint: removed {removed_3} samples")

    # Constraint 4: Thallium-Vessels consistency
    thal_vessels_inconsistent = (
        (df_filtered['Thallium'] == 7) &
        (df_filtered['Vessels'] > 0)
    )
    removed_4 = thal_vessels_inconsistent.sum()
    df_filtered = df_filtered[~thal_vessels_inconsistent]
    print(f"Thallium-Vessels constraint: removed {removed_4} samples")

    total_removed = len(df) - len(df_filtered)
    print(f"Total: {total_removed} samples removed ({100*total_removed/len(df):.1f}%)")

    return df_filtered

# Apply constraints
train_constrained = apply_medical_constraints(train)

# Check impact on distribution
auc_before = 0.501  # Train/test are similar
auc_after = check_adversarial_auc(train_constrained, test, features)
# Result: 0.661 ❌ Distribution changed!

# Check impact on model performance
cv_before = 0.96998
cv_after = 0.94258  # Dropped by 0.027!

# Conclusion: Medical constraints are HARMFUL
# Decision: DO NOT use them
```

## Notes

### Why Domain Constraints Can Fail

**1. Edge Cases Are Real**
- "Impossible" values may be valid edge cases
- Removing them hurts model's ability to handle extremes
- Example: High heart rate in athlete with heart condition

**2. Train/Test Mismatch**
- If you filter training data but not test data
- Model trains on "clean" data but tested on "real" data
- Performance drops due to distribution shift

**3. Information Loss**
- Even "impossible" values contain signal
- They help model learn decision boundaries
- Removing them reduces model capacity

**4. Over-Filtering**
- Domain experts are often too conservative
- "That can't be right" → data entry error
- But it's actually a rare but valid case

### When Constraints ARE Appropriate

**Safe to use**:
- Clear data entry errors (e.g., negative age)
- Impossible physical values (e.g., heart rate = 1000)
- Verified with domain experts AND validated empirically
- Don't change adversarial AUC significantly

**Require validation**:
- "Unusual but possible" values
- Removing >5% of data
- Change train/test distribution
- Drop CV performance

### Alternatives to Hard Constraints

Instead of filtering, consider:

**1. Weighting Examples**
```python
# Down-weight outliers instead of removing
sample_weights = np.ones(len(train))
sample_weights[outliers] = 0.5  # Down-weight, don't remove
model.fit(X, y, sample_weight=sample_weights)
```

**2. Robust Models**
```python
# Use models that handle outliers well
from sklearn.ensemble import HistGradientBoostingClassifier
model = HistGradientBoostingClassifier()  # More robust to outliers
```

**3. Feature Engineering**
```python
# Add indicator for "edge case" instead of filtering
train['is_unusual'] = (
    (train['Max HR'] > (220 - train['Age'])) |
    (train['Cholesterol'] > 400)
).astype(int)
# Model learns to handle unusual cases
```

**4. Post-Processing Rules**
```python
# Apply domain rules AFTER prediction, not before
pred = model.predict(X)
# If prediction says "high risk" but HR is normal, flag for review
high_risk_unusual_hr = pred & (X['Max HR'] < 100)
```

## References

**Verified On**:
- Kaggle Playground Series S6E2 (Heart Disease)
- Medical constraints reduced CV AUC from 0.970 to 0.943
- Adversarial AUC shift from 0.501 to 0.661 confirmed distribution change

**Related Skills**:
- [adversarial-validation-implementation](../adversarial-validation-implementation/):
  For checking train/test distribution alignment
- [kaggle-auc-binary-submission-bug](../kaggle-auc-binary-submission-bug/):
  For proper AUC submission format
