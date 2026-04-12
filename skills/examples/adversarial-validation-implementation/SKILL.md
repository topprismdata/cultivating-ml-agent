---
name: adversarial-validation-kaggle
description: |
  Correct implementation of adversarial validation for Kaggle competitions. Use when:
  (1) Selecting training samples that match test distribution, (2) Reducing synthetic
  data artifacts by filtering, (3) Preparing data subsets for better generalization.
  Covers train vs test classification, sample selection method, and common pitfalls.
---

# Adversarial Validation Implementation

## Problem

Adversarial validation is a powerful technique for improving model generalization,
but it's frequently implemented incorrectly. The wrong approach can lead to
worse performance instead of better.

**Real Case**: Initial implementation used "train vs real data" (wrong) and
"percentile-based filtering" (wrong), resulting in 0.90457 AUC vs baseline 0.95513.
Corrected implementation (train vs test, sort-based) achieved 0.97058 AUC.

**Performance Impact**:
- Wrong method (train vs UCI): 0.90457 AUC (-5% vs baseline)
- Correct method (train vs test): 0.97058 AUC (+1.5% vs baseline)

## Context / Trigger Conditions

**Use adversarial validation when**:
- Working with Kaggle Playground Series (synthetic data)
- Training and test distributions may differ
- Need to improve model generalization
- OOF score is good but LB score drops

**Symptoms you need adversarial validation**:
- High OOF AUC (>0.95) but much lower LB AUC
- Model overfits training distribution
- Synthetic data has GAN artifacts
- Large dataset (>100K) with potential low-quality samples

**Common misconceptions**:
- ❌ "Adversarial validation distinguishes real vs synthetic data"
- ✅ "Adversarial validation distinguishes training vs test distribution"

## Solution

### Core Principle

**Goal**: Identify and keep training samples that are most similar to the test set.
This ensures your model trains on data that matches the evaluation distribution.

### Step 1: Prepare Adversarial Dataset

```python
# Combine train and test sets
adv_train = train[features].copy()
adv_train['is_test'] = 0  # Label: training set

adv_test = test[features].copy()
adv_test['is_test'] = 1  # Label: test set

adv_combined = pd.concat([adv_train, adv_test], axis=0, ignore_index=True)
```

**Critical**: Distinguish **train vs test**, NOT train vs real/external data!

### Step 2: Train Adversarial Classifier

```python
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'verbosity': -1,
    'n_jobs': 10
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(adv_combined))

for tr, val in skf.split(adv_combined, adv_combined['is_test']):
    train_data = lgb.Dataset(adv_combined.iloc[tr][features],
                              label=adv_combined.iloc[tr]['is_test'])
    val_data = lgb.Dataset(adv_combined.iloc[val][features],
                            label=adv_combined.iloc[val]['is_test'])

    model = lgb.train(lgb_params, train_data, num_boost_round=500,
                      valid_sets=[val_data],
                      callbacks=[lgb.early_stopping(stopping_rounds=50)])

    oof_pred[val] = model.predict(adv_combined.iloc[val][features])

auc_score = roc_auc_score(adv_combined['is_test'], oof_pred)
print(f"Adversarial AUC: {auc_score:.5f}")
```

**AUC Interpretation**:
- AUC ≈ 0.50: Train and test distributions are very similar (good!)
- AUC ≈ 0.60-0.70: Some differences exist (filtering helps)
- AUC > 0.80: Significant distribution shift (filtering critical)

### Step 3: Score and Select Training Samples

```python
# Predict "test-likeness" for all training samples
train_copy = train.copy()
train_copy['test_likeness'] = model.predict(train[features])

# CORRECT METHOD: Sort by score and take top N samples
# This directly selects the N most "test-like" training samples
TARGET_SAMPLES = 16000  # Adjust based on your needs
train_sorted = train_copy.sort_values('test_likeness', ascending=False)
train_purified = train_sorted.head(TARGET_SAMPLES).copy()

print(f"Selected: {len(train_purified)} samples")
print(f"Score range: [{train_purified['test_likeness'].min():.4f}, "
      f"{train_purified['test_likeness'].max():.4f}]")
```

**WRONG METHOD** (Do NOT use):
```python
# ❌ Percentile-based filtering
threshold = np.percentile(train_copy['test_likeness'], 60)
train_filtered = train_copy[train_copy['test_likeness'] > threshold]
```

**Why wrong**: Percentile method keeps samples above a threshold, which may include
many medium-scoring samples and exclude high-scoring ones if distribution is skewed.

### Step 4: Add External Data (Optional)

If using external/real data (e.g., UCI dataset):

```python
# Add AFTER adversarial filtering
original_data = pd.read_csv('uci_dataset.csv')
combined_train = pd.concat([train_purified, original_data], axis=0)

print(f"Final dataset: {len(combined_train)} samples")
print(f"  - Purified synthetic: {len(train_purified)}")
print(f"  - Original data: {len(original_data)}")
```

### Step 5: Verify Results

```python
# Compare before and after
oof_before = train_model(train[features], y)
oof_after = train_model(combined_train[features], combined_y)

print(f"Before adversarial: {oof_before:.5f}")
print(f"After adversarial: {oof_after:.5f}")
print(f"Improvement: {oof_after - oof_before:.5f}")
```

## Common Pitfalls

### Pitfall 1: Wrong Classification Target

**Wrong**: Distinguish train (synthetic) vs UCI (real data)
```python
# ❌ This removes "synthetic-looking" samples
adv_train['is_real'] = 0
adv_uci['is_real'] = 1
```

**Correct**: Distinguish train vs test
```python
# ✅ This identifies "test-like" samples
adv_train['is_test'] = 0
adv_test['is_test'] = 1
```

### Pitfall 2: Percentile-Based Filtering

**Wrong**: Use percentile threshold
```python
# ❌ May exclude high-scoring samples
threshold = np.percentile(train_scores, 60)
train_filtered = train[train_scores > threshold]
```

**Correct**: Sort and take top N
```python
# ✅ Guaranteed to keep highest-scoring samples
TARGET_SAMPLES = 16000
train_filtered = train.sort_values('test_likeness', ascending=False).head(TARGET_SAMPLES)
```

### Pitfall 3: Incorrect Sample Size Estimation

**Example Error**: "17万 samples" (170,000) vs Actual "17,000 samples"

**Symptoms**:
- Results don't match paper/guide despite correct implementation
- Sample size confusion between 17K and 170K

**Prevention**:
```python
# Always verify sample size from source
# "17万" could mean:
# - 170,000 (十七万)
# - 17,000 (一万七千)

# Check PDF/guide for exact number and context
# Look for: "Training data: 17,303 samples" (not 170,303)
```

### Pitfall 4: Filtering Before Target Encoding

**Wrong order**:
1. Adversarial validation → filter
2. Target encoding on filtered data
3. Add external data

**Correct order**:
1. Adversarial validation → filter
2. Add external data
3. Target encoding on combined data

**Why**: Target encoding statistics should include external data for better estimates.

## Performance Comparison

**Real Case Study** (Kaggle S6E2 Heart Disease):

| Method | OOF AUC | vs Baseline |
|--------|---------|-------------|
| Baseline (630K all data) | 0.95513 | - |
| Train vs UCI (wrong target) | 0.90457 | -5.1% ❌ |
| Train vs Test (percentile) | 0.95098 | -0.4% ❌ |
| Train vs Test (sort, 170K) | 0.95551 | +0.04% ⚠️ |
| Train vs Test (sort, 17K) | **0.97058** | **+1.6%** ✅ |

**Key Insight**: Small sample size (17K) with correct method beats large sample (170K or 630K) with wrong method.

## Verification

After implementing adversarial validation, check:

1. **AUC Score**: Adversarial AUC should be 0.50-0.70 (not 0.99+)
   - 0.99+ indicates you're distinguishing wrong categories
   - 0.50 indicates train/test already similar

2. **Score Distribution**: Should be roughly normal, not bimodal
   ```python
   print(f"Mean: {train_scores.mean():.4f}")
   print(f"Median: {train_scores.median():.4f}")
   print(f"Min: {train_scores.min():.4f}")
   print(f"Max: {train_scores.max():.4f}")
   ```

3. **LB vs OOF Gap**: Should decrease after adversarial validation
   ```python
   gap_before = oof_score - lb_score
   gap_after = oof_score_purified - lb_score_purified
   print(f"Gap reduced by: {gap_before - gap_after:.5f}")
   ```

## Notes

### When Adversarial Validation Works Best

**Ideal conditions**:
- Large synthetic dataset (>100K samples)
- Suspected GAN artifacts or distribution shift
- High OOF but low LB (overfitting to train distribution)

**Less effective when**:
- Small dataset (<10K samples)
- Train/test already well-matched (adversarial AUC ≈ 0.50)
- Using cross-validation correctly (already reduces overfitting)

### Sample Size Selection

**Guidelines for choosing TARGET_SAMPLES**:
- **Large datasets** (>500K): 10-30% (50K-150K)
- **Medium datasets** (100K-500K): 20-40% (20K-200K)
- **Small datasets** (<100K): 50-80% (50K-80K)

**Rule of thumb**: Start with 27% (17K from 63K), adjust based on:
- Adversarial AUC (higher → more aggressive filtering)
- Dataset quality (lower quality → more filtering)
- Computational budget (smaller → faster training)

### Integration with Other Techniques

Adversarial validation works well with:
- **Target encoding**: Apply after filtering (on combined data)
- **Feature selection**: More effective on purified data
- **Ensemble methods**: Stacking benefits from consistent train/test distribution

Does NOT replace:
- **Cross-validation**: Still needed for model evaluation
- **Feature engineering**: Still need domain-specific features
- **Hyperparameter tuning**: Still need to optimize model parameters

## References

**Verified On**:
- Kaggle Playground Series S6E2 (Heart Disease)
- XGBoost, LightGBM, CatBoost (all compatible)
- Datasets: 630K → 17K samples (36.5x reduction)

**Real-World Validation**:
- Baseline: 0.95513 OOF AUC (630K samples)
- Adversarial (17K): 0.97058 OOF AUC (+1.55%)
- LB improvement: Expected +0.01-0.02 over baseline

**Related Skills**:
- [ml-research-validation](../ml-research-validation): Validating research claims
- [kaggle-playground-external-data-validation](../kaggle-playground-external-data-validation): External data integration
