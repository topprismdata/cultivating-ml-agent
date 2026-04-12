---
name: kaggle-top-performer-replication
description: |
  Systematically replicate and learn from top Kaggle performers. Use when:
  (1) Stuck at a plateau and need new ideas, (2) Want to understand how
  top performers achieved their scores, (3) Looking for proven techniques from
  successful submissions, (4) OOF keeps improving but LB stays same or gets worse.
  Covers: code analysis workflow, Frequency+Target encoding, Meta-learning
  with rank-transformed predictions, Temperature Scaling, and Multi-seed training.
---

# Kaggle Top Performer Replication Strategy

## Problem

In Kaggle competitions, it's common to hit a performance plateau. Reading discussion
posts or papers often doesn't reveal the actual implementation details that led to
top scores. The best way to learn is to directly analyze and replicate top performer's
code.

**Key Insight**: Top performers rarely share all their secrets in discussions. Their
notebooks contain the actual implementation details.

## Context / Trigger Conditions

Use this strategy when:
- CV score improves but LB score plateaus or drops
- Gap between your score and top performers is >0.001
- Multiple attempts at feature engineering aren't yielding improvements
- Want to understand state-of-the-art techniques

**Key indicator**: OOF-LB gap widening or stagnation despite various approaches

## Solution

### Step 1: Download Top Performer Code

Use Kaggle API to download top performer notebooks:

```bash
# List top notebooks by votes
kaggle kernels list --competition <competition-name> --sort voteCount

# Download specific notebook
kaggle kernels pull <username>/<notebook-name> -p /tmp/kaggle_notebooks/

# Read the notebook
# The notebook is in .ipynb format - read cells to extract code
```

**What to look for**:
1. Feature engineering approaches (encoding strategies, interaction features)
2. Model architectures (depth, learning rate, regularization)
3. Ensemble/stacking methods
4. Data preprocessing steps
5. Cross-validation strategies

### Step 2: Identify Key Differences

Create a comparison table:

| Technique | Your Approach | Top Performer | Impact |
|-----------|--------------|---------------|---------|
| Encoding | One-hot | Frequency+Target | ? |
| Model depth | 3 | 2 | ? |
| Seeds | 1 | 2+ | ? |
| Meta-learning | None | RF on ranks | ? |

### Step 3: Prioritize Techniques by Expected Impact

Focus on techniques that are:
1. **Easier to implement** (quick wins)
2. **Well-understood** (not black magic)
3. **Verifiable** (can test independently)

**Priority Order**:
1. Feature engineering (highest impact, lowest risk)
2. Model architecture changes
3. Ensemble/stacking methods
4. Advanced techniques (calibration, etc.)

### Step 4: Implement and Validate

Implement one technique at a time and validate:

```python
# Example: Test Frequency Encoding
def test_technique(base_model, new_features):
    oof_base = cross_validate(base_model, X, y)
    oof_new = cross_validate(base_model, X_with_new, y)

    lb_base = submit(oof_base)
    lb_new = submit(oof_new)

    print(f"OOF: {oof_base:.5f} → {oof_new:.5f}")
    print(f"LB:  {lb_base:.5f} → {lb_new:.5f}")

    return lb_new > lb_base
```

**Validation Criteria**:
- ✓ OOF improvement >0.001
- ✓ LB improvement >0 (or OOF-LB gap doesn't widen)
- ✗ OOF improves but LB drops → overfitting, discard

## Verified Techniques (from S6E2 Heart Disease)

### 1. Frequency + Smoothed Target Encoding

**Impact**: +0.00339 LB improvement (0.95021 → 0.95360)

```python
# Frequency Encoding
def freq_encode(train, test, cols):
    all_df = pd.concat([train, test])
    tr_out, te_out = pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)
    for c in cols:
        freq = all_df[c].value_counts(normalize=True)
        tr_out[c + "_freq"] = train[c].map(freq).fillna(0)
        te_out[c + "_freq"] = test[c].map(freq).fillna(0)
    return tr_out, te_out

# Smoothed Target Encoding with K-Fold (prevents leakage!)
def smooth_te(train, test, col, target, alpha=15):
    global_mean = train[target].mean()
    kf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = np.zeros(len(train))

    for tr_idx, val_idx in kf.split(train, train[target]):
        stats = train.iloc[tr_idx].groupby(col)[target].agg(['mean','count'])
        smooth = (stats['mean']*stats['count'] + alpha*global_mean) / (stats['count']+alpha)
        oof[val_idx] = train.iloc[val_idx][col].map(smooth)

    # Full training data for test
    stats_full = train.groupby(col)[target].agg(['mean','count'])
    smooth_full = (stats_full['mean']*stats_full['count'] + alpha*global_mean) / (stats_full['count']+alpha)
    test_enc = test[col].map(smooth_full).fillna(global_mean)

    return oof, test_enc
```

**Why alpha=15**:
- Balances category-level mean with global mean
- Prevents overfitting on rare categories
- Standard value that works across many competitions

### 2. Interaction Features

```python
# Domain-knowledge driven interactions
train["Age_MaxHR"] = train["Age"] * train["Max HR"]
train["ST_Exercise"] = train["ST depression"] * train["Exercise angina"]
train["Vessels_Thallium"] = train["Number of vessels fluro"] * train["Thallium"]
```

**Key**: Interactions should be medically/physically meaningful, not just random combinations.

### 3. Rank-Transformed Meta Learning

**Impact**: Often adds +0.0005 to +0.002 over simple ensemble

```python
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier

def rank_transform(preds):
    """Convert predictions to normalized ranks"""
    ranks = np.zeros_like(preds)
    for i in range(preds.shape[1]):
        ranks[:,i] = (rankdata(preds[:,i]) - 1) / (len(preds) - 1)
    return ranks

# Train base models (multi-seed)
base_oof = []  # Collect OOF from 2 seeds × 3 frameworks = 6 models
base_test = []

# Stack on ranks
meta_oof_rank = rank_transform(np.column_stack(base_oof))
meta_test_rank = rank_transform(np.column_stack(base_test))

# Meta model
rf_meta = RandomForestClassifier(
    n_estimators=1000, max_depth=6, min_samples_leaf=50,
    class_weight="balanced", n_jobs=-1, random_state=42
)
rf_meta.fit(meta_oof_rank, y)

final_oof = rf_meta.predict_proba(meta_oof_rank)[:, 1]
```

**Why ranks**:
- Reduces scale sensitivity between different models
- More robust than probability-based stacking
- Captures non-linear interactions between base predictors

### 4. Temperature Scaling

**Impact**: +0.0001 to +0.0005 final AUC

```python
from scipy.optimize import minimize

def apply_temp(p, t):
    p = np.clip(p, 1e-6, 1-1e-6)
    logit = np.log(p/(1-p))
    return 1/(1+np.exp(-logit*t))

def objective(t):
    preds = apply_temp(final_oof, t[0])
    return -roc_auc_score(y, preds)

# Optimize temperature on OOF predictions
res = minimize(objective, x0=[1.0], bounds=[(0.8, 1.2)])
T_opt = res.x[0]

# Apply to test predictions
final_test = apply_temp(final_test, T_opt)
```

**Why it works**:
- Adjusts probability sharpness for AUC metric
- Preserves ranking while optimizing calibration
- Small temperature changes (<10%) can improve AUC

## Verification

Validate each technique independently:

1. **Feature engineering**: Add to baseline, test OOF+LB
2. **Model changes**: Compare same features, different architecture
3. **Ensemble**: Test meta-model vs simple average
4. **Calibration**: Check if temperature improves OOF

**Success criteria**:
- ✓ OOF improves >0.001
- ✓ LB improves (or OOF-LB gap doesn't widen)
- ✗ OOF improves but LB drops → overfitting

## Common Pitfalls

### Don't Blindly Copy Everything

Top performers may use:
- External data (not allowed in some competitions)
- Ensembling with many weak models (diminishing returns)
- Competition-specific tricks (not generalizable)

**Apply judgment**: Focus on generalizable techniques, not competition-specific hacks.

### Feature Combination Incompatibility

**Discovery**: Combining different feature engineering approaches can hurt performance.

**Example from S6E2**:
```
P_Silent domain features (18): CV 0.95224 ✅
Simple interaction features (6): CV 0.95152 ❌
Combined hybrid (25): CV 0.95210 ❌ (-0.00014 vs P_Silent alone)
```

**Root cause**: Different feature types produce conflicting signals that model capacity cannot reconcile.

**Guideline**: Test feature combinations; if hybrid < best individual, discard the hybrid.

### OOF-LB Gap Warning

**Critical**: SMOTE and data augmentation often show large OOF improvements but minimal LB gains.

**Example from S6E2**:
```
SMOTE 30%: OOF +0.00982, LB +0.00001 (0.1% conversion)
Pseudo-labeling: OOF +0.01326, LB -0.00001 (negative conversion)
```

**Lesson**: Always validate with LB. OOF improvements don't always translate.

## Multi-Seed vs Cross-Model Ensemble

**When multi-seed works better**:
- Same model architecture, different random seeds
- Reduces variance through averaging
- Typically improves LB by +0.0001 to +0.0005

**When cross-model works better**:
- Different architectures (XGBoost, LightGBM, CatBoost)
- Different feature sets
- Low model correlation (<0.99)

**Rule of thumb**: Check model correlation before deciding:
```python
from scipy.stats import pearsonr

corr, _ = pearsonr(oof_pred1, oof_pred2)
if corr > 0.999:
    print("Models too correlated for stacking")
    print("Use multi-seed instead")
```

See also: `model-ensemble-negative-weight-effect`

## Example: Complete Replication Workflow

```python
# 1. Download top notebook
kaggle kernels pull <username>/<notebook> -p /tmp/notebook/

# 2. Extract techniques
# 3. Implement priority features first
# 4. Validate each independently
# 5. Combine what works

# Typical priority order:
#    a) Frequency+Target encoding
#    b) Interaction features
#    c) Multi-seed training (2-3 seeds)
#    d) Meta-learning (if still gap)
#    e) Temperature scaling (final optimization)
```

## Notes

### When Not to Use This Strategy

- **Simple datasets** (<10K rows): Feature engineering may not help
- **Time constraints**: Full replication takes significant time
- **Early competition**: Leaderboard still volatile, wait for stabilization

### Best Practices

1. **Start simple**: Implement basic version first, then add complexity
2. **Test independently**: One change at a time, validate with LB
3. **Document everything**: Keep track of what worked and what didn't
4. **Share back**: Contribute improved techniques back to community

### Research Claims Validation

Be skeptical of bold claims:
- External data → +0.01 AUC? Verify at your scale
- New architecture → SOTA? Compare to strong baselines
- "Guaranteed improvement"? Test with your data

See also: `ml-research-validation`

## References

- [Kaggle Competitions API](https://github.com/Kaggle/kaggle-api)
- [S6E2 Leaderboard](https://www.kaggle.com/competitions/playground-series-s6e2/leaderboard)
- [Kaggle Ensemble Guide](https://www.kaggle.com/code/artgor/ensemble-guide/notebook)

**Verified on**: S6E2 Heart Disease Competition (LB 0.95021 → 0.95360, gap to first place reduced by 85%)
