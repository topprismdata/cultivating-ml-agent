---
name: model-ensemble-negative-weight-effect
description: |
  Avoid ensembling models with significantly different performance. Use when:
  (1) Considering weighted average of models with divergent OOF/LB scores,
  (2) One model clearly underperforms another, (3) Ensemble shows lower
  validation score than best individual model. Covers the negative weight
  effect where adding a weaker model reduces overall performance despite
  theoretically sound weighting strategy.
---

# Model Ensemble Negative Weight Effect

## Problem
Ensembling a strong model with a weak model can actually **decrease** performance,
even when using theoretically sound weighting strategies like exp(OOF AUC).

## Context / Trigger Conditions
- Two models with significantly different validation scores (e.g., ΔAUC > 0.03)
- Weighted ensemble produces lower score than the best single model
- Weights calculated as `exp(score) / sum(exp(scores))`
- Temptation to ensemble "just in case" it helps
- Models using different feature sets or strategies

## Solution

### Rule of Thumb: Only Ensemble Similar-Performing Models

**Don't ensemble when:**
```
Model A OOF: 0.955
Model B OOF: 0.906
ΔOOF: 0.049  ← Too large! Model B will drag down Model A
```

**Do ensemble when:**
```
Model A OOF: 0.955
Model B OOF: 0.953
ΔOOF: 0.002  ← Similar enough, ensemble may help
```

### Verification Method

Before committing to ensemble, test:

```python
# Calculate theoretical weight
import numpy as np
w_a = np.exp(0.955) / (np.exp(0.955) + np.exp(0.906))  # 0.73
w_b = np.exp(0.906) / (np.exp(0.955) + np.exp(0.906))  # 0.27

# If weaker model gets >20% weight, it's risky
print(f"Weaker model weight: {w_b:.2%}")  # If >20%, be careful
```

### Alternative Strategies

1. **Skip ensemble**: Use best single model
2. **Threshold ensemble**: Only include models within ΔX of best
   ```python
   threshold = 0.01  # Only models within 1% AUC
   candidates = [m for m in models if best_score - m.score < threshold]
   ```
3. **Blend only top-N**: Take top 3 models, ignore rest
4. **Stacking**: Use meta-learner instead of simple weighted average

## Example

**S6E2 Heart Disease Competition:**

| Model | OOF AUC | LB AUC | Weight (exp) |
|-------|---------|--------|--------------|
| V3 Single | 0.95545 | 0.95358 | - |
| V4.1 Raw | 0.90639 | ~0.90 | **48.8%** |
| **V3+V4 Ensemble** | - | **0.95348** | - |

**Result**: Ensemble (0.95348) < V3 Single (0.95358) < V3 Multiseed (0.95359)

**Why it failed:**
- `exp(0.95545) / (exp(0.95545) + exp(0.90639))` ≈ 0.512
- V4.1 got **48.8%** weight despite being Δ0.049 worse!
- exp归一化在小差异下几乎等于线性归一化
- Weaker model's nearly 50% weight dragged down performance
- No complementary signal between feature engineering and raw features

**Lesson**: V3 Multiseed (5 seeds, Δ≈0.000) gave +0.00001 improvement, while V3+V4 ensemble gave -0.00010

## Verification

Check if ensemble is worthwhile:

```python
# 1. Calculate score gap
gap = best_score - worst_score
if gap > 0.03:  # 3% AUC difference
    print("WARNING: Gap too large for ensemble")

# 2. Check weak model weight
weak_weight = exp(worst_score) / (exp(best_score) + exp(worst_score))
if weak_weight > 0.20:  # More than 20%
    print("WARNING: Weak model has significant weight")

# 3. Validate on holdout set
ensemble_pred = w1 * pred1 + w2 * pred2
if ensemble_score < best_single_score:
    print("Ensemble degrades performance!")
```

## Notes

### When Ensemble Works
- Models have similar performance (Δ < 0.01-0.02 AUC)
- **Models have LOW correlation (<0.99)** ← Critical!
- Models capture different patterns (e.g., different algorithms, not just different seeds)
- Multiseed averaging of same architecture (variance reduction only)
- Complementary errors (model A wrong where B is right, and vice versa)

### When Ensemble Fails
- Large performance gap between models
- One model is strictly worse than another
- **Models have VERY HIGH correlation (>0.999)** ← Correlation trap!
- Different feature sets but weaker model has no unique signal
- Weighted average with weak model getting >20% weight

### Model Correlation Trap (New Discovery 2026-02-26)

**The Problem**: Even with similar performance, highly correlated models provide no stacking benefit.

**Real Example from S6E2**:
```
XGBoost OOF:   0.95221
CatBoost OOF:  0.95213
Performance Δ: 0.00008 (excellent!)

Correlation:   0.99951 ← Too high!

Stacking Result:
  Simple Average: 0.95222
  Stacking (LR):  0.95222
  Improvement:     0.00000
```

**Why**: When correlation > 0.999, models make nearly identical predictions. Stacking just learns:
```python
# Logistic Regression learns:
y = 3.1976 * xgb_pred + 3.2102 * cat_pred - 3.2118
# Which is essentially:
y = (xgb_pred + cat_pred) / 2  # Simple average!
```

**Rule of Thumb**:
```
Correlation < 0.95:  Stacking likely effective
Correlation 0.95-0.99:  Marginal gains
Correlation > 0.999:  No benefit, use simple average
```

**How to Check**:
```python
from scipy.stats import pearsonr

corr, _ = pearsonr(oof_pred1, oof_pred2)
print(f"Model correlation: {corr:.5f}")

if corr > 0.999:
    print("WARNING: Correlation too high for stacking!")
    print("  Use simple average instead")
```

**Diverse Models for Stacking**:
- XGBoost + LightGBM (usually <0.99 correlation)
- GBDT + Neural Network (if NN performance is competitive)
- Different feature subsets
- Different random seeds (only variance reduction)

### Mathematical Intuition

For two models with scores s₁ > s₂:

```
If s₁ - s₂ > log((1-w₂)/w₂):
    Ensemble < s₁ (degradation occurs)

For s₁=0.955, s₂=0.906, w₂=0.27:
0.955 - 0.906 = 0.049
log((1-0.27)/0.27) = log(2.7) ≈ 0.99

Since 0.049 < 0.99, ensemble should work... BUT this assumes
independent errors. In practice, errors are correlated, so the
weaker model just adds noise.
```

### Key Insight

**The exp() normalization trap**

```python
# exp()归一化在小差异下几乎等于线性归一化
w_a = exp(0.955) / (exp(0.955) + exp(0.906))  # ≈ 0.51
w_b = exp(0.906) / (exp(0.955) + exp(0.906))  # ≈ 0.49

# 线性归一化
w_a = 0.955 / (0.955 + 0.906)  # ≈ 0.51
w_b = 0.906 / (0.955 + 0.906)  # ≈ 0.49
```

**结论**: exp(score)/sum(exp(scores)) 在score接近0.9~1.0时几乎等于线性归一化！
如果想放大差异，需要用exp(k*score)其中k>1，或者直接手动指定权重。

**Multiseed > Cross-Model Ensemble**

- Multiseed: Same model, different randomness → reduces variance
- Cross-model: Different models/features → requires genuine diversity

If models aren't bringing unique signal, you're just averaging noise.

## References

- [Kaggle Ensemble Guide](https://www.kaggle.com/code/artgor/ensemble-guide/notebook)
- [Why Ensemble Might Not Work](https://machinelearningmastery.com/why-ensemble-models-fail/)
- S6E2 Competition: V3 (0.95545 OOF) + V4 (0.90639 OOF) → 0.95348 LB (worse than V3 alone)
