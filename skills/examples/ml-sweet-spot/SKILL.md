---
name: ml-sweet-spot-principle
description: |
  ML optimization sweet spot principle: "More is not always better".
  Use when: (1) Increasing model complexity but validation score plateaus or drops,
  (2) OOF keeps improving but LB stays same or gets worse, (3) Debating between
  simpler vs more complex models, (4) Feature selection or hyperparameter tuning,
  (5) Considering more seeds in ensemble (beyond 5-10), (6) Research claims show
  unexpected results at your scale, (7) Combining different feature engineering approaches.
  Covers OOF-LB gap analysis, overfitting detection, ensemble sweet spots,
  research claim validation, and feature combination incompatibility.
  **UPDATED 2026-02-27**: Added feature combination incompatibility - combining
  different feature types can hurt performance due to conflicting signals.
---

# ML Sweet Spot Principle: "过犹不及" (More is Not Always Better)

## Problem
In ML optimization, there's a common intuition that "more is better":
- More features → better performance?
- More trees → better performance?
- Lower learning rate → better performance?
- Deeper trees → better performance?

**Reality**: Each has an optimal point beyond which performance degrades.

## Context / Trigger Conditions

Use this principle when:
- OOF score keeps improving but LB score plateaus or drops
- Adding more features/features causes performance decline
- More complex model (deeper, more trees) doesn't help
- Debate between simple vs complex approaches
- Hyperparameter tuning shows diminishing returns

**Key indicator**: OOF-LB gap widening as complexity increases

## Solution

### Step 1: Systematic Boundary Testing

Don't guess—test systematically:

```python
# Example: Feature count search
for n_features in [15, 16, 17, 18, 19, 20, 21, 22]:
    features = top_features[:n_features]
    oof_score = cross_validate(model, X[:, :n_features], y)
    lb_score = submit_and_check(features)

    results.append({
        'n_features': n_features,
        'oof': oof_score,
        'lb': lb_score
    })
```

**Pattern to recognize**:
```
Too few → Underfitting (both OOF and LB low)
Sweet spot → Optimal (OOF and LB both peak)
Too many → Overfitting (OOF high, LB drops)
```

### Step 2: Detect Overfitting via OOF-LB Gap

| Scenario | OOF | LB | Interpretation | Action |
|----------|-----|-----|----------------|--------|
| Healthy | High | High | Generalizes well | Keep current setup |
| **Overfitting** | **High** | **Low/Drop** | **Memorizing training** | **Reduce complexity** |
| Underfitting | Low | Low | Not learning enough | Add capacity/features |

**Real example from S6E2**:
```
XGBoost n2603 lr0.0378:  OOF=0.95551, LB=0.95369  ✅ Sweet spot
XGBoost n3000 lr0.032:    OOF=0.95551, LB=0.95368  ❌ Slight overfit
```

Both have same OOF, but the simpler configuration generalizes better!

### Step 3: Model Complexity vs Performance

Common complexity knobs and their effects:

| Knob | Increasing it... | When to stop |
|------|-----------------|--------------|
| **n_estimators** | Improves until plateau | OOF stops improving |
| **learning_rate** | Lower = better (to a point) | LB starts dropping |
| **max_depth** | Deeper = more overfitting | Validation drops |
| **features** | More = noise | LB peaks then falls |
| **seeds** | More = stable | Diminishing returns |

**Key insight**: The optimal point depends on the specific dataset and problem.

### Step 4: Experimental Validation Workflow

```
1. Start with baseline
2. Make single change
3. Measure BOTH OOF and LB
4. If OOF↑ but LB↓ or = → Overfitting, revert
5. If both ↑ → Improvement, keep
6. Repeat
```

## Verification

**Success indicators**:
1. Found clear peak in performance (not just OOF)
2. Simplest model that achieves peak performance
3. Low OOF-LB gap (<0.001)
4. Stable across multiple random seeds

**Example from S6E2**:
```
Tested: Top15, 16, 17, 18, 19, 20, 21, 22 features
Found: Top19 is optimal
Verified: Top19 + 5-seed = LB 0.95369 (best)
Rejected: Top22, Top21 (too many features = overfit)
Rejected: Top15, Top16 (too few = underfit)
```

## Example: Complete Sweet Spot Detection

```python
def find_sweet_spot(X, y, X_test, feature_importance):
    """Find optimal number of features"""

    results = []

    # Test different feature counts
    for n in range(15, 23):
        X_subset = X[:, :n]

        # Cross-validation
        model = XGBClassifier(**best_params)
        oof_scores = cross_val_score(model, X_subset, y, cv=5)
        oof_mean = oof_scores.mean()

        # Train and predict
        model.fit(X_subset, y)
        preds = model.predict_proba(X_test[:, :n])[:, 1]

        # Submit to get LB
        lb_score = submit_to_kaggle(preds)

        results.append({
            'n_features': n,
            'oof': oof_mean,
            'lb': lb_score
        })

    # Find sweet spot
    best = max(results, key=lambda x: x['lb'])

    print(f"Sweet spot: Top{best['n_features']}")
    print(f"OOF: {best['oof']:.5f}, LB: {best['lb']:.5f}")

    return best
```

## Notes

### Key Insights from Real Competitions

1. **Complexity ≠ Performance**
   - CatBoost: 339s/seed, OOF=0.95544 ❌
   - XGBoost: 52s/seed, OOF=0.95551 ✅
   - Simpler, faster model won

2. **Fewer Features Can Be Better**
   - **16 features (no interactions): OOF=0.95553** ✨
   - 19 features: OOF=0.95551
   - **Removed**: st_slope, age_st, vessels_thal
   - **Lesson**: Interaction features can hurt generalization

3. **Interaction Feature Trap**
   - st_slope (ST × Slope of ST)
   - age_st (Age × ST depression)
   - vessels_thal (Vessels × Thallium)
   - These may signal in training but noise in test
   - **Systematically test which interactions help**

4. **OOF Optimization Trap**
   - It's possible to improve OOF while hurting LB
   - Always validate with real test set (LB)
   - OOF is local, LB is global

5. **Parameter-Feature Coupling** (New Discovery 2026-02-25)
   - Optimal parameters DEPEND on feature set
   - **Example from S6E2**:
     - Top19 features: `reg_lambda=2.35` optimal
     - Top16 features: `reg_lambda=1.5` optimal (+0.00001 OOF)
   - **Why**: Fewer features → less noise → less regularization needed
   - **Action**: Re-tune parameters when changing features
   - **Anti-pattern**: Tune params once, keep across feature changes

6. **AutoML ≠ Performance** (New Discovery 2026-02-25)
   - **AutoGluon 100 models**: LB 0.95287
   - **Single XGBoost**: LB 0.95369
   - **Result**: 100 models < 1 model by -0.00082
   - **Why**: Ensemble complexity without diversity = overfitting
   - **Lesson**: Model count ≠ performance

7. **Data Augmentation Fake-Outs** (New Discovery 2026-02-25)
   - **Pseudo-labeling**: OOF +0.01326 → LB -0.00001 (0%转化)
   - **SMOTE 30%**: OOF +0.00982 → LB +0.00001 (0.1%转化)
   - **SMOTE 15%**: OOF +0.00532 → LB -0.00006 (负转化)
   - **Root cause**: Synthetic samples don't reflect test distribution
   - **Lesson**: OOF提升必须LB验证，数据增强尤其容易产生gap

8. **Over-Ensembling Trap** (New Discovery 2026-02-26)
   - **5-seed ensemble**: OOF 0.95369 ✅
   - **20-seed ensemble**: OOF 0.95200 ❌ (-0.00169)
   - **Root cause**: Weak learner noise accumulation
   - **Why**: More seeds → averaging in more weak predictions → signal diluted
   - **Lesson**: Ensemble has sweet spot, more ≠ better
   - **Rule**: 3-5 diverse models optimal, beyond 10 shows diminishing returns

9. **Research Claim Validation** (New Discovery 2026-02-26)
   - **Claim**: UCI external data → +0.0099 AUC
   - **Reality**: Only +0.00009 AUC (0.09% of claimed)
   - **Scale ratio**: 630,000:293 = 2150:1
   - **Root cause**: External data effect scales with ratio
   - **Lesson**: Always validate claims at your scale before full implementation
   - **See also**: [ml-research-validation](../ml-research-validation)

10. **Feature Combination Incompatibility** (New Discovery 2026-02-27)
    - **P_Silent domain features (18)**: CV 0.95224 ✅
    - **Simple interaction features (6)**: CV 0.95152 ❌
    - **Combined hybrid (25)**: CV 0.95210 ❌ (-0.00014 vs P_Silent alone)
    - **Root cause**: Different feature engineering approaches produce conflicting signals that model capacity cannot effectively utilize
    - **Why**: Tree models with limited capacity (depth=3) struggle to reconcile diverse feature types (domain knowledge vs statistical interactions)
    - **Lesson**: Combining feature types can hurt performance - stick with the better individual approach
    - **Key insight**: More features ≠ better when features have incompatible signals
    - **Guideline**: Test feature combinations; if hybrid < best individual, discard the hybrid

6. **Diminishing Returns**
   - First 5 seeds: large improvement
   - Next 5 seeds: minimal gain
   - Know when to stop

### When This Principle Doesn't Apply

**More might actually be better when**:
- Very large dataset (>1M samples)
- Very deep learning (not tree models)
- Massive regularization in place
- Clear underfitting pattern

**Signs you need more complexity**:
- Both OOF and LB are low
- Training error >> validation error
- Model is extremely simple

### Anti-Patterns to Avoid

1. **Chasing OOF alone** - Always look at LB
2. **Adding complexity blindly** - Test systematically
3. **Ignoring speed** - 6x slower for 0.00001 worse is bad
4. **Over-tuning** - If improvement <0.0001, move on
5. **Forgetting simplicity** - Simple > complex if similar performance

## References

- [Bias-Variance Tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html)
- Kaggle S6E2 Competition - Verified with LB 0.95369 achievement

## 2026-02-25 Update: Comprehensive Experimental Validation

**100+ Kaggle Experiments Summary**:

| 方法 | 复杂度 | OOF | LB | 结论 |
|------|--------|-----|-----|------|
| XGBoost基线 | 1模型, 16特征 | 0.95555 | **0.95369** | 🥇 最优 |
| SMOTE 30% | +合成样本 | 0.96537 | 0.95370 | 持平 |
| 伪标签 | +测试样本 | 0.96881 | 0.95368 | 持平 |
| AutoGluon | 100模型 | 0.95478 | 0.95287 | **下降** |
| Stacking | 3模型 | 0.95226 | 0.95020 | **下降** |
| 医学特征 | 41特征 | 0.95218 | 未提交 | 过拟合 |
| **混合特征 (2026-02-27)** | P_Silent + 交互 (25特征) | 0.95210 | - | **不如单独** |
| 简单交互特征 | 6交互特征 | 0.95152 | - | 更差 |
| P_Silent特征 | 18特征 | **0.95224** | 0.95021 | 当前最佳 |

**核心发现验证**:
1. ✅ **OOF ≠ LB**: 100+实验证明OOF提升不转化LB
2. ✅ **简单 > 复杂**: 1个XGBoost > 100个AutoGluon模型
3. ✅ **数据增强陷阱**: SMOTE/伪标签OOF提升完全虚假
4. ✅ **过拟合模式**: 复杂度增加导致LB下降

**关键数据**:
- AutoGluon权重: XGBoost 85%, CatBoost 10%, XGBoost 5%
- 3模型相关系数: 0.99983 (几乎完全相同)
- 对抗验证AUC: 0.50226 (训练/测试分布相同)

**教训**: "过犹不及"不是口号，是数据科学的基本规律。
