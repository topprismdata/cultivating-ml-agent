---
name: cv-lb-gap-acknowledgment
description: |
  CV-LB Gap Acknowledgment: CV improvement does NOT equal LB improvement.
  This is one of the most important MLOps principles. Use when: (1) OOF score
  keeps improving but LB score plateaus or drops, (2) Spending days tuning
  hyperparameters without LB improvement, (3) Comparing models on CV only
  without LB validation, (4) Trusting cross-validation as the "final" metric.
  Covers the mathematical reasons for CV-LB gap (overfitting to OOF noise,
  distribution shift train vs test, hyperparameter over-tuning to CV),
  empirical validation (0.005-0.01 gap observed in tabular competitions),
  the 5-stage validation pipeline (CV → submission → LB → analysis),
  when to stop iterating on CV alone, and the 80/20 rule for time allocation
  between CV optimization and LB validation.
---

# CV-LB Gap Acknowledgment

## Problem
A common MLOps trap: **trusting CV score as the final metric**.

**Reality**:
- Many experiments show **CV improvement but LB degradation**
- The gap can be 0.005-0.01 (5-10% of total error) on tabular competitions
- Days of CV optimization may yield **worse** LB results
- "Best CV" ≠ "Best LB"

**The trap**: Continue iterating on CV because "the score is improving", missing that CV is overfitting to noise.

## Context / Trigger Conditions

Use this skill when:
- **OOF score keeps improving but LB score plateaus or drops** (red flag!)
- **Spending days tuning hyperparameters** without LB improvement
- **Comparing models on CV only** without LB validation
- **Trusting cross-validation as the "final" metric** for submissions
- **Adding more features** because CV says so
- **More complex model** beats simpler one on CV
- **Longer training** (more iterations) improves CV

## Solution: Mandatory 5-Stage Validation Pipeline

```
Stage 1: CV (Cross-Validation)
  ↓ Compute OOF score
Stage 2: Submission
  ↓ Submit to LB
Stage 3: LB Validation
  ↓ Compare CV vs LB
Stage 4: Analysis
  ↓ Diagnose gap
Stage 5: Decision
  ↓ Iterate / Pivot / Stop
```

**Never skip Stage 2-4**! CV is necessary but not sufficient.

## Mathematical Reasons for CV-LB Gap

### 1. Overfitting to OOF Noise

OOF predictions are still **predictions on training data** (just held out in folds). They are not independent of training process.

**Variance decomposition**:
```
Var(OOF) = Var(true_error) + Var(model_selection) + Var(noise)
```

- OOF score includes **model selection variance** (you picked this model because OOF was high)
- LB score is **out-of-sample** (no selection bias)

### 2. Distribution Shift (Train vs Test)

- **Train**: Same distribution you optimize on
- **Test**: May have:
  - Different feature distributions (covariate shift)
  - Different label distributions (prior shift)
  - Different relationships (concept shift)

**Example (Spaceship Titanic)**:
- Train: 8693 rows
- Test: 4277 rows (different distribution)
- CV-LB gap: 0.005-0.01 (0.8124 OOF → 0.8078 LB)

### 3. Hyperparameter Over-tuning to CV

When you optimize hyperparameters based on CV score, you implicitly **fit to CV noise**.

**Example**:
- Hyperparameter search finds: depth=7, lr=0.01, n_est=3000 (best on CV)
- This specific config may be **lucky on this CV split** but not on test
- True generalization: depth=6, lr=0.02 (more robust, slightly worse CV)

### 4. Submission Variance

Kaggle's public LB uses ~50% of test set. Private LB (final) uses 100%.
**Public LB ≠ Private LB** in some competitions.

## Empirical Validation (2026-06-13)

| Experiment | OOF | LB | Gap | Verdict |
|------------|-----|----|----|---------|
| House Prices V7 (XGBoost) | 0.128785 | 0.13804 | **+0.009** | CV 改善但 LB 也退化 |
| House Prices V11 (Stack LR) | 0.141298 | 1.16254 | **+1.02** | ❌ 完全失败 |
| Spaceship V6 (3 new features) | 0.8136 | 0.80687 | **-0.0067** | CV 高 LB 低 |
| Spaceship V6 (Top-5) | 0.8136 | 0.80430 | **-0.0093** | CV 高 LB 低 |
| Spaceship V4 (15 CAT) | 0.8131 | 0.80570 | **-0.0074** | CV 高 LB 低 |
| Spaceship V2 (Top-5 CAT) | 0.8126 | 0.80780 | **-0.0048** | Best LB (lowest gap) |
| Spaceship V8 (AutoGluon) | 0.8317 | pending | TBD | TBD |

**Key Observations**:
- **Gap is consistent**: 0.005-0.01 across experiments
- **More complex features → bigger gap** (V6 vs V2)
- **Simpler is better**: V2 (basic 5 CatBoost) had lowest CV-LB gap

## 4 Practical Rules

### Rule 1: Submit After Major CV Improvements

```python
# After each major CV improvement, submit to LB
if oof_improved_by_0.001:
    submit_to_lb()
    compare_cv_vs_lb()
```

**Don't wait for "perfect" CV**. Submit early and often.

### Rule 2: Trust the Simpler Model When CV-LB Gap is Large

If model A has CV 0.001 better than model B but LB 0.005 worse:
- **Choose model B** (simpler, more robust)
- CV improvement of 0.001 may be noise
- LB degradation of 0.005 is real

### Rule 3: Allocate 80% Time to LB Validation, 20% to CV Optimization

```python
time_allocation = {
    'cv_optimization': 0.20,  # 20% on improving CV
    'lb_validation': 0.80,     # 80% on LB validation + analysis
}
```

**Why?**: CV is faster but unreliable. LB is slower but real.

### Rule 4: Watch for the "CV Improvement Trap"

```python
# Red flag pattern:
for v in [v1, v2, v3, ...]:
    if v.oof > previous.oof:
        print(f"V{v} CV improved! Continue iterating")
        submit_to_lb()
    else:
        print(f"V{v} CV worse. Pivot to new approach")
```

**Better**:
```python
for v in [v1, v2, v3, ...]:
    if v.oof > previous.oof and v.lb > previous.lb:
        print(f"V{v} Both CV and LB improved. Continue")
    elif v.oof > previous.oof and v.lb < previous.lb:
        print(f"V{v} CV up but LB down. STOP, pivot")
    else:
        print(f"V{v} Both worse. Pivot")
```

## When to STOP Iterating on CV

Stop if any of these are true:
- **3 consecutive CV improvements all have LB degradation**
- **CV is improving but LB is plateauing** (within noise of best LB)
- **You can't improve CV by >0.001** (noise level)
- **Time budget exhausted** (80% of time on CV, 20% on LB)

## How to Diagnose CV-LB Gap

### Step 1: Submit Current Best CV Model
```python
best_cv_model = max(experiments, key=lambda e: e.oof)
best_cv_model.submit_to_lb()
```

### Step 2: Analyze
```python
gap = best_cv_model.lb - best_cv_model.oof
if gap > 0.01:  # Large gap
    print("CV-LB gap > 0.01: Model is overfitting CV")
    print("Try: simpler model, more regularization, fewer features")
elif gap < 0.005:  # Small gap
    print("CV-LB gap < 0.005: Model is generalizing well")
    print("Continue iterating on CV")
else:  # Medium gap
    print("CV-LB gap 0.005-0.01: Moderate overfitting")
    print("Submit and validate before continuing")
```

### Step 3: Compare to Baseline
```python
baseline_gap = baseline_model.lb - baseline_model.oof
if current_gap > 2 * baseline_gap:
    print("Gap is 2x worse than baseline. Revert to baseline")
```

## Empirical Pattern: Which CV Improvements Transfer to LB?

**Transfer well**:
- Simple model improvements (fewer features, more regularization)
- Better preprocessing (missing values, encoding)
- Ensemble of diverse algorithms (AutoGluon, multi-family GBDT)

**Transfer poorly**:
- Adding many engineered features (especially interactions)
- Aggressive hyperparameter tuning
- More model variants (15+ same family)
- Target encoding without nested CV
- Log-transform / Box-Cox on top of base features

## Anti-Patterns to Avoid

❌ **Don't trust CV alone** (always validate on LB)
❌ **Don't continue iterating** when CV improves but LB doesn't
❌ **Don't add complexity** without LB validation
❌ **Don't tune hyperparameters** extensively without LB check
❌ **Don't ignore the CV-LB gap** (it's a signal, not noise)

## Practical Workflow

```python
# Stage 1: Quick CV experiment
v1 = Experiment()
v1.run_cv()
v1.submit_to_lb()  # Always submit after first CV

# Stage 2: Iterate
for v in [v2, v3, v4]:
    v.run_cv()
    if v.oof > previous.oof:
        v.submit_to_lb()  # Submit if CV improved
        if v.lb < previous.lb:
            print("CV up but LB down. Revert to previous version")
            v = previous  # Revert

# Stage 3: Final
best_lb_version = max(submitted_versions, key=lambda v: v.lb)
best_lb_version.save_submission()
```

## Related Skills

- **ml-sweet-spot** — When to stop adding complexity
- **autogluon-first** — AutoGluon is robust to CV-LB gap
- **multi-model-diversity** — Diversity reduces overfitting to CV
- **controlled-submission-experiment** — How to test changes safely
- **progressive-verification-debugging** — Validate each step

## Validation Source
- House Prices: 0.009 gap (V7), 1.02 gap (V11/V12 catastrophic)
- Spaceship Titanic: 0.005-0.01 gap across all V2-V8 experiments
- Pattern observed across 2+ competitions: gap is real and consistent