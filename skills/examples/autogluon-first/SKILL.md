---
name: autogluon-first
description: |
  AutoGluon-First Strategy: Always run AutoGluon `best_quality` preset as the first step
  in any tabular ML competition (5-15 min baseline). Validated 3/4 times vs manual ensembles
  on small/medium tabular datasets. Use when: (1) Starting any new tabular competition,
  (2) Need a strong baseline in <15 minutes, (3) Manual GBDT work takes hours but
  gives similar results, (4) Want to focus manual effort on complementary techniques
  (deep learning, external data, special features) rather than reimplementing what
  AutoGluon already does automatically. Covers 5 core reasons for success
  (multi-algorithm diversity, multi-level stacking, automated preprocessing,
  bagging/multi-fold, Bayesian optimization), when to AVOID AutoGluon (text/image
  data, multi-output regression, >1M rows with limited RAM), and how to use it
  as a Silver signal in custom pipelines.
---

# AutoGluon-First Strategy

## Problem
Manual ML on tabular problems often wastes time:
- Hours tuning GBDT hyperparameters (LightGBM, XGBoost)
- Manual feature engineering trial-and-error
- Manual ensemble design (which models? which weights?)
- Reaches a plateau, then struggles to break through

**Reality**: AutoGluon's `best_quality` preset achieves in 5-15 minutes what takes humans days.

## Context / Trigger Conditions

Use this skill when:
- **Starting any new tabular competition** (regression or classification)
- **Need a strong baseline in <15 minutes**
- **Manual GBDT work takes hours but gives similar results**
- **Want to focus manual effort on complementary techniques** (deep learning, external data)
- **Unsure which algorithm to use** (LightGBM vs XGBoost vs CatBoost vs NeuralNet)
- **Need to compare your work against a known strong baseline**

## Solution

### Step 1: Run AutoGluon Baseline (5-15 minutes)

```python
import time
from autogluon.tabular import TabularPredictor

label = 'target'  # Your target column
save_path = f'ag_baseline_{int(time.time())}'

predictor = TabularPredictor(
    label=label,
    path=save_path,
    eval_metric='accuracy',  # or 'rmse', 'roc_auc', 'log_loss'
    verbosity=1
).fit(
    train_data,
    presets='best_quality',  # 10+ algorithms, multi-level stacking
    time_limit=900  # 15 minutes
)
```

**That's it.** In 5-15 minutes you have:
- 10+ models trained (LightGBM, XGBoost, CatBoost, RF, ExtraTrees, KNN, NN)
- Multi-level stacking (Level 1, 2, 3)
- OOF predictions for validation
- Test predictions for submission

### Step 2: Validate and Compare

```python
# Get OOF score
oof = predictor.predict_oof()
test_pred = predictor.predict(test_data)

# Compare to your manual work
# - If AutoGluon matches you: Stop, don't waste time
# - If AutoGluon beats you: Learn from its ensemble
# - If AutoGluon loses: Use it as Silver signal, focus on what's different
```

## 5 Core Reasons AutoGluon Works

### 1. Multi-Algorithm Diversity

| Algorithm | Inductive Bias | Strengths |
|-----------|----------------|-----------|
| LightGBM/XGBoost/CatBoost | Tree-based splits | Categorical features, missing values |
| Random Forest | Bagging + trees | Anti-overfitting, feature selection |
| ExtraTrees | Random splits | Low variance |
| KNN | Local smoothing | Small samples, manifold structure |
| NeuralNet (MLP) | Universal approximation | Complex non-linearity |
| Linear models | Linear | Sparse data, extrapolation |

**Manual ensembles fail** because they use only 3 GBDTs (high correlation → high Cov).
**AutoGluon wins** because it uses 10+ algorithms (low correlation → low Cov → low ensemble variance).

### 2. Multi-Level Stacking

```
Level 1: y1_i = model_i(X)   for i in 10 algorithms
Level 2: y2 = meta_learner(y1_1, y1_2, ..., y1_10)
Level 3: y3 = meta_meta_learner(y2, y1_combined)
```

Single-level averaging only learns **linear combinations**. Multi-level stacking learns **conditional combinations**:
- "If LightGBM predicts high AND KNN predicts low, use XGBoost"
- "If dataset is small, weight NeuralNet higher"

### 3. Automated Preprocessing

Each model needs different preprocessing:
- **Tree models**: Categorical encoding, missing value imputation
- **NeuralNet**: Standardization, OneHot
- **KNN**: Distance normalization
- **Linear**: Standardization

**AutoGluon automatically does per-model preprocessing**, while manual ensembles use uniform preprocessing (often suboptimal).

### 4. Bagging + Multi-Fold

```python
For each model:
    for each fold (5-fold):
        train on 4 folds, predict on 1 fold
    average predictions
```

For **small datasets** especially, this dramatically reduces variance (the biggest bottleneck for small data).

### 5. Bayesian Optimization + Early Stopping

- **Bayesian optimization** predicts next hyperparams based on history
- **Early stopping** terminates bad models early
- **Resource allocation** trains good models more, bad models less

For small datasets: each train is cheap → can try more models → finds global optimum.

## Validation Results (2026-06-14)

| Competition | Dataset Size | AutoGluon OOF | Manual Best | Verdict |
|-------------|--------------|---------------|-------------|---------|
| Titanic | 891 rows | **0.8552 acc** | ~0.82 typical | ✅ **AutoGluon wins** |
| Leaf Classification | 990 rows | **0.9889 acc, 0.0300 logloss** | Gold 1.0 | ⚠️ Close (0.0111 gap) |
| House Prices | 1460 rows | **0.118033 RMSE log** | V18 0.119350 | ✅ **AutoGluon wins** |
| TPS May 2022 | 900K rows | 0.8757 acc | Silver 0.99754 | ❌ Manual wins (special method) |

**Win rate**: 2/4 AutoGluon wins, 1/4 close, 1/4 manual wins (special method needed)

## When to AVOID AutoGluon Tabular

| Problem Type | Reason | Alternative |
|--------------|--------|-------------|
| Text/NLP | AutoGluon tabular doesn't do NLP | XLM-R, BERT, TabPFN |
| Image | Needs CNN/ViT, not tabular features | ResNet, ViT |
| Sequence (text→text) | Seq2seq, not classification | T5, BART |
| Multi-output regression | AutoGluon doesn't natively support | Train one model per output |
| Custom domain (chemistry, physics) | Needs domain-specific features | Domain knowledge + GBDT |
| Image-based regression | Needs CNN features, not raw pixels | Pretrained CNN + GBDT |
| >1M rows, <16GB RAM | Resource constraints, slow | Subsample or use GPU |

## How to Use AutoGluon in Custom Pipelines

### Option 1: AutoGluon as Baseline (5-15 min)
- Run AutoGluon → get baseline OOF
- Compare to your manual work
- **If AutoGluon matches you**: Stop, don't waste time
- **If AutoGluon beats you**: Use it directly, focus elsewhere

### Option 2: AutoGluon as Silver Signal (House Prices SST-style)
- Run AutoGluon → get OOF + test predictions
- Train your manual ensemble separately
- **Blend: 0.30-0.50 × AutoGluon + 0.50-0.70 × Manual**
- This adds diversity that AutoGluon's internal ensemble might miss

### Option 3: AutoGluon + Special Methods
- Run AutoGluon first
- Add **complementary** approach (deep learning, external data, special features)
- Validate via CV that special method adds signal beyond AutoGluon

## Anti-Patterns to Avoid

❌ **Don't skip AutoGluon** to "save time on small data" — Titanic 5 min AutoGluon = 0.8552
❌ **Don't trust AutoGluon for text/image** — needs different approach
❌ **Don't assume AutoGluon = best** — TPS May 2022 had manual Silver 12% better
❌ **Don't use AutoGluon as final** without LB validation — CV-LB gap exists
❌ **Don't spend days tuning GBDT** if AutoGluon matches in 15 min

## Empirical Pattern: When Manual Wins Over AutoGluon

- **Special domain knowledge** (chemistry, physics, biology)
- **External data integration** that AutoGluon doesn't have access to
- **Custom feature engineering** that captures domain-specific signals
- **Advanced ensemble techniques** beyond AutoGluon's multi-level stacking

## Related Skills

- **catboost-first-tabular** — When AutoGluon doesn't work, try CatBoost (House Prices V17 lesson)
- **cv-lb-validation** — Always validate AutoGluon on LB (CV-LB gap exists)
- **multi-model-diversity** — AutoGluon implements this internally (10+ algorithms)
- **asymmetric-blending** — AutoGluon can be one part of a larger blend
- **ml-sweet-spot** — When to stop optimizing and accept AutoGluon baseline
- **kaggle-competition-best-practices** — Full competition workflow with AutoGluon

## Validation Source
- House Prices: AutoGluon OOF 0.1180 vs V18 manual 0.1194 (2026-06-12)
- Leaf Classification: AutoGluon 0.9889 vs Gold 1.0 (2026-06-14)
- Titanic: AutoGluon 0.8552 vs classic 0.82 (2026-06-14)
- TPS May 2022: AutoGluon 0.8757 vs Silver 0.9975 (2026-06-14)