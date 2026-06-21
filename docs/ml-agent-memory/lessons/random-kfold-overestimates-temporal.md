---
type: Lesson
title: Random k-fold overestimates temporal data by 0.05+ AUC (rainfall-dataset validation, 2026-06-21)
description: Random k-fold CV overestimates AUC by 0.05+ on temporal data (train years 1-6, predict year 7+). Use time-aware CV (last year as holdout) for any comp with day/timestamp features.
tags: [lesson, rainfall-dataset, time-aware-cv, temporal-data, cv-overestimate, kfold-bias]
timestamp: 2026-06-21T00:00:00Z
---

# Random k-fold overestimates temporal data

## 关键数字（rainfall-dataset 2026-06-21）

| CV method | AUC | Gap vs LB |
|---|---|---|
| Random k-fold (5-fold) | 0.9041 | **+0.0513** ⚠️ |
| Year-6 time-holdout | 0.8803 | **+0.0147** ✅ |
| **Public LB** | **0.86564** | (reference) |

**Δ between random OOF and time-holdout: 0.024** (random OOF overestimates by ~2.4% AUC)

## 为什么 random k-fold 过估计

1. **Random k-fold** sees all years in both train AND val
2. Model learns year-agnostic patterns (e.g., "high humidity → rain")
3. But **test is from future years** (7-8), unseen at train time
4. Year-6 holdout forces model to extrapolate to new year → more honest estimate

## When to use time-aware CV

| Feature | Use time-aware CV? |
|---|---|
| `day`, `month`, `year`, `timestamp`, `date` | ✅ YES |
| Sequential ID column | ✅ YES (might be time-ordered) |
| Train rows ≤ test rows (rolling forecast) | ✅ YES |
| All other tabular | ❌ random k-fold OK |

## Decision rule

```python
# For temporal data: use TimeSeriesSplit or last 20% as holdout
from sklearn.model_selection import TimeSeriesKFold

# OR manual split
if "day" in features or "date" in features:
    train["__year__"] = train["id"] // 365  # or pd.to_datetime + dt.year
    val = train[train["__year__"] == train["__year__"].max()]
    train_remain = train[train["__year__"] < train["__year__"].max()]
    # train on train_remain, validate on val
```

## Lessons from rainfall run

1. **AutoGluon best_quality** on all 6 years: LB 0.86564 (with FE)
2. **AutoGluon best_quality** on years 1-5 + year-6 holdout: AUC 0.8803
3. **AutoGluon best_quality** random k-fold on all 6 years: AUC 0.9041 (overestimates)

**Takeaway**: Use time-aware CV to estimate true LB performance for temporal data. Random OOF is too optimistic.

## 关联

- [Rainfall Dataset](../competitions/rainfall-dataset.md) - 验证案例
- [S6E5 - Adversarial Validation Failure](../lessons/s6e5_adversarial_validation_failure.md) - 互补（adversarial 检验特征 shift，本 lesson 检验 OOF 真实性）
- [S6E6 - CV-LB gap stacker overfit](../lessons/s6e6-cv-lb-gap-stacker-overfit.md) - 另一类 CV-LB gap

## Anti-pattern

❌ Random k-fold for temporal data → overestimates → bad decisions based on inflated scores
✅ Time-aware CV → honest OOF → correct model selection

---

#lesson #rainfall-dataset #time-aware-cv #temporal-data #cv-overestimate #kfold-bias