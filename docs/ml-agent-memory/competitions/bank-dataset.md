---
type: Competition
title: Bank Dataset - Binary Classification with a Bank Dataset (clone)
description: 113-day binary classification (deposit subscription). AutoGluon + hyperparameter tuning = LB 0.97092. Validated manual FE doesn't help, UCI external fusion hurts.
tags: [competition, bank-dataset, binary-classification, autogluon, hyperparameter-tuning, okf-demo]
timestamp: 2026-06-22T00:00:00Z
---

# Bank Dataset

> **状态**: SHIPPED + Public LB **0.97092** (2026-06-22, 113 days before deadline)
> **指标**: AUC (binary, proba submission confirmed by sample_submission y=0.5)
> **数据**: 750K train / 250K test (3:1), 17 features (9 categorical + 7 numeric)

## 关键数字

| Submission | Method | OOF | Public LB | Δ vs baseline |
|---|---|---|---|---|
| **submission_tuned.csv** | **v5 hyperparameter tuning** | **0.96990** | **0.97092** ⭐ | **+0.00059** |
| submission_bag3_rank.csv | v3 3-seed bag rank avg | 0.9700 | 0.97036 | +0.00003 |
| submission_bag3_arith.csv | v3 3-seed bag arith mean | 0.9700 | 0.97035 | +0.00002 |
| submission_proba.csv | v1 AutoGluon baseline | 0.96958 | 0.97033 | baseline |
| submission_external.csv | v6 + UCI 45K real data | 0.96861 | not submitted | -0.00097 (worse) |
| submission_tabpfn.csv | v7 TabPFN 4K subsample | 0.9484 | not submitted | -0.022 (much worse) |
| v4 FE+TE | categorical interactions + target encoding | 0.96879 | not submitted | -0.0008 (worse) |

## 5 个实验的发现

### 1. v1 AutoGluon baseline (default best_quality)
- 11 models, OOF AUC 0.96958
- WeightedEnsemble_L3 best
- LB 0.97033 (gap -0.0007, no overfit)
- Class balance match (train 12.07% vs test pred 12.02%)

### 2. v3 3-seed bagging
- Per-seed Val AUC 0.9700 (very consistent)
- Bagged avg: 0.97036 LB (+0.00003 vs baseline)
- **S6E6 single > blend lesson reinforced again**

### 3. v4 FE + target encoding
- Added categorical interactions (job×education, marital×housing, etc.)
- Added target encoding (mean y per category with smoothing)
- OOF AUC **0.96879 (worse than baseline by 0.0008)**
- **Manual FE doesn't help when AutoGluon can auto-discover interactions**
- (Similar to rainfall-dataset v2 finding)

### 4. v5 hyperparameter tuning ⭐ WINNER
- Custom hyperparameter grids for XGBoost, LightGBM, CatBoost
- WeightedEnsemble_L4 with 15+ tuned models
- OOF AUC **0.96990** (vs baseline 0.96958)
- LB **0.97092** (+0.00059 over baseline)
- Top base: XGBoostTuned_BAG_L2 (0.969857)
- **Real improvement from tuning — not just stacker overfit**

### 5. v6 + UCI bank marketing (45211 real rows)
- Combined train (750K) + UCI (45K) = 795K
- OOF AUC **0.96861 (worse by 0.001)**
- UCI real data distribution != synthetic test distribution
- **S6E5 external data fusion lesson: REVERSE here — external data HURTS when distributions differ**
- Skipped submission

### 6. v7 TabPFN (subsample 4K)
- Val AUC 0.9484 (vs tuned 0.9699)
- TabPFN CPU inference too slow for 250K test (~2.5 hours estimated)
- Subsample too small for 750K-scale task
- Skipped submission

## 关键洞察

| Lesson | Insight |
|---|---|
| **Hyperparameter tuning > ensemble size** | +0.0006 from custom configs vs +0.00003 from bagging |
| **Single > blend** (S6E6 lesson) | bag3 +0.00003 is noise, tuning +0.0006 is real |
| **Manual FE often hurts** (rainfall lesson) | AutoGluon auto-discovers interactions |
| **External data fusion can hurt** | UCI real data ≠ synthetic test distribution |
| **TabPFN limited to small data** | Subsample loses too much signal for 750K tasks |
| **Class balance = proxy for distribution match** | Train 12.07% ↔ Test 12.02% (good) |

## GSD 5-Phase Trace

| Phase | Output |
|---|---|
| DISCUSS | `.planning/01-discuss.md` (AUC metric, no temporal split, 113 days) |
| PLAN | `.planning/02-plans.md` (multiple experiments) |
| EXECUTE | 7 runs: v1, v3, v4, v5, v6, v7, v8 (in progress) |
| VERIFY | `.planning/04-verify.py` 7/7 PASS on all submissions |
| SHIP | `.planning/05-ship.md` (4 valid submissions, best 0.97092) |

## Anti-patterns Avoided

- ❌ Hard label for AUC metric (S6E2 lesson) → used proba → LB 0.97033 baseline
- ❌ Random k-fold on temporal data (rainfall lesson) → not applicable (no temporal)
- ❌ Self-trained OOF stacker (S6E6 lesson) → using AutoGluon internal ensemble
- ❌ Manual FE without testing → v4 FE+TE skipped when OOF worse
- ❌ External data without distribution check → v6 UCI skipped

## 关联

- [S6E6 - Stellar Class](s6e6.md) — single > blend pattern (validated here)
- [S6E2 - Heart Disease](s6e2.md) — AutoGluon-first pattern
- [S6E4 - Irrigation](s6e4.md) — external data fusion success (counter-example)
- [S6E5 - F1 Pit Stop](s6e5.md) — adversarial validation lesson
- [rainfall-dataset](rainfall-dataset.md) — manual FE hurts (validated here)
- [s6e6-cv-lb-gap-stacker-overfit](../lessons/s6e6-cv-lb-gap-stacker-overfit.md)
- [submission-format-by-metric skill](../skills/submission-format-by-metric.md)

---

#competition #bank-dataset #binary-classification #autogluon #hyperparameter-tuning #okf-demo