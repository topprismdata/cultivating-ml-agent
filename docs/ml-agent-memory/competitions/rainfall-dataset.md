---
type: Competition
title: Rainfall Dataset - 9-day quick validation competition
description: Community tabular binary classification (Kudos). AutoGluon best_quality + FE = LB 0.86564. Validated S6E5 adversarial lesson, S6E6 single-best pattern, discovered random k-fold overestimates temporal data.
tags: [competition, rainfall-dataset, binary-classification, autogluon, okf-demo, kaggle-validated]
timestamp: 2026-06-21T00:00:00Z
---

# Rainfall Dataset

> **状态**: SHIPPED + Public LB **0.86564** (2026-06-21, 9 days before deadline)
> **指标**: AUC (binary, proba submission)
> **数据**: 2190 train (6 years) / 730 test (2 years), 11 numeric features

## 关键数字

| Submission | Method | Public LB | Δ vs best |
|---|---|---|---|
| **submission_v2_all.csv** | v2 + FE + 6 years | **0.86564** ⭐ | best |
| submission_3way_rank.csv | Rank avg blend | 0.85974 | -0.0059 |
| submission_3way_avg.csv | Mean blend | 0.85787 | -0.0078 |
| submission_proba.csv | v1 baseline (random k-fold) | 0.85277 | -0.0129 |
| submission_hard.csv | Hard label (wrong format) | 0.76133 | -0.1043 |

## 验证的 lessons

1. **S6E5 adversarial validation**: AUC 0.50 → no shift ✅
2. **S6E6 single > blend**: v2 alone beats 3-way blend ✅
3. **S6E4 mirror bug**: proba for AUC, hard for accuracy ✅
4. **AutoGluon best_quality**: 10 min gives 0.85277 baseline ✅
5. **GSD 5-phase on live comp**: validated end-to-end ✅

## 新发现

### Random k-fold overestimates temporal data (NEW)
- Random OOF AUC: 0.9041 (using 5-fold within years 1-6)
- Year-6 holdout AUC: 0.8803 (more honest)
- Public LB: 0.86564 (real)
- Gap: random OOF vs LB = **0.0513** ⚠️ (HUGE!)

**Why**: Random k-fold sees all 6 years in both train AND val. Model learns year-agnostic patterns. But test is from entirely new years (7-8), so the model overfits to within-year patterns.

**Decision rule**: For any comp with `day` or temporal features, use TIME-AWARE CV (last X% of train as holdout).

### Feature engineering helps at LB even when year-6 holdout worsens
- v1 (no FE) year-6 AUC: 0.8803
- v2 (with FE: cloud_humidity, cloud_sunshine_ratio, etc.) year-6 AUC: 0.8777 (-0.0026)
- v1 LB: 0.85277
- v2 LB: **0.86564** (+0.0129!)

**Why**: Year-6 holdout is a single point estimate (high variance with 365 samples). LB is more reliable signal. Feature engineering that captures genuine interactions will win at LB even if it doesn't help a single fold.

## 关键发现摘要

| Finding | Insight |
|---|---|
| Random k-fold > year-6 holdout > LB | Time-aware CV is essential for temporal data |
| Adversarial AUC = 0.50 | No feature distribution shift (year cycles are similar) |
| cloud + sunshine are top features | Inversely correlated (more cloud = less sun = more rain) |
| day, maxtemp have negative importance | Drop them |
| v2 with FE wins at LB | AutoGluon can't discover all interactions; manual FE helps |
| v2 alone > 3-way blend | Single best > ensemble (S6E6 schema8 pattern) |

## GSD 5-Phase Trace

| Phase | Output |
|---|---|
| DISCUSS | `.planning/01-discuss.md` (metric=AUC, format=proba, deadline 2026-06-30) |
| PLAN | `.planning/02-plans.md` (6-plan DAG) |
| EXECUTE | v1 baseline + adversarial + time-CV + v2 FE |
| VERIFY | `.planning/04-verify.py` 7/7 PASS |
| SHIP | `.planning/05-ship.md` (5 submissions, best 0.86564) |

## 关联

- [S6E2 - Heart Disease](s6e2.md) - 同样 binary AUC, 同样 AutoGluon-first
- [S6E4 - Irrigation](s6e4.md) - 同样格式陷阱 (proba vs hard label)
- [S6E5 - F1 Pit Stop](s6e5.md) - adversarial validation 起源
- [S6E6 - Stellar Class](s6e6.md) - single best > blend pattern
- [s6e2-lr-stacker-no-gain lesson](../lessons/s6e2-lr-stacker-no-gain.md)
- [s6e6-cv-lb-gap-stacker-overfit lesson](../lessons/s6e6-cv-lb-gap-stacker-overfit.md)
- [s6e5-adversarial-validation-failure lesson](../lessons/s6e5_adversarial_validation_failure.md)
- [submission-format-by-metric skill](../skills/submission-format-by-metric.md)

---

#competition #rainfall-dataset #binary-classification #autogluon #okf-demo #kaggle-validated #time-aware-cv #feature-engineering