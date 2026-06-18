---
type: Experiment Index
title: 实验记录总览
description: '28 条实验记录时间线汇总（来自 experiment_log.jsonl）。'
tags: [experiment, timeline, s6e5]
timestamp: 2026-06-15T10:05:55Z
---

# 实验记录总览

> 共 28 次实验，来源: `experiment_log.jsonl`

---

## S6E5 - F1 Pit Stop Prediction

### exp_000_baseline
- **CV/LB**: 0.9432 / 0.9432
- **做法**: LightGBM baseline，无外部数据，无 TE
- **诊断**: 建立 F1 领域特征基线
- **标签**: baseline, lgb

### exp_001_orig_features
- **CV/LB**: 0.9604 / 0.9504
- **做法**: LGB + 原始数据融合 (weight=0.35) + TE + group stats (59 feat)
- **诊断**: 外部数据融合 +0.007 CV，最大单一改进
- **关键洞察**: `external_data_fusion_largest_single_step_improvement`
- **标签**: external_data, target_encoding, domain_features

### exp_002_multimodel
- **CV/LB**: 0.9596 / 0.9505
- **做法**: LGB+XGB+CB weighted ensemble (3 seeds × 3 models = 9)
- **诊断**: 多模型略差于 exp_001 单模型
- **关键洞察**: `model_diversity_only_helps_when_single_model_optimized`
- **标签**: multi_model, ensemble

### exp_026_hillclimb
- **CV/LB**: 0.9622 / 0.9512
- **做法**: Hill-climbing 15 models (9 GBDT + 3 MLP + 3 TabPFN)
- **诊断**: 最佳 CV 0.9622，LB 改善到 0.9512
- **标签**: hill_climbing, ensemble, nn_diversity

### exp_035_tabpfn
- **CV/LB**: 0.9524 / 0.9510
- **做法**: 9 GBDT + 3 MLP ensemble，OOF-based blending
- **诊断**: CV 下降，LB 略微改善
- **标签**: nn, ensemble, mlp

### v15f_domain_features
- **CV/LB**: 0.9536 / 0.9526
- **做法**: GBDT (3 seeds × 3 models) + RealMLP (n_ens=16)，103 领域特征
- **诊断**: GBDT OOF AUC 0.9529，RealMLP MPS SIGKILL，GBDT-only 提交
- **关键洞察**: `hardware_constraint_means_skip_nn_route`
- **标签**: domain_features, gbdt, realmlp, mps_crash

### v16_adv_validation
- **CV**: 0.9529
- **做法**: Adversarial validation → 选择 60K test-like 样本训练
- **诊断**: Adversarial AUC 0.50309 ≈ 0.50，train/test 已对齐，净化不改善
- **关键洞察**: `adversarial_validation_a050_no_benefit`
- **标签**: adversarial_validation, domain_features

---

## S6E4 - Irrigation Prediction

| 版本 | LB | 关键变化 | 洞察 |
|------|-----|---------|------|
| R01 | 0.97476 | LightGBM baseline 38 feat | 建立基线 |
| R02 | 0.97586 | Target encoding | +0.00110 |
| R04 | 0.97656 | Pairwise TE: 8 cats × 11 nums = 135 feat | +0.00070 |
| R05 | 0.97730 | Multi-model + threshold opt | +0.00074 |
| R08 | 0.97782 | Pseudo-labeling (1 round, th=0.90) | |
| R09 | 0.97785 | 10-model + stacking | CV-LB gap 0.00176 |
| **R12** | **0.97742** | **Iterative pseudo (3 rounds)** | **WORSE** — iterative hurts |
| **R13** | **0.97750** | **stacking + threshold 捆绑** | **WORSE** — bundled changes |
| R15 | 0.97847 | 13-model + stacking | 最佳自训练 |
| R16 | 0.97901 | 5-way external blend | 首次外部有帮助 |
| **R17** | **0.98150** | **Nina's 23 sources + schema8** | **BEST** — external dominate |

**关键洞察**:
- R12: `iterative_pseudo_labeling_hurts_decreasing_threshold_noise`
- R13: `controlled_variable_principle_bundle_causes_uninterpretable`
- R17: `signal_dilution_23sources_less_than_4sources`

---

## Store Sales - Time Series

| 版本 | CV | LB | 关键变化 | 洞察 |
|------|-----|-----|---------|------|
| v1 | 0.3751 | 2.6719 | LightGBM + lag/rolling, NaN as feat | mean_ratio=0.11 → 灾难性低估 |
| v2 | 0.3750 | 2.8311 | Fill NaN with 0 | **更差** — false zero |
| v3 | 0.3750 | 1.9025 | Forward-fill lag features | preds.mean()/train.mean()=0.78 |

**关键洞察**:
- `lag_feature_nan_cascade_explodes_rmsle`
- `false_zero_rmsle_penalty_is_32x_false_positive`

---

## House Prices

| 版本 | CV | 关键变化 |
|------|-----|---------|
| exp_000 | 0.1267 | LightGBM baseline |
| exp_016 | 0.1147 | Polynomial: top 8 feat degree=2 → 28 interactions |
| **exp_017** | **0.1109** | **Remove outliers (SalePrice > 4 std, n=4)** — 最大改进 |
| exp_018 | 0.1151 | KernelRidge + SVR (7 models) — 边际收益有限 |

**关键洞察**: `outlier_removal_largest_single_step_improvement_small_dataset`

---

## Spaceship Titanic

| 版本 | CV | 关键变化 |
|------|-----|---------|
| exp_000 | 0.80 | LightGBM baseline |
| **exp_019** | **0.8151** | **Meta_lr stacking: 9 models** — 最佳 |

---

## 跨竞赛模式

### pairwise_target_encoding
- 有效于: s6e4, s6e5
- 效果: +0.001 ~ +0.002
- 条件: categorical × numerical 创建有意义的交互

### external_data_fusion
- 有效于: s6e4, s6e5
- 效果: +0.001 ~ +0.007
- 条件: 外部数据有相同目标定义和低噪声

### pseudo_labeling
- 有效于: s6e4
- 效果: 单轮 +0.001，迭代 -0.001
- 条件: 高置信度阈值 (≥0.90)，单轮

---

## 关联

- 主页: [ML Agent Memory Dashboard](../dashboard.md)
- 原则集: [16 条 Layer 3 原则](../16-principles.md)
- 案例: [s6e5 — F1 停车预测](../competitions/s6e5.md) | [s6e4 — 灌溉预测](../competitions/s6e4.md)

更新: 2026-05-24 | 实验数: 28
