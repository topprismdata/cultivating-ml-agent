---
type: Index
title: Lessons
description: 单场比赛 / 单次重跑发现的具体教训。一个 lesson concept 通常关联到一个或多个 skill concept 的更新。
tags: [index]
timestamp: 2026-06-18T00:00:00Z
---

# Lessons

> 单场比赛 / 单次重跑发现的具体教训。
> 每个 lesson 是一个独立的 concept，通常会反向更新一个或多个 skill concept。

## By competition

### S6E2 (Heart Disease, AutoGluon rerun)
* [S6E2 - AutoGluon best_quality 验证](s6e2_autogluon_first.md) (OOF 0.95554 / Private 0.95510, 15 分钟)
* [S6E2 - CatBoost 占 ensemble 68.8%](s6e2_catboost_first.md) (单模型 OOF ≈ 全 ensemble)
* [S6E2 - CV-LB gap 0.07 是格式错误不是过拟合](s6e2_cv_lb_gap.md) (CV-LB gap 警告阈值细化)
* [S6E2 - 概率 vs 0/1 提交，PR #5 起源](s6e2_submission_format.md) (新 skill 起源)

### S6E4 (Irrigation)
* [S6E4 - 外部预测主导 vs 自训练](s6e4_external_blend.md) (LB 0.98150, 23 源 < 4 源, signal dilution)

### S6E5 (F1 Pit Stop)
* [S6E5 - 外部数据融合 ROI 7x](s6e5_external_data_fusion.md) (单步 +0.007 CV, 最大单一改进)
* [S6E5 - 对抗净化无效，AUC≈0.50](s6e5_adversarial_validation_failure.md) (净化无改善，AUC < 0.55 立即停止)

### Store Sales (Time Series, RMSLE)
* [Store Sales - v1→v3 ffill 修复，LB -0.77](store-sales-ffill-fix.md) (lag NaN cascade, mean_ratio 0.11 诊断)
* [RMSLE - false_zero 惩罚 32x 于 false_positive](rmsle-zero-threshold-asymmetry.md) (对数指标不对称)

### S6E6 (Stellar Class, Multi-class BAC)
* [S6E6 - schema8 单模型 > 所有 blend](s6e6-schema8-single-best.md) (LB 0.96719, 单模型 OOF 0.9668 > 任何 blend)
* [OKF enrich 不适合 Kaggle 数据](okf-enrich-not-for-kaggle.md) (hand-craft 比 auto-enrich 强)

## By principle (跨比赛)

- **外部数据主导**: [S6E4](s6e4_external_blend.md), [S6E5](s6e5_external_data_fusion.md)
- **Quality > Quantity**: [S6E4 signal dilution](s6e4_external_blend.md)
- **CV-LB gap 警告阈值细化**: [S6E2](s6e2_cv_lb_gap.md)
- **autoML vs manual stacking**: [S6E2](s6e2_autogluon_first.md)
- **CatBoost dominance**: [S6E2](s6e2_catboost_first.md)
- **对抗净化决策规则**: [S6E5](s6e5_adversarial_validation_failure.md)
- **0/1 vs 概率提交**: [S6E2](s6e2_submission_format.md)
- **简单诊断优先**: [Store Sales ffill](store-sales-ffill-fix.md)
- **指标不对称决定策略**: [RMSLE zero 陷阱](rmsle-zero-threshold-asymmetry.md)
- **单源 > 过度 blend**: [S6E6 schema8](s6e6-schema8-single-best.md)
- **工具 vs 场景**: [OKF enrich 不适合 Kaggle](okf-enrich-not-for-kaggle.md)