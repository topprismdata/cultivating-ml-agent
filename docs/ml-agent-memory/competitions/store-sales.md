---
type: Competition
title: Store Sales Time Series Forecasting
description: 时间序列预测（RMSLE）。v3 ffill 修复把 LB 从 2.67 拉到 1.90（关键诊断教训）。
tags: [competition, store-sales, time-series, rmsle, agentic]
timestamp: 2024-08-31T00:00:00Z
---

# Store Sales Time Series Forecasting

> **状态**: 已结束 | **最佳**: Public LB 1.90248 (v3 ffill) | **指标**: RMSLE

## 竞赛信息

- **类型**: Time Series Regression (per store × family × date)
- **指标**: RMSLE (Root Mean Squared Logarithmic Error)
- **数据**: ~3M 行训练 (2013-01-01 ~ 2017-08-15) + 外部数据 (油价、节假日、交易)
- **预测目标**: 2017-08-16 ~ 2017-08-31, 28,512 个 (store, family, date) 组合
- **TOP 排名**: ~0.33 (按公开榜)

## 提交轨迹（3 个关键版本）

| 版本 | 模型 | CV RMSLE | Public LB | 说明 |
|---|---|---|---|---|
| v1 | LightGBM (NaN) | 0.3751 | 2.67192 | Round 1 Baseline — test lag 特征 NaN 级联 |
| v2 | LightGBM (sales=0) | 0.3750 | 2.83107 | sales 填 0 → 更差（lag 全 0）|
| **v3** | **LightGBM (ffill)** | **0.3750** | **1.90248** | **ffill 修复 → 大幅改善** |
| v4 | LGB+XGB+CB Ensemble | - | - | Round 2 集成（未完成）|

## 关键教训（已提取为 lesson concept）

1. **lag NaN cascade** → `preds.mean()/train.mean()` 0.11 立即诊断 → 用 ffill 修复（v1 → v3 改善 -0.77 LB）
2. **RMSLE false-zero 惩罚不对称** → 不能简单归零
3. **Agentic 架构**（Feature Eng / Model / Ensemble Agent + NotebookLM RAG）是这套 workflow 的特色

## 关联

- [16 条 Layer 3 原则](../16-principles.md) → simple_diagnostic, work_smart_not_hard, metric_asymmetry
- [Lesson: store-sales ffill 修复](lessons/store-sales-ffill-fix.md)
- [Lesson: RMSLE 零策略陷阱](lessons/rmsle-zero-threshold-asymmetry.md)
- 项目: `~/projects/kaggle-store-sales/` (NotebookLM 智囊架构)

---

#competition #store-sales #time-series #rmsle