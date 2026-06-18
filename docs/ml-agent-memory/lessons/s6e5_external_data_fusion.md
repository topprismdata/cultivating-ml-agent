---
type: Lesson
title: S6E5 - 外部数据融合 ROI 7x (2026-05)
description: S6E5 实验记录显示外部数据融合是最大单一改进 (+0.007 CV)，是 feature engineering 的 ~7x ROI。
tags: [lesson, external-data-fusion, s6e5, work_smart_not_hard]
timestamp: 2026-05-24T00:00:00Z
---

# S6E5 - 外部数据融合 ROI 7x

## 关键数字

| 实验 | 做法 | CV | Δ |
|---|---|---|---|
| exp_000 | LightGBM baseline | 0.9432 | baseline |
| exp_001 | + 外部数据融合 + 领域特征 | 0.9604 | **+0.007** ⭐ |

**单步改进 +0.007**，远大于后续 25 轮迭代中的任何一步（大多 < 0.001）。

## 启示

- **外部数据 > 内部特征工程**：花 1 小时找外部源 > 调 1 周超参数
- **前提**：公开数据集有相关字段（这里是赛车原始数据 + 公开赛事数据）
- **不适用**：数据完全私有（医疗、金融），无外部源

## 关联

- [16 条 Layer 3 原则](../16-principles.md) → work_smart_not_hard, leverage_principle
- [S6E5 - F1 Pit Stop (源比赛)](../competitions/s6e5.md)

---

#lesson #external-data #s6e5 #work-smart