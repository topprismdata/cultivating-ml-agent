---
type: Competition
title: Jaguar Re-Identification Challenge
description: 已结束图像比赛，V8 模型 Private LB 0.960 (排名 #7 / 348 teams)。
tags: [competition, jaguar, re-identification, image, cosine-similarity]
timestamp: 2026-03-14T00:00:00Z
---

# Jaguar Re-Identification Challenge

> **状态**: 已结束 | **最佳**: Private LB 0.960 (V8) | **排名**: #7 / 348 teams

## 竞赛信息

- **类型**: Image Re-Identification (cosine similarity)
- **指标**: mean average precision (mAP) — 不是 AUC
- **冠军**: ButWhy? 0.979 | 亚军: Bull 0.978 | 季军: Elmir Mamedov 0.972
- **差距**: 距冠军 -0.019

## 提交轨迹（8 次提交）

| 版本 | Public LB | 关键变化 |
|---|---|---|
| baseline | 0.929 | 初始 baseline |
| improved | 0.94 | 改进特征 |
| tta | 0.95 | TTA (test-time augmentation) |
| reranking | 0.96 | Re-ranking 后处理 |
| **V8 (best)** | **0.960** | **最终模型** |
| windows_model | 0.95 | Windows-specific 微调 |

## 关键技术

- **Backbone**: EVA02 (large vision transformer) 或类似强 backbone
- **TTA**: 翻转 + 多尺度
- **Re-ranking**: k-reciprocal re-ranking 是 ReID 常见提分手段
- **Submission**: cosine similarity matrix + rank-based 输出

## 与 ARC Prize / Vesuvius 的区别

- **ARC Prize**: Python 编程题（提交代码，不是预测）
- **Vesuvius**: 2D 图像分割（卷纸 + mask）
- **Jaguar**: 图像检索（query vs gallery）

> 这三个都不是表格比赛，跳过 AutoGluon 重跑。

## 关联

- [16 条 Layer 3 原则](../16-principles.md) → quality_over_quantity (8 次提交而非 18)
- [S6E2 - Heart Disease](s6e2.md) — 对比表格 vs 图像的提交格式差异

---

#competition #jaguar #reid #image #done