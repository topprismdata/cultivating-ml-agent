---
type: Lesson
title: S6E4 - 外部预测主导 vs 自训练 (2026-04)
description: S6E4 最佳方案 (LB 0.98150) 完全由外部预测主导，自训练模型被稀释。
tags: [lesson, external-data, signal-dilution, s6e4, quality_over_quantity]
timestamp: 2026-04-15T00:00:00Z
---

# S6E4 - 外部预测主导 vs 自训练

## 关键数字

| 实验 | 做法 | LB |
|---|---|---|
| R15 | 13 模型 + stacking (全自训练) | 0.97847 |
| R16 | 5-way external blend (首次外部有帮助) | 0.97901 |
| R17 | **Nina's 23 sources + schema8 (最佳)** | **0.98150** |
| R17 breakdown | 4-源 ensemble | 0.98145 |
| | 23-源 ensemble | 0.98115 (反而下降!) |

## 启示

1. **信号稀释**: 23 个源 (0.98115) < 4 个源 (0.98145) — 甜区在 **4-6 个高质量源**
2. **外部预测主导**: 最终方案完全由外部预测驱动，自训练模型被稀释
3. **Quality > Quantity**: 添加低于共识阈值的源产生负边际价值
4. **R12 / R13 反例**: 迭代伪标签 + stacking + threshold 捆绑，**都比 R11 差**——"改进组合"反而破坏已找到的好解

## 关联

- [16 条 Layer 3 原则](../16-principles.md) → work_smart_not_hard, quality_over_quantity, local_optimum_trap
- [S6E4 - Irrigation (源比赛)](../competitions/s6e4.md)
- [S6E5 - 外部数据 ROI 7x](s6e5_external_data_fusion.md) — 同主题，对比

---

#lesson #external-data #signal-dilution #s6e4