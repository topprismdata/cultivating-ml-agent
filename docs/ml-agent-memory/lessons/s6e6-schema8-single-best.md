---
type: Lesson
title: S6E6 - schema8 单模型 0.96719 > 所有 blend (2026-06-18)
description: S6E6 Stellar Class: schema8 单模型 OOF BAC 0.9668 / LB 0.96719，超过所有 2/3/4-way blend。
tags: [lesson, s6e6, blend, signal-saturation, multi-class]
timestamp: 2026-06-18T00:00:00Z
---

# S6E6 - schema8 单模型 > 所有 blend

## 关键数字

| Source | OOF BAC | Public LB |
|---|---|---|
| **schema8 单模型** | **0.9668** | **0.96719** ⭐ |
| AutoGluon best_quality | 0.9630 | 0.95021 |
| xgb_v8_star | 0.9615 | (未单独提交) |
| cb_v9_star | 0.9556 | (未单独提交) |
| realmlp | 0.9434 | (未单独提交) |
| 3-way blend (xgb+cb+schema8) | 0.9637 | 0.96379 |
| 4-way blend (+realmlp) | 0.9620 | (未提交) |
| 用户历史最佳 (majority ensemble) | n/a | 0.97077 |

## 关键洞察

**schema8 单模型 0.96719 LB 是这次会话的最佳**。即使 blend 在 OOF 上"看起来"好（0.9637），实际 LB 也只有 0.96379——比 schema8 单模型还差 0.0034。

## 为什么 blend 失败

1. **信号饱和**: schema8 已经吃掉了大部分信号——其他模型都是次优近似
2. **equal-weight 不优化 BAC**: 加权平均不保证 LB 提升
3. **CV gap 不等于 blend gain**: blend 在 OOF 上稳定，但 LB 可能因分布偏移而变差

## 启示

| 原则 | 含义 |
|---|---|
| **单模型 OOF 高不一定是 LB 高** | schema8 OOF 0.9668 → LB 0.96719 (gap 0.0004, 极好) |
| **Blending 不保证 LB 提升** | 3-way blend OOF 0.9637 < schema8 OOF 0.9668 |
| **OOF 高的 blend 应该直接 ship 单模型** | 不要为了 blend 而 blend |

## 关联

- [S6E6 - 完整竞赛记录](../competitions/s6e6.md) (待写)
- [16 条 Layer 3 原则](../16-principles.md) → quality_over_quantity, signal_dilution
- [S6E4 - 外部预测主导](s6e4_external_blend.md) — 同主题（外部/单源 > 过度 blend）

---

#lesson #s6e6 #blend #signal-saturation #multi-class