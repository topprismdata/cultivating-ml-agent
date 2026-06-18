---
type: Lesson
title: S6E5 - 对抗净化无效，AUC≈0.50 (2026-05)
description: S6E5 v16 净化后 AUC ≈ 0.50，train/test 已对齐，净化没改善。
tags: [lesson, adversarial-validation, s6e5, distribution_mismatch]
timestamp: 2026-05-24T00:00:00Z
---

# S6E5 - 对抗净化无效，AUC≈0.50

## 关键数字

| | 值 |
|---|---|
| 对抗验证 AUC | **0.50309** |
| 净化前 (439K 全部) | CV 0.95290 |
| 净化后 (60K) | CV 0.95290 (持平) |
| LB 改善 | 0 |

**AUC ≈ 0.5 意味着 train 和 test 分布几乎不可区分**——净化徒劳。

## 启示

- **对抗 AUC ≈ 0.50 → 立即停止净化**，不要浪费时间
- 把时间花在特征工程 / 集成 / 外部数据上
- 与 S6E5 v16 同步：发现 CV-LB gap ~0.01 不是分布不匹配造成的

## 决策规则

```
跑对抗验证 (train vs test 分类)
│
├─ AUC < 0.55 → 分布几乎对齐，净化无效，停止
├─ 0.55 < AUC < 0.70 → 弱分布差异，净化谨慎（可能伤害）
└─ AUC > 0.70 → 强分布差异，净化必要
```

## 关联

- [adversarial-validation skill](../skills/adversarial-validation.md)
- [16 条 Layer 3 原则](../16-principles.md) → adversarial_validation_limitation
- [S6E5 - F1 Pit Stop (源比赛)](../competitions/s6e5.md)

---

#lesson #adversarial-validation #s6e5