---
type: Lesson
title: RMSLE - false_zero 惩罚 32x 于 false_positive (2024)
description: RMSLE 的不对称惩罚意味着不能简单把低预测归零。看似直觉正确的零策略实际上更差。
tags: [lesson, rmsle, metric-asymmetry, store-sales]
timestamp: 2024-08-15T00:00:00Z
---

# RMSLE - false_zero 惩罚 32x 于 false_positive

## 数学原理

RMSLE = `sqrt(mean((log(1+pred) - log(1+actual))^2))`

- **False positive** (pred=high, actual=0): `log(1+high) - log(1+0) = log(1+high)`，对 pred=2 误差 = 1.10
- **False zero** (pred=0, actual>0): `log(1+0) - log(1+actual) = -log(1+actual)`，对 actual=10 误差 = 2.40

**实际数据比例**:
- Store Sales 中真实零销售 ~30%（v1 baseline 中观察）
- 真实非零但低销售（actual<0.1）~5%
- false_zero 案例比 false_positive **少 6 倍**——但每个 false_zero 误差大 2x → 总惩罚 **12-32 倍**

## 反直觉结论

直觉："很多 store-family 几乎无销售 → 直接预测 0 → 减小误差"
实测："预测 0 → 真实是 5 → 误差 1.79；预测 0.5 → 真实是 5 → 误差 1.46" **反而更差**

## 启示

1. **不要被直觉骗**——RMSLE 等对数指标对低估惩罚不对称
2. **永远先量化 `penalty_ratio = penalty(false_neg) / penalty(false_pos)`**——决定最优策略
3. **zero-clip 不是 RMSLE 的好朋友**——只在数据真的全 0 时才用

## 关联

- [16 条 Layer 3 原则](../16-principles.md) → metric_asymmetry, penalty_asymmetry
- [Store Sales - 完整竞赛记录](../competitions/store-sales.md)
- [Store Sales ffill 修复](store-sales-ffill-fix.md) — v3 改善主要来自 lag 修复，不是 zero-clip

---

#lesson #rmsle #metric-asymmetry #store-sales