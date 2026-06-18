---
type: Lesson
title: Store Sales - v1→v3 ffill 修复，LB -0.77 (2024)
description: lag NaN cascade 是时间序列比赛的常见 bug；一行 mean_ratio 诊断 + ffill 修复 = 0.77 LB 改善。
tags: [lesson, time-series, lag-feature, simple-diagnostic, store-sales]
timestamp: 2024-08-15T00:00:00Z
---

# Store Sales - v1→v3 ffill 修复

## 关键数字

| 版本 | Public LB | Δ |
|---|---|---|
| v1 (NaN lags) | 2.67192 | baseline |
| v2 (sales=0) | 2.83107 | **+0.16（更差）** |
| **v3 (ffill)** | **1.90248** | **-0.77** ⭐ |

**单步修复 -0.77 LB**——比外部数据融合还大的改进。

## Bug 诊断（一行诊断 → 5 秒定位）

```python
# 关键诊断 (ml-sweet-spot 原则: simple diagnostic over complex debugging)
mean_ratio = test_preds.mean() / train_actual.mean()
# 输出: 0.11（应该是 ~1.0）
# → 模型在预测一个比真实销售低 9 倍的值 → 必然是 lag 特征问题
```

**lag 特征 NaN cascade**: test 集的 lag_1 用前一天的 sales，但前一天的 sales **也是预测的**，前前一天也是……一路 NaN 到训练集边缘，导致整个 lag 链失效。

## 修复

用前向填充（ffill）替代 NaN：
- 取最近已知值（不一定是 lag_1）
- 或者使用 rolling mean 替代 raw lag
- 或者用 last_known_value 全局填充

## 启示

1. **永远先写 mean_ratio 诊断**——5 秒定位问题，比看 1000 行预测分布快 100 倍
2. **NaN 级联是时间序列的隐形杀手**——所有 lag/rolling/ewm 特征都受影响
3. **不要用 sales=0 填 NaN**——会引入 bias，v2 反而更差就是证据

## 关联

- [16 条 Layer 3 原则](../16-principles.md) → simple_diagnostic, distribution_mismatch
- [Store Sales - 完整竞赛记录](../competitions/store-sales.md)
- [S6E5 - 对抗净化无效](s6e5_adversarial_validation_failure.md) — 同主题"诊断先于模型"
- [ml-sweet-spot skill](../skills/ml-sweet-spot.md)

---

#lesson #time-series #lag-feature #simple-diagnostic #store-sales