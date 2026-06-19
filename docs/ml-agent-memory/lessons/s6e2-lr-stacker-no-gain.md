---
type: Lesson
title: S6E2 - LR stacker 没有增益（AutoGluon ensemble 已经很优）(2026-06-20)
description: 复制 cdeotte-style LR stacker 到 S6E2，结果比 AutoGluon WeightedEnsemble 差（0.9549 vs 0.9552），因为 AutoGluon ensemble 已经是最优组合。
tags: [lesson, s6e2, cdeotte-stacker, lr-stacker, signal-saturation]
timestamp: 2026-06-20T00:00:00Z
---

# S6E2 - LR stacker 没有增益

## 关键数字

| Source | OOF AUC | 备注 |
|---|---|---|
| AutoGluon WeightedEnsemble_L3 | **0.9552** | 7 个 base models 的加权组合 |
| LR stacker (LR_C0.1, this run) | 0.9549 | 同样 7 个 base 训练 LR |
| LR stacker (LR_C1.0) | 0.9545 | C=1.0 default |
| LR stacker (LR_C10) | 0.9544 | C=10 overfits |
| LR stacker (LR_L1) | 0.9544 | L1 penalty |
| Single LightGBM_BAG_L1 | 0.9552 | 1 个 base |

**Δ vs AutoGluon: -0.000274**（**变差**）

## 为什么 LR stacker 失败

1. **AutoGluon WeightedEnsemble_L3 已经是 weighted ensemble**——它**已经学过**如何给每个 base model 加权（看 ensemble_weights：LightGBM_BAG_L1 = 1.0）
2. **LR stacker 给了相近权重**（LightGBM_BAG_L1 = 5.656, LightGBMXT_BAG_L1 = 0.799）——基本等价
3. **没有新的 base predictions**——cdeotte 在 S6E6 用的是 7+ 个独立训练（lr_stacker / realmlp / xgb 等）的 OOF，不是 AutoGluon 内部 stack 的

## 与 S6E6 cdeotte 对比

| | S6E6 cdeotte_lr | S6E2 LR stacker (this) |
|---|---|---|
| Source | **公开 kernel**（外部 6 个 base） | AutoGluon **内部** 7 个 base |
| Base predictions | 独立训练，diverse | 同源训练，correlated |
| Best result | OOF 0.9703, LB 0.97101 | OOF 0.9549 |
| Insight | 单源 ≥ blend | Stacker ≤ ensemble |

**关键差异**：cdeotte 的 base predictions 来自多个独立 kernel / 不同模型族。AutoGluon 内部 stack 的 base 都是用同一个训练 pipeline，diversity 不足。

## 启示

| 原则 | 含义 |
|---|---|
| **LR stacker 不会从 correlated base 受益** | AutoGluon 内部 ensemble 已经最优化 |
| **真正的 stacker 需要 diverse base** | 不同训练方法 / 不同数据子集 / 不同源 |
| **S6E2 的天花板** | 14 个特征 + 630K 行，模型已经接近 Bayes-optimal |
| **复制别人 stacker 要 copy 别人的 base** | 不只是 stacker 形式，是 base predictions |

## 关联

- [S6E6 - cdeotte_lr_stacker 新最佳](../lessons/s6e6-cdeotte-lr-best.md) — 对照案例（成功）
- [S6E2 - Heart Disease 完整记录](../competitions/s6e2.md)
- [16 条 Layer 3 原则](../16-principles.md) → signal_dilution, quality_over_quantity

---

#lesson #s6e2 #lr-stacker #signal-saturation #correlated-base