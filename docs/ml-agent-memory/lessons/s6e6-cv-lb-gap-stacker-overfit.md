---
type: Lesson
title: S6E6 - 自训练 stacker 严重 OOF 过拟合 (OOF 0.9724 → LB 0.9626, gap 0.0098) (2026-06-21)
description: 自训练 8-source LR stacker 在 OOF 上看起来比 cdeotte_lr_stacker 好 (0.9724 vs 0.9703)，但 LB 实际差 0.0084 (0.9626 vs 0.97101)。教训: OOF 高 ≠ LB 高，OOF features 上的 stacker 会过拟合。
tags: [lesson, s6e6, cv-lb-gap, stacker-overfit, oof-leakage]
timestamp: 2026-06-21T00:00:00Z
---

# S6E6 - 自训练 stacker OOF 过拟合

## 关键数字（震惊的反向）

| Stacker | OOF BAC | Public LB | Gap | 解读 |
|---|---|---|---|---|
| **自训 8-source LR** | **0.9724** | **0.96264** | **+0.0098** ⚠️ | OOF 最高，LB 最低 |
| 7-source LR (no cdeotte_realmlp) | 0.9725 | 0.96273 | +0.0098 | exp_020 同样问题 |
| 9-source LGB stacker | 0.9632 | 0.96250 | +0.0007 | 偶然一致 |
| **cdeotte_lr_stacker** | **0.9703** | **0.97101** | **-0.0007** | OOF 略低，LB 最高 |

**Δ LB**: 自训 8-LR 比 cdeotte_lr_stacker **差 0.0084**！

## 为什么会 OOF 过拟合

1. **OOF 是 leak-prone features** — 同一折的 OOF 预测包含了该折的 train 信息
2. **8 个 base 都用同一种 5-fold split** — 它们的 OOF 在每个样本上都有相同的 fold 来源
3. **LR 在这些 OOF 上学到 fold-specific patterns** — 在 OOF 上 AUC 高，但放到 test（无 fold 信息）就崩
4. **cdeotte 的 stacker 用了不同的 fold split**（GPU stacker 自己内部 8-fold），所以他的 OOF 没有这个 leak

## 验证：cdeotte 自己承认 v9 是 8-model

```
exp_019: "8-model LogReg stacker on log(probs), no class_weight. 5-fold CV.
         Sources: lgb_v1, lgb_v3, cdeotte_lr, yekenot_realmlp, pilkwang_gbdt, 
         cdeotte_xgb_v5, cdeotte_realmlp_v5, deeplearnerrr_blender.
         OOF acc 0.97245 (vs prior CD 0.96975, +0.0027)"
```

但 cdeotte v9 (8-model, GPU) 提交到 LB 是 0.97101 — 他自己的实现没有过拟合！我们用同样的 8 source 重写就过拟合。

**差异点**: cdeotte 用 GPU + 自定义训练 loop + 可能不同的 fold split + log(probs) 处理。我们直接 sklearn LogisticRegression + 5-fold。

## OOF gap 诊断阈值（更新）

| Gap | 解读 |
|---|---|
| **< 0.005** | OOF 是 LB 的可靠代理 |
| **0.005-0.01** | 轻微过拟合，可能 submission 格式问题 |
| **> 0.01** ⚠️ | **严重过拟合** — OOF 不能信，立刻停止 stacker |

## 决策规则（可重复使用）

```
if oof_score > base_best_oof + 0.005:
    suspect: OOF 过拟合
    action: 用 5-seed bagging 或不同 fold split 重训
    fallback: 用单 base（cdeotte_lr_stacker 这种稳定的）
```

## 与其他 lessons 的关系

- **S6E2 - LR stacker 没有增益** ([s6e2-lr-stacker-no-gain.md](s6e2-lr-stacker-no-gain.md)): 同源 base stacker 无信号
- **S6E2 - CV-LB gap 0.07 是格式错误** ([s6e2_cv_lb_gap.md](s6e2_cv_lb_gap.md)): 0.07 是 0/1 vs proba 的格式 bug
- **本文**: 自训练 stacker 的 CV-LB gap 0.0098 是**真实的过拟合 gap**，不是格式问题

## 启示

| 原则 | 含义 |
|---|---|
| **OOF 高 ≠ LB 高** | 在 OOF features 上 stack 容易过拟合 |
| **用 base 预测做 meta features 时要警惕 fold leakage** | 同 fold split 的 OOF = 半 leak |
| **OOF 和 test 用不同的 fold split 才能避免 leak** | 或用 cdeotte 那种 GPU 独立 stacker |
| **CV-LB gap 是 stacker 健康度指标** | gap > 0.01 立刻停手 |

## 关联

- [S6E6 - Stellar Class 完整记录](../competitions/s6e6.md)
- [S6E6 - cdeotte_lr_stacker 新最佳](../lessons/s6e6-cdeotte-lr-best.md) — 对照案例（成功）
- [S6E2 - LR stacker 没有增益](s6e2-lr-stacker-no-gain.md) — 相同 LR stacker 在 S6E2 失败
- [submission-format skill (PR #5)](../skills/submission-format-by-metric.md) — format vs overfit gap 区分

---

#lesson #s6e6 #cv-lb-gap #stacker-overfit #oof-leakage #fold-leakage