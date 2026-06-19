---
type: Lesson
title: S6E6 - cdeotte_lr_stacker 单模型 0.97101 新最佳 (2026-06-18)
description: 引入 cdeotte 公开 GPU Logistic Regression Stacker 后，Public LB 0.97101 超过用户历史最佳 0.97077。再次验证"单源 > blend"。
tags: [lesson, s6e6, cdeotte, external-data-fusion, single-best]
timestamp: 2026-06-18T00:00:00Z
---

# S6E6 - cdeotte_lr_stacker 单模型新最佳

## 关键数字

| Submission | OOF BAC | Public LB |
|---|---|---|
| **cdeotte_lr_stacker single** | **0.9703** | **0.97101** ⭐ |
| 5-way weighted (cdeotte+own) | 0.9698 | 0.97021 |
| 用户历史最佳 (majority ensemble) | n/a | 0.97077 |
| 3-way blend (xgb+cb+schema8) | 0.9637 | 0.96379 |
| schema8 single (own) | 0.9668 | 0.96719 |
| AutoGluon best_quality (own) | 0.9630 | 0.95021 |

**Δ vs 用户历史最佳**: **+0.00024** (新 Public LB 最佳！)

## 关键洞察

1. **外部 OOF 比自训练更有价值**: cdeotte_lr_stacker 公开 OOF 0.9703，比我们所有自训练 OOF (最高 schema8 0.9668) 都高
2. **单源 > blend**: 单模型 cdeotte_lr 0.97101 > 任何 blend (最高 0.97021)
3. **CV 几乎完美对齐**: OOF 0.9703 → LB 0.97101，gap 0.0007（正常）
4. **AutoGluon + GSD 完整跑通**: 11 分钟 baseline → 30 分钟加外部 OOF → +0.0208 LB 改善

## 启示

- **公开 kernel 永远先看**: external_sources/INDEX.md 给了 7 个有 OOF 的 kernel，cdeotte_lr 最强
- **LR stacker 强于 GBDT/MLP**: 这是反直觉的——一个 logistic regression over base predictions 比 XGBoost 还好
- **"单源 vs blend"反复验证**: S6E6 上至少 3 个不同 blend 都比最佳单源差——信号已经被最强单源吃掉

## 关联

- [S6E6 - 完整竞赛记录](../competitions/s6e6.md) (待更新)
- [S6E6 schema8 单模型 > blend](s6e6-schema8-single-best.md) — 同一方向的 lesson
- [16 条 Layer 3 原则](../16-principles.md) → work_smart_not_hard, quality_over_quantity, external_data_fusion_largest_single_step_improvement

---

#lesson #s6e6 #cdeotte #external-data #single-best