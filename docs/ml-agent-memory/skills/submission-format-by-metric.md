---
type: Skill
title: submission-format-by-metric
description: 关键 skill：AUC 等排名指标比赛必须提交概率文件，不是 0/1 阈值化。S6E2 rerun 验证：0/1 提交让 LB 从 0.95357 跌到 0.88403 (0.07 drop).
tags: [skill, submission-format, AUC, okf-demo, new-skill]
timestamp: 2026-06-15T00:00:00Z
---

# submission-format-by-metric

> **状态**: PR #5 已合入 cultivating-ml-agent | **S6E2 验证**

## 核心问题

**AUC / log_loss / MAP / NDCG 等排名指标必须提交概率文件**，不是 0/1 阈值化。

**S6E2 实证**：同一个 AutoGluon 模型（OOF AUC 0.95554）：
- `submission_autogluon.csv` (0/1 阈值化) → Public LB **0.88403** ❌
- `submission_autogluon_proba.csv` (概率) → Public LB **0.95357** ✅
- 0.07 LB drop from format alone.

## 决策树

```
metric 是 ranking 类 (AUC, log_loss, MAP, NDCG, ...) ?
├── YES → 用 predict_proba() / 概率列
└── NO (accuracy, f1, kappa) → 用 predict() / 类标签
```

## 框架 cheatsheet

| 框架 | 正确写法 |
|---|---|
| AutoGluon | `predictor.predict_proba(test)` |
| sklearn | `model.predict_proba(test)[:, 1]` |
| xgboost (sklearn API) | `model.predict_proba(test)[:, 1]` |
| lightgbm | `model.predict_proba(test)[:, 1]` |
| catboost | `model.predict_proba(test)[:, 1]` |

## 为什么是个 bug

1. OOF AUC 看不出来（fold 内预测连续值）
2. 本地验证通过（OOF 看着很美）
3. 只有 LB 才暴露（因为 LB 把 0/1 当 ties）
4. 错误症状像 overfitting，容易让人去"修模型"而不是"修格式"

## 关联

- [S6E2 - Heart Disease (发现此 bug 的比赛)](../competitions/s6e2.md)
- [cv-lb-gap-acknowledgment skill](cv-lb-gap-acknowledgment.md) — 区分格式错误 vs 真实 CV-LB gap
- [autogluon-first skill](autogluon-first.md) — 默认用 `predict_proba()` 提交

## 详见

- cultivating-ml-agent PR #5: https://github.com/topprismdata/cultivating-ml-agent/pull/5
- Skill source: `~/projects/cultivating-ml-agent/skills/examples/kaggle-submission-format-by-metric/SKILL.md`

---

#skill #submission-format #AUC #okf-demo