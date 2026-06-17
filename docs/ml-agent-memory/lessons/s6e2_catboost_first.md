---
type: Lesson
title: S6E2 - CatBoost 占 ensemble 68.8% (2026-06-14)
description: CatBoost 在 S6E2 ensemble 中占绝对主导权重，单模型 OOF 0.95550 ≈ 全 ensemble 0.95554。
tags: [lesson, catboost-first, validated, s6e2]
timestamp: 2026-06-14T11:50:00Z
---

# S6E2 - CatBoost 占 ensemble 68.8%

## 单模型 OOF AUC 排名

| 模型 | OOF AUC | Ensemble 权重 |
|---|---|---|
| **CatBoost BAG L1** | **0.95550** | **0.688** ⭐ |
| LightGBM BAG L1 | 0.95523 | 0.062 |
| XGBoost BAG L1 | 0.94784 | 0 |

## 启示

- **CatBoost 在 14 特征表格数据上默认第一选择**
- **native categorical handling** 直接消化掉 7 个类别型特征
- **ordered boosting** 防止 target leakage（医疗数据小，重要）
- 单一最佳模型 + 几个补强 ≥ 10 个同族变种

## 关联

- [catboost-first skill (上游)](../skills/catboost-first-tabular.md)
- [S6E2 - Heart Disease (源比赛)](../competitions/s6e2.md)
- cultivating-ml-agent/skills/examples/catboost-first-tabular/SKILL.md

---

#lesson #catboost-first #s6e2