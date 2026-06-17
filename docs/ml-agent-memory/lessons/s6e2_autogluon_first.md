---
type: Lesson
title: S6E2 - AutoGluon best_quality 验证 (2026-06-14)
description: AutoGluon 15 分钟跑出 OOF 0.95554 / Private 0.95510，与用户多日手动 stacking (0.95516) 统计上不可区分。
tags: [lesson, autogluon-first, validated, s6e2]
timestamp: 2026-06-14T11:50:00Z
---

# S6E2 - AutoGluon best_quality 验证

## 关键数字

| | 值 |
|---|---|
| AutoGluon OOF AUC | 0.95554 |
| AutoGluon Private LB | 0.95510 |
| 用户历史 Private LB | 0.95516 |
| Δ | -0.00006 (统计上不可区分) |
| AutoGluon 训练时间 | 15.48 分钟 |
| 用户历史工作时间 | 多日（18 次提交）|

## 启示

- **手工 stacking 对 S6E2 类表格不再有显著优势**
- **建议未来工作**: 先跑 AutoGluon baseline，看是否值得人工迭代；如果 OOF 已经很高，直接 blend 即可
- **多日 vs 15 分钟** — ROI 倒挂

## 关联

- [autogluon-first skill (上游)](../skills/autogluon-first.md)
- [S6E2 - Heart Disease (源比赛)](../competitions/s6e2.md)
- cultivating-ml-agent/skills/examples/autogluon-first/SKILL.md

---

#lesson #autogluon-first #s6e2