---
type: Lesson
title: S6E2 - CV-LB gap 0.07 是格式错误不是过拟合 (2026-06-14)
description: 实测 CV-LB gap 从 0.002 (概率文件) 到 0.07 (0/1 文件)，整整 35 倍差距。CV-LB gap 警告阈值需要分情况。
tags: [lesson, cv-lb-gap, refined, s6e2]
timestamp: 2026-06-14T11:51:00Z
---

# S6E2 - CV-LB gap 0.07 是格式错误不是过拟合

## 实测对比

| 提交 | 格式 | Public LB | CV-LB gap |
|---|---|---|---|
| submission_autogluon.csv | 0/1 阈值化 | 0.88403 | **0.07** ❌ |
| submission_autogluon_proba.csv | 概率 | 0.95357 | **0.002** ✅ |

**同一个模型，差距 35 倍**。

## 启示

`cv-lb-gap-acknowledgment` skill 原文说 "0.005-0.01 是典型 gap"。**这个数字只在概率提交下成立**。0/1 提交会让 gap 膨胀到 0.05-0.10+。

## 决策树（更新）

```
LB 比 OOF 低很多 ?
│
├─ 先检查提交格式（不是 0/1 ?）
│   └─ YES → 重新提交概率版，不要去修模型
│
├─ 格式对，gap < 0.01 → 正常，可以接受
├─ 格式对，0.01 < gap < 0.05 → 警戒
└─ 格式对，gap > 0.05 → 真的过拟合，回退或 pivot
```

## 关联

- [cv-lb-gap-acknowledgment skill (上游)](../skills/cv-lb-gap-acknowledgment.md)
- [submission-format-by-metric skill (新)](../skills/submission-format-by-metric.md)
- [S6E2 - Heart Disease (源比赛)](../competitions/s6e2.md)

---

#lesson #cv-lb-gap #s6e2