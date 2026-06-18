---
type: Lesson
title: S6E2 - 概率 vs 0/1 提交，PR #5 起源 (2026-06-14)
description: S6E2 rerun 触发 submission-format-by-metric skill 的诞生，并提 PR #5 到 cultivating-ml-agent。
tags: [lesson, submission-format, new-skill, s6e2]
timestamp: 2026-06-15T00:00:00Z
---

# S6E2 - 概率 vs 0/1 提交，PR #5 起源

## 时间线

1. **2026-06-14**: S6E2 AutoGluon rerun 完成（OOF 0.95554）
2. **第一次提交**: 0/1 阈值化（用 `predictor.predict()`）→ **Public LB 0.88403** ❌
3. **怀疑 overfitting** — OOF 那么高，LB 那么低
4. **诊断**: 同一模型换 `predict_proba()` 重新提交 → **Public LB 0.95357** ✅
5. **根因**: AUC 排名指标下，0/1 没有排序信息，等同随机
6. **写 skill + 提 PR**: 2026-06-15，PR #5 到 cultivating-ml-agent
7. **PR #5 已合并**

## 启示

- **OOF 好不代表提交正确** — 这是常见的"silent"错误
- **本地验证不会暴露** — 只有 LB 才会
- **症状像模型问题，但其实是格式问题** — 容易让人浪费时间去修模型

## 防御措施

- 写一个 helper: `def save_submission(test_pred, test_ids, metric, path)`
  - 根据 metric 自动决定 0/1 还是概率
  - 永远默认 AUC 类用概率

## 关联

- [submission-format-by-metric skill (新)](../skills/submission-format-by-metric.md)
- [S6E2 - Heart Disease (源比赛)](../competitions/s6e2.md)
- cultivating-ml-agent PR #5: https://github.com/topprismdata/cultivating-ml-agent/pull/5

---

#lesson #submission-format #new-skill #s6e2