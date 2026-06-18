---
type: Lesson
title: OKF enrich 不适合 Kaggle 数据 (2026-06-18)
description: OKF 的 `enrich` 命令是 BigQuery 数据目录工具，不适用于 Kaggle 比赛数据。手写 OKF bundle 更合适。
tags: [lesson, okf, enrich, bigquery, kaggle]
timestamp: 2026-06-18T00:00:00Z
---

# OKF enrich 不适合 Kaggle 数据

## 关键发现

`enrichment-agent enrich --source bq` 是为 **production BigQuery 数据目录** 设计的，不是为 Kaggle 比赛。

**原因**:
1. Kaggle Playground 系列（s6e*/s5e* 等）**没有公开 BigQuery 镜像**
2. 比赛数据是 CSV，schema 简单到不值得 LLM 介入
3. ML experiment trail（你的模型 + lesson + skill）需要**手写** frontmatter 控制
4. BigQuery metadata → OKF concept 的转换**信息增益小**（CSV header 已经够了）

## OKF enrich 的正确使用场景

| 场景 | enrich 适用？ |
|---|---|
| 文档化生产 BigQuery 表 | ✅ YES |
| 文档化 Kaggle 比赛数据 | ❌ NO（手写）|
| ML experiment trail | ❌ NO（手写）|
| 公共数据集元数据丰富 | ⚠️ 需要 web pass |

## 实操：手写 vs enrich

**手写一个 S6E2 OKF bundle**：
- 时间: ~30 分钟（12 concepts, 23 edges）
- 收益: 完全控制 type/title/description/tags/cross-links
- 例子: `~/projects/s6e2-okf-bundle/`

**跑 enrich on 真实生产数据**：
- 时间: ~10-30 分钟（取决于 table 数）
- 收益: 自动生成 frontmatter，但 description 是 LLM 写的，可能需要人工 review
- 需要: Gemini API key + GCP 凭证

## 推荐混合工作流

```
1. 跑 AutoGluon 拿到模型 + OOF + 提交 (15 min)
2. 手写 OKF bundle (30 min, ~12 concepts)
3. 用 visualize 渲染图谱
4. [可选] 如果你有相关生产数据表 → enrich 单独跑
```

## 关联

- [OKF enrich runbook](../../../../knowledge-catalog/ENRICH_RUNBOOK.md) — 完整命令参考
- [okf-visualize-knowledge skill](../skills/okf-visualize-knowledge.md)
- [S6E2 - Heart Disease (手写 bundle 案例)](../competitions/s6e2.md)

---

#lesson #okf #enrich #bigquery #kaggle