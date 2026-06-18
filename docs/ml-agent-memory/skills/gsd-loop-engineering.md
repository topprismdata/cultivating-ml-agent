---
type: Skill
title: gsd-loop-engineering
description: GSD Core 5-phase loop adapted to ML：DISCUSS → PLAN → EXECUTE → VERIFY → SHIP。在 S6E2 上验证：同模型两次提交 LB 差 0.07，因为 verifier 抓到 0/1 vs proba 错误。
tags: [skill, gsd, loop-engineering, fresh-context, verification, okf-demo]
timestamp: 2026-06-18T00:00:00Z
---

# gsd-loop-engineering

> **状态**: 仓库已写入（main @ 55d96a0）| **S6E2 验证**: 0/1 vs proba 提交 0.07 LB 差

## 5-Phase Loop

```
DISCUSS → PLAN → EXECUTE → VERIFY → SHIP
              ↑                │
              └──── FIX ───────┘
                 (BLOCKER → re-PLAN)
```

每阶段都有：
- **Fresh-context subagent**（避免 context rot）
- **Spec-driven artefacts** on disk（survive session boundary）
- **Wave-based DAG** execution（独立 plan 并行）

## 5 个核心 pattern

1. **Fresh-context subagents** — 主 session 保持薄，heavy 工作扔给子 agent
2. **Goal-backward verification** — 默认假设目标**未达成**，直到 codebase 证据证明
3. **Wave DAG execution** — 计划声明 depends_on，独立 plan 并行
4. **Escalation Gate** — BLOCKER 必须抛给开发者，不静默猜测
5. **Spec-driven artefacts** — 每阶段产出文件，session 边界后还能接力

## 应用到 ML 的对应

| GSD Phase | ML 对应 |
|---|---|
| DISCUSS | metric / submission format / deadline / data quirks |
| PLAN | features / baseline / ensemble / verify plans |
| EXECUTE | AutoGluon 训练 + 数据下载并行 |
| VERIFY | format check / CV-LB gap / adversarial validation |
| SHIP | kaggle submit + archive |

## 安装 + 使用

```bash
# 安装 (一次性)
~/projects/setup_gsd.sh

# 在任何 ML 项目里用
cd ~/projects/kaggle-ps-s6e4
claude
# /gsd-new-project
# /gsd-discuss-phase  ← 讨论 metric / submission format
# /gsd-plan-phase     ← 计划 baseline + verify
# /gsd-execute-phase  ← 跑训练
# /gsd-verify         ← 检查 submission 格式、C AUC gap ← 这里抓 0/1 bug
# /gsd-ship           ← submit 到 LB
```

## 关联

- [S6E2 - 概率 vs 0/1 提交 lesson](../lessons/s6e2_submission_format.md) — VERIFY phase 抓到的真实 bug
- [kaggle-submission-format-by-metric](../skills/submission-format-by-metric.md) — VERIFY step 1
- [autogluon-first](../skills/autogluon-first.md) — EXECUTE phase standard recipe
- [cv-lb-gap-acknowledgment](../skills/cv-lb-gap-acknowledgment.md) — VERIFY step 4
- [ml-sweet-spot](../skills/ml-sweet-spot.md) — 何时 STOP（loop 终止条件）

## 来源

- **GSD Core (canonical)**: https://github.com/open-gsd/gsd-core
- **原仓库 (deprecated)**: https://github.com/gsd-build/get-shit-done (64K stars)
- **Skill source**: `cultivating-ml-agent/skills/examples/gsd-loop-engineering/SKILL.md`

---

#skill #gsd #loop-engineering #verification #okf-demo