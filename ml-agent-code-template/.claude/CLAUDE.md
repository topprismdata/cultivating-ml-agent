# Project CLAUDE.md — Template

## 修改说明

此文件是模板。复制到你的项目根目录后，按以下顺序定制：

1. **数据路径** — 更新 `~/Library/Caches/mle-bench/data/` 等路径
2. **竞赛列表** — 保留你要参加的竞赛，删除其他的
3. **关键参数** — 填写你的模型参数范围、阈值
4. **已知死路** — 从反馈文档中复制相关条目

---

# CLAUDE.md — Global Instructions

## 项目概述

```json
{"project": "MLE-Bench Kaggle", "goal": "6+ Gold medals", "competitions": 12}
```

## 数据路径

```
~/Library/Caches/mle-bench/data/<competition>/prepared/public/
~/Library/Caches/mle-bench/data/<competition>/prepared/private/
~/mle-bench/submissions/<competition>/submission.csv
```

## 核心原则

### 1. Always Start Simple
先建立基线，再增量加特征/模型/集成。每次只改一个变量。

### 2. One Hypothesis Per Experiment
每次实验只有一个假设变更，清晰记录为什么成功或失败。

### 3. CV Before LB
先跑 cross-validation 验证，再提交。不用 daily quota 去赌未验证的方向。

### 4. Extract + Persist Knowledge
突破后立即记录到 memory/。禁止突破后不记录。

### 5. Consult Memory Before Acting
开始新竞赛/新方向前，先用 `claude-mem mem-search` 检索历史经验。

## MLE-Bench 竞赛最佳实践

| 模式 | 核心 | 证据 |
|------|------|------|
| 特征工程 > 元学习器优化 | 当堆叠收敛时，只有新特征能突破 | TPS May 2022 f_27 |
| 外部数据融合 ROI ~7x | 最大单一改进超过调参 | Jigsaw Toxic XLM-R |
| 模型多样性 > 模型数量 | 相关性 < 0.95 才有效益 | 跨meta-learner无增益 |
| Priv AUC 是真实指标 | Public LB 可能误导 | TPS May 2022 |
| Adversarial Validation 门槛 | AUC ≈ 0.50 → 停止净化 | train/test 对齐 |

## 堆叠上限公式

当所有基模型 Priv AUC > 0.99 AND 平均相关性 > 0.93：
- 堆叠上限 ≈ 当前最佳 + 0.001 AUC
- **只有添加相关性 < 0.85 的新基模型才能突破**

## 反模式（禁止重复）

| 反模式 | 教训 |
|--------|------|
| bundle_causes_uninterpretable | 同时改多变量无法归因 |
| local_optimum_trap | 3次 <0.0001 改进必须 pivot |
| CatBoost as primary on large tabular | fold AUC 比 LGB 低 2-3% |
| multi-seed meta-learner averaging | 无互补性，无改善 |
| Nelder-Mead on private test | 过拟合 |
| cross-meta-learner blending | 相关性极高，无增益 |

## 实验追踪 SOP

```
1. 创建 memory/experiments/exp_<YYYYMMDD>_<name>.md
2. 记录：假设 → 操作 → 结果 → 结论
3. 突破 → 立即更新 memory/competitions/<name>.md
4. 失败 → 记录到 feedback_no_recheck_confirmed_dead.md
```

## Memory 路径

```
~/.claude/projects/-Users-mac/memory/
memory/MEMORY.md — 项目内索引
memory/experiments/ — 实验记录
memory/competitions/ — 竞赛日志
memory/principles/ — 跨领域原则
memory/skills/ — 可提取的技能
```

## 已知死路

参见 `memory/feedback_no_recheck_confirmed_dead.md`（开始前必查）