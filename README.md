<p align="center">
  <img src="https://raw.githubusercontent.com/topprismdata/.github/main/assets/brand/topprism-repo-header.png" alt="TopPrism dual-prism visual" width="100%" />
</p>

# Cultivating ML Agent

### 通过项目实践、知识结晶与可复用 Skills 持续进化的 ML Agent

`Purpose: NATIVE AI` · `Maturity: APPLIED INTERNAL` · `Evidence: MULTI-PROJECT EXPERIENCE`

> 属于 **TopPrism Native AI** — 让组织级机器学习能力在跨项目实践中持续复利的工程体系。
>
> 竞赛只是可量化、可快速反馈的学习环境；最终目标是覆盖更广泛的企业机器学习工作。

[English Version](README_EN.md) | 英文版

---

## Why this exists

大多数 ML Agent 在组织维度是无状态的：每个新项目几乎都从与上一个项目相同的通用模型知识开始。

TopPrism 的目标不同：

> **每一个完成的 ML 项目，都应让 Agent 在下一个 ML 项目里更强。**

项目把反复的实践沉淀成可复用的组织能力 —— 通过知识结晶、结构化 Skills、共享 MLOps 框架与显式评估。

## What this project decides / enables

输入 → 决策/能力 → 输出：

```text
真实 ML 项目
     ↓
实验 & 失败记录
     ↓
经过验证的经验
     ↓
知识结晶（Knowledge Crystallization）
     ↓
可复用 Skill / 框架模式
     ↓
在后续项目中自动激活
     ↓
下一个 ML 项目的更好起点
```

关键产出不仅是训练好的模型或排行榜分数，更是一份**机器可读、可复用的 ML 能力清单**。

## Where it fits at TopPrism

TopPrism 用 Native AI 在 Customer Decision Intelligence 之外构建第二条复利闭环：

```text
客户项目产生经验
     ↓
经验成为可复用的机器知识
     ↓
内部 Agent 能力提升
     ↓
下一个客户 / 内部项目以更强的起点开始
```

关联 TopPrism Native AI 项目：

- `agent-nurture-framework` — 项目驱动的 Agent 能力培养方法论
- `skill-tester` — Skill 质量与触发评估闸门
- `notebook-knowledge-distillation` — 外部知识 → 经验证 Skill 的流水线
- `three-layer-wisdom-extraction` — 把项目经验提升为领域知识与跨领域原则

## Evidence

这是 **跨多个任务族、跨多月**的纵向证据，而不是单一排行榜分数：

- **当前 `skills/examples/` 下有 63 份 `SKILL.md`**（机器可数：`git ls-files skills/examples | grep SKILL.md`）—— 覆盖时间序列、表格预测、视觉、医学影像、音频、博弈/游戏 AI、模型压缩、LLM 推理等任务族
- **9 份独立的竞赛/项目文档**位于 `docs/ml-agent-memory/competitions/`（House Prices、S6E2/3/4/5/6/7、Store Sales、Jaguar Re-ID、Rainfall 等）
- **16 份领域教训文档**位于 `docs/ml-agent-memory/lessons/`
- 后续项目复用前期项目抽取的 Skill 和 framework 组件
- 失败路径作为"负面知识"被保留而非删除
- 共享框架与 Agent 指令显著减少每次需要重新发明的工作量

### 重要边界（brief §5）

- **竞赛名次 ≠ 项目成功的定义**。Kaggle 等环境有用是因为反馈快、可量化，但目标能力是 Agent 通过**任何充分仪器化的 ML 项目**都能提升
- 历史观察（如"time-to-strong-result 显著缩短"）是**纵向案例证据**，不是"每个未来 ML 任务都会按相同倍数改善"的普适保证
- `120+ skills` 这一历史数字与当前仓库不一致；当前实际为 **63**（GitHub Tree API 验证）

## Architecture

```text
                    ML Agent
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   Project Context  Skill Library   Shared Framework
        │              │              │
        └──────────────┼──────────────┘
                       ↓
                 Experiment Loop
                       ↓
             Evaluation / Evidence
                       ↓
            Knowledge Crystallization
                       ↓
             Updated Agent Capability
```

关键仓库区域：

```text
AGENTS.md                 Agent 操作指令
framework/                可复用 ML / MLOps 框架组件
skills/                   结晶的可复用能力
ml-agent-code-template/   新 ML 项目的起始结构
templates/                可复用项目制品
docs/                     方法论与累积知识
examples/                 worked examples
tests/                    共享组件检查
```

## Nurture-First Development

项目遵循 **Nurture-First** 原则：

> 不要尝试一次性编码完整 ML 剧本。让真实项目暴露能力缺口、解决问题、验证方案，再把可复用的部分结晶下来。

典型学习循环：

```text
学习 (理论) → 验证 (Notebook) → 应用 (项目) → 提取 (结晶) → 规划 (差距分析)
```

这区分了**推测性知识**和**经过实际使用存活下来的知识**。

## What this project is NOT

- ❌ 不是单一 AutoML 模型
- ❌ 不是仅供 Kaggle 使用的 Agent
- ❌ 不是静态 prompt 集合
- ❌ 不主张"累积的 Skills 消除了项目特定推理的需要"

## Quick Start

把本仓库当作**项目操作系统**使用，而不仅仅是代码库：

1. 读 `AGENTS.md`
2. 从 `ml-agent-code-template/` 启动新 ML 项目
3. 复用 `framework/`，不要重建常见 MLOps 管线
4. 从 Skill 库激活相关 Skill
5. 记录实验与失败
6. 项目结束后，只把**经过验证的**可复用知识结晶下来

## 已覆盖项目（机器可验证）

| # | 项目 | 领域 | 来源文档 |
|---|------|------|----------|
| 1 | Kaggle S6E2 | 表格 | `docs/ml-agent-memory/competitions/s6e2.md` |
| 2 | Kaggle S6E4 | 时空图 | `docs/ml-agent-memory/competitions/s6e4.md` |
| 3 | Kaggle S6E5 | 表格 | `docs/ml-agent-memory/competitions/s6e5.md` |
| 4 | Kaggle S6E6 | 表格 | `docs/ml-agent-memory/competitions/s6e6.md` |
| 5 | Kaggle Store Sales | 时序 | `docs/ml-agent-memory/competitions/store-sales.md` |
| 6 | Jaguar Re-ID | 计算机视觉 | `docs/ml-agent-memory/competitions/jaguar-reid.md` |
| 7 | Bank Dataset | 表格 | `docs/ml-agent-memory/competitions/bank-dataset.md` |
| 8 | Rainfall Dataset | 表格 | `docs/ml-agent-memory/competitions/rainfall-dataset.md` |
| 9 | Playground s6e7 | 表格 | `docs/ml-agent-memory/competitions/index.md` |

实际项目经验（README 历史 changelog 中记录的）还覆盖：House Prices、Spaceship Titanic、WorldQuant Brain Alpha、AIMO3、Vesuvius Challenge、BirdCLEF+ 2026、March Madness 2026、ISEC 2026、PTCG AI Battle、NeuroGolf 2026、ROGII Wellbore、Biohub Cell Tracking、nnU-Net Medical 等。具体方法论与 LB 数字以各自项目文档与对应 Skill 为准。

## 最新里程碑（沿用历史 changelog，保留为证据）

### v0.9.0 (2026-08-05) — Time-Series & Recommendation Era

新增 3 个 skills（基于零售 SKU 推荐项目实战经验）：
- `time-series-walk-forward-validation` — 时间序列项目标准评估规范
- `cross-competition-feature-transfer` — 跨竞赛特征迁移方法论
- `feature-engineering-saturation-detection` — "特征工程已饱和"的 4 个判断信号

### v0.7.0 (2026-06-14) — AutoGluon Era

新增 3 个 skills：`autogluon-first`、`catboost-first-tabular`、`cv-lb-gap-acknowledgment`。

### v0.5.0 (2026-05-31) — ML Agent Code Template

新增 `ml-agent-code-template/`（9 hooks, 6 commands, 2 agents），在 8 个 MLE-Bench 竞赛中验证（6 Gold, 2 Silver）。

## Boundaries & Limitations

- 仓库 About 描述中"120+ skills, 13 projects, 2 months"为早期版本数字；当前事实是 63 skills（机器可数）、4+ months、9 份正式 competition docs
- 任何项目 LB / 排行分数只在**对应项目与对应协议**下成立；不可作为"通用 ML Agent 能力"的直接证据
- Skills 累积不替代项目特定推理；新问题仍需要人工判断哪些 Skill 真正适用
- 失败经验作为负面知识保留，但**并不自动防止**在未来项目里重复同一类错误；交叉 Skill 触发的可靠性仍取决于 skill-tester 的评估

## Related TopPrism Projects

- `agent-nurture-framework` — 项目驱动的 Agent 能力培养方法论（Pin 仓库）
- `skill-tester` — Skill 质量与触发评估闸门
- `notebook-knowledge-distillation` — 外部知识 → 经验证 Skill 流水线
- `three-layer-wisdom-extraction` — 项目经验 → 领域知识 → 跨领域原则

## Repository Structure

```text
cultivating-ml-agent/
├── README.md                    # 本文件（中文旗舰）
├── README_EN.md                 # 英文旗舰
├── AGENTS.md                    # 自主 Agent 指令
├── LICENSE                      # MIT
├── CONTRIBUTING.md
├── docs/                        # 方法论与累积知识
│   ├── cultivating-ml-agent-expert.md
│   ├── framework/
│   └── ml-agent-memory/         # 项目案例、教训、原则
├── framework/                   # 可复用 ML / MLOps 框架
├── skills/examples/             # 63 份 SKILL.md
├── ml-agent-code-template/      # Claude Code 起始模板
├── templates/
├── examples/
└── tests/
```

## License

MIT — 详见 [LICENSE](LICENSE)。

## Citation / Contributing

详见 [CONTRIBUTING.md](CONTRIBUTING.md)。
