---
type: Report
title: OKF Migration Report — ml-agent-memory
description: 从单文件 markdown memory 到 OKF 知识图谱的迁移记录，含 ROI 量化、lesson 提取、与传统方案对比。2026-06-17.
tags: [report, okf, knowledge-graph, migration, ml-agent-memory]
timestamp: 2026-06-17T00:00:00Z
---

# OKF Migration Report — `docs/ml-agent-memory/`

> **TL;DR**: 把 6 个分散的 `.md` 文件改成 OKF 格式后，知识图谱从 0 边（OKF 工具读不懂）跳到 **35 边**、12 个概念、64KB 可视化 HTML。整体改造时间 **< 1 小时**，ROI 显著。

## 1. 起点：分散的 markdown 文件（无 OKF 兼容）

`docs/ml-agent-memory/` 在 2026-06-14 之前是这样的：

```
docs/ml-agent-memory/
├── 16-principles.md              # 16 条 Layer 3 原则
├── dashboard.md                  # 快速查询入口
├── competitions/
│   ├── s6e4.md                   # Irrigation
│   └── s6e5.md                   # F1 Pit Stop
├── experiments/
│   └── experiments.md            # 28 条实验时间线（单文件！）
└── skills/
    └── adversarial-validation.md # 1 个 skill
```

**OKF 视角**: 6 个概念，0 边（因为引用都是行内反引号 `` `file.md` ``，不是 markdown 链接）。

**问题**:
- 概念间关系**机器不可读**（除非 grep）
- 没法做"哪些概念被引用最多"这种图查询
- 没法可视化
- 团队成员看不到你的知识图谱

## 2. 迁移：3 步走

### Step 1 — 加 YAML frontmatter（10 分钟）

每个 `.md` 文件顶部加：

```yaml
---
type: <类型>
title: <显示名>
description: <一行摘要>
tags: [list, of, tags]
timestamp: 2026-06-15T10:05:55Z
---
```

`type` 是 OKF 唯一**必需**字段——它决定 viz 里的节点颜色和路由。OKF SPEC §4.1 列出推荐但不强求其他字段。

**Type 选择**（`ml-agent-memory` 实际使用 8 种）：

| 文件 | type |
|---|---|
| `dashboard.md` | `Dashboard` |
| `16-principles.md` | `Principle Set` |
| `competitions/s6e*.md` | `Competition` |
| `experiments/experiments.md` | `Experiment Index` |
| `skills/*.md` | `Skill` |
| `lessons/*.md` | `Lesson` |
| `*/index.md` | `Index` |

### Step 2 — 加 `index.md` 子目录入口（5 分钟）

每个 subdirectory 加一个 `index.md`（带 `type: Index`），列出该目录下的所有概念。这给 viz 一个 **hub 节点**，让 graph 更可导航。

### Step 3 — 转 cross-link 格式（10-20 分钟）

把 `` `competitions/s6e4.md` `` 这种行内引用转成 `[s6e4](competitions/s6e4.md)` 真正的 markdown 链接。这样 OKF viz 工具会把它们算成 graph edge。

**关键**:
- 用 **file-relative** 链接（不要 `/tables/users.md` 这种绝对路径，PR #45 修复了这点）
- 链接从根目录开始算路径

## 3. 终点：可观测的知识图谱

迁移后：

```
Wrote 12 concept(s), 35 edge(s), 64117 bytes → /tmp/viz.html
```

viz.html 是一个自包含的 force-directed graph：
- 12 节点按 type 上色（原理 = 紫，技能 = 珊瑚，比赛 = 绿）
- 35 条带向箭头（markdown 链接方向）
- 详情面板（点节点看 frontmatter + body）
- 反向链接（"被谁引用"）
- 全文搜索 + 类型过滤

**完全离线**，50KB，可以 `scp` 给同事 / 嵌进 docs site。

## 4. ROI 量化

### 投入时间

| 阶段 | 时间 |
|---|---|
| 安装 + 学习 OKF | 30 分钟（读 SPEC.md） |
| 写 `migrate_to_okf.py` 自动化脚本 | 15 分钟 |
| 实际跑 + 修 cross-link | 20 分钟 |
| 加 S6E2 内容 + 4 lessons | 30 分钟 |
| 写 `okf-visualize-knowledge` skill | 15 分钟 |
| **总计** | **~2 小时** |

### 产出

| | 数量 |
|---|---|
| OKF 概念 | 12 |
| OKF 边 | 35 |
| Type 覆盖 | 8 种 |
| 可视化 HTML | 64 KB（自包含） |
| 新 skill（教别人用） | 1 (`okf-visualize-knowledge`) |
| Lesson concept | 7（从 3 个比赛提取） |
| PR（已合入 + 待审） | 2 (#5 已合，#6 开放) |
| OKF 仓库 issue | 3 (#60/#71/#72) |

### 未来增量成本

加新比赛或 lesson 只需 **5 分钟**：
1. 复制模板
2. 改 `type` / `title` / `description`
3. 加 cross-link 到已有概念（通常 2-3 个）
4. `enrichment-agent visualize` 重渲染

**零 schema 改动**——这是 OKF 的最大优势。

## 5. 与传统方案对比

| 维度 | 单文件 markdown | Obsidian | OKF + visualize |
|---|---|---|---|
| 版本控制 | ✅ git-friendly | ⚠️ .obsidian/ 文件夹污染 git | ✅ git-friendly |
| 离线 HTML | ❌ | ❌ | ✅ 50 KB 自包含 |
| 跨工具兼容 | ⚠️ Markdown 解析器各异 | ❌ Obsidian-only | ✅ OKF 规范、Markdown 兼容 |
| Type-driven 着色 | ❌ | ⚠️ YAML frontmatter 可识别 | ✅ 原生支持 |
| Graph 视图 | ❌ | ✅ 内置 | ✅ `visualize` 命令 |
| 学习曲线 | 0 | 低 | 中（要懂 OKF spec）|
| 互操作 | ⚠️ 各工具语法略不同 | ⚠️ Obsidian-specific links | ✅ 标准 markdown links |

**OKF 在 "vendor-neutral + git-friendly + offline graph" 这个交集上没有对手**。

## 6. 关键决策记录

### 决策 1: 为什么选 OKF 而不是 Obsidian？

- 你已经在用 git 管理 `~/projects/`，Obsidian 会引入 `.obsidian/` 配置文件夹污染 git
- OKF 输出**一个** HTML 文件可以 `scp` / 邮件 / 嵌入 wiki
- OKF 是 google 的标准（虽然不是官方产品）

### 决策 2: 为什么把 lessons 拆成独立 concept？

之前 `experiments.md` 是**单文件汇总**所有 28 条实验。OKF 视角看：
- 1 个 `Experiment Index` concept（hub）
- 不易引用单个 lesson

迁移方向：把每个有价值的 lesson 拆成独立 `.md` 文件（带 frontmatter），这样：
- 可以被 viz 显示成独立节点
- 可以 cross-link 到 skill（验证/细化）或 competition（源）
- `lessons/index.md` 提供分组入口

### 决策 3: Type 命名约定

遵循 SPEC §4.1 推荐的 type values + 上下文补充：

| 域 | type 值 |
|---|---|
| 容器 | `Bundle`, `Index`, `Dashboard` |
| 知识 | `Principle`, `Principle Set`, `Lesson` |
| 行动 | `Skill`, `Playbook` |
| ML 数据 | `Competition`, `Experiment`, `Model`, `Submission` |
| 抽象 | `Reference` |

`type` 值**不注册**到中央目录，所以可以自由扩展。OKF 维护者提了 [issue #60](https://github.com/GoogleCloudPlatform/knowledge-catalog/issues/60) 提议标准化 ML 类型，采纳后可考虑迁移。

## 7. 下一步

| 优先级 | 行动 |
|---|---|
| 🟢 高 | 等 OKF maintainer 回复 issue #60 看是否合并 ML type 提案 |
| 🟢 高 | 把 lessons/ 模式扩展到所有历史比赛（store-sales / jaguar 等） |
| 🟡 中 | 写一篇对比 Obsidian / Logseq 的博客文 |
| 🟡 中 | 给 OKF repo 提 PR 加 ML sample bundle（S6E2 已经准备好） |
| 🟢 低 | 探索 OKF 的 `enrich` 命令（需 Gemini API key，目前无） |

## 8. 引用

- **OKF SPEC**: https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md
- **本仓库 PR**:
  - #5: submission-format-by-metric skill
  - #6: ml-agent-memory OKF migration（this work）
- **OKF 仓库 issues**:
  - #60: ML-canonical type 提案
  - #71: ML experiment trail sample
  - #72: 新建 issue
- **本地 viz.html**: `/tmp/ml-agent-memory_viz.html`（12 nodes / 35 edges / 64 KB）

---

#report #okf #migration #knowledge-graph #ml-agent-memory