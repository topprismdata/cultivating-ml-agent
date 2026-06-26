# Cultivating ML Agent Expert — 培养 ML Agent 专家

> 通过知识结晶（Knowledge Crystallization）系统化训练 AI Agent，从 ML 新手到竞赛 Top 10% 的完整指南。

> A systematic guide for training AI agents from ML novice to competition Top 10% through knowledge crystallization.

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Skills](https://img.shields.io/badge/skills-28%2B-blue)](https://github.com/topprismdata/cultivating-ml-agent/tree/main/skills/examples)
[![Competitions](https://img.shields.io/badge/competitions-15%2B-success)](https://github.com/topprismdata/cultivating-ml-agent#covered-projects-15)
[![Version](https://img.shields.io/badge/version-0.7.0-orange)](https://github.com/topprismdata/cultivating-ml-agent/releases)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/topprismdata/cultivating-ml-agent/pulls)
[![Stars](https://img.shields.io/github/stars/topprismdata/cultivating-ml-agent?style=social)](https://github.com/topprismdata/cultivating-ml-agent/stargazers)

</div>

[English Version](README_EN.md) | 英文版

---

基于 **3+ 个月** 使用 Claude Code 的真实实验，覆盖 **15+ 个 ML 竞赛/项目**，涵盖 Re-ID、时序预测、表格数据、定量 Alpha、医学影像、音频分类、数学推理等领域。

**核心成果**：Agent 从首个竞赛需要 **2 周**达到 Top 10%，进化到 **Top 5%** — 整整 **14x 加速**，全部归功于累积和结晶的知识。

---

## 🆕 v0.7.0 新版本 (2026-06-14) - AutoGluon Era

### 🏆 最新成就

| 竞赛 | 最佳成绩 | 方法 |
|------|----------|------|
| **House Prices Advanced Regression** | **0.11750 LB** | 用户混合 V16+V17 (30/70) — 超越 cs229_v9 0.11765 |
| **Spaceship Titanic** | **0.80780 LB** | SST_v2 Top-5 CatBoost — 超越 V12 0.8066 |
| **AutoGluon 历史重跑** | 4/12 胜 | AutoGluon 2/4 战胜手动集成 |

### 📚 新 Skills (28+ 总数, v0.7.0 新增 3 个)

| Skill | 用途 | 验证数据 |
|-------|------|----------|
| **autogluon-first** | 任何表格比赛第一步跑 AutoGluon `best_quality` (5-15 min baseline) | House Prices CV 0.1180 vs V18 0.1194 |
| **catboost-first-tabular** | 手动 GBDT 时首选 CatBoost（5 变体 ensemble sweet spot）| SST OOF 0.8124 vs XGB 0.8003 |
| **cv-lb-gap-acknowledgment** | CV 改善 ≠ LB 改善。必须 LB 验证 | SST 0.005-0.01 gap |

### ✏️ 增强 Skills (v0.7.0 增强 2 个)
- **ml-sweet-spot** — 新增 CatBoost-First 证据 + AutoGluon-First 对比
- **kaggle-optimal-blending** — 新增 asymmetric-blending (30/70) 原则

---

## 🚀 快速开始

### 👤 人类用户

1. 阅读 [主指南](docs/cultivating-ml-agent-expert.md) (1088 行, ~30 min)
2. 浏览 [示例 skills](skills/examples/) — **28+ skills** 覆盖表格、NLP、视觉、时序
3. 使用 [模板](templates/) 创建自己的 skills
4. **🆕 v0.7.0**: 表格问题先看 `skills/examples/autogluon-first/`

### 🤖 AI Agents (Claude Code 等)

1. 阅读 [AGENTS.md](AGENTS.md) 获取自主 ML workflow 指令
2. 使用 `framework/` 模块做结构化 pipeline (config, logging, validation, MLflow)
3. 遇到匹配问题时从 `skills/examples/` 激活 skills
4. **🆕 v0.7.0**: 试用 [`ml-agent-code-template/`](ml-agent-code-template/) 现成的 Claude Code 配置 (auto-activation, cross-model review, memory health checks)

---

## 💡 核心概念

### 1. 培养优先开发 (Nurture-First Development)

不要预先编程所有知识。构建 **知识结晶循环**：
> 在实践中遇到问题 → 提取可复用模式 → 组织成结构化 skills → 遇到类似问题自动激活

### 2. 三层知识架构

| 层级 | 内容 | 更新频率 |
|------|------|----------|
| L1: 核心能力 | ML 基础、数据科学工作流 | 每月 |
| L2: 领域 Skills | 竞赛特定技术、反模式 | 每周 |
| L3: 智慧原则 | 跨领域通用原理 | 每里程碑 |

### 3. 五阶段学习循环

```
学习 (理论) → 验证 (Notebook) → 应用 (竞赛) → 提取 (结晶) → 规划 (差距分析)
```

### 4. 🆕 v0.7.0: AutoML 优先策略

**任何表格问题，AutoGluon 是最佳第一步**（5-15 min）。只在以下情况转手动 GBDT：
- AutoGluon OOF 不达标
- 有 AutoGluon 抓不到的领域知识
- 想用 AutoGluon 作为 "Silver" 信号加入自定义 pipeline

验证：AutoGluon 在小/中等表格数据上 **2/4 战胜** 手动集成。

---

## 📁 项目结构

```
cultivating-ml-agent/
├── README.md                    # 本文件 (中文)
├── README_EN.md                 # 英文版
├── AGENTS.md                    # 自主 Agent 指令
├── docs/
│   ├── cultivating-ml-agent-expert.md   # 主指南 (1088 行)
│   └── framework/                       # 框架文档
├── framework/                   # 可复用 MLOps 框架
├── skills/
│   └── examples/                # 28+ 个真实 skills
│       ├── autogluon-first/            # 🆕 v0.7.0
│       ├── catboost-first-tabular/     # 🆕 v0.7.0
│       ├── cv-lb-gap-acknowledgment/   # 🆕 v0.7.0
│       ├── claudeception/              # 自动 skill 提取
│       ├── three-layer-wisdom-extraction/
│       ├── agent-nurture-framework/
│       ├── ml-sweet-spot/              # ✏️ v0.7.0 更新
│       ├── kaggle-optimal-blending/     # ✏️ v0.7.0 更新
│       └── ... (28+ 总数)
└── templates/
    ├── bug-fix-skill.md
    └── knowledge-skill.md
```

---

## 🏆 已覆盖项目 (15+)

| # | 项目 | 领域 | 关键成就 |
|---|------|------|----------|
| 1 | Kaggle S6E2 | 表格 | 首个竞赛, Top 9% |
| 2 | Kaggle S6E3 | 表格 | 对抗验证突破 |
| 3 | Kaggle S6E4 | 时空图 | 24h 到 Top 10% |
| 4 | WorldQuant Brain Alpha | 量化 | Alpha 因子挖掘 |
| 5 | Jaguar Re-ID | 计算机视觉 | 94.46% 验证准确率 |
| 6 | AIMO3 | 数学推理 | SC-TIR + Qwen3.5 |
| 7 | Store Sales | 时序 | LB 1.859 → 0.399 (4.7x 改进) |
| 8 | Vesuvius Challenge | 3D 分割 | nnU-Net + RAG 研究 |
| 9 | BirdCLEF+ 2026 | 音频分类 | 234 野生物种 |
| 10 | March Madness 2026 | 体育预测 | Elo/Massey 评分系统 |
| 11 | ISEC 2026 | 软件缺陷 | SMOTE + 多项式特征 |
| 12 | Store Sales R11 | 时序 | Top 5% (最新) |
| 13 | nnU-Net Medical | 医学影像 | Apple Silicon 训练 |
| 14 | **House Prices Advanced Regression** | **表格** | **🆕 v0.7.0: LB 0.11750** |
| 15 | **Spaceship Titanic** | **表格** | **🆕 v0.7.0: LB 0.80780** |
| 16 | **PTCG AI Battle** | **博弈 RL** | **🆕 v0.8.0: V_net + 2-ply search = 59% vs rank-304 heuristic (Deep RL Phase 2)** |

---

## 🛠️ 关键方法论 (SOPs)

主指南中最重要的 **5 个 SOPs**：

1. **竞赛启动 SOP** — 从数据下载到首次提交的系统化工作流
2. **模型调试 SOP** — 从预测幅度到特征重要性的渐进诊断
3. **Skill 提取 SOP** — 通过 claudeception 自动知识结晶
4. **实验管理 SOP** — 命名约定的可重复迭代
5. **集成学习 SOP** — 从相关性检查到最优混合

### 🆕 v0.7.0 新方法论洞察

| 洞察 | 为什么重要 |
|------|------------|
| **AutoGluon 是表格第一步** | 5-15 min baseline 匹配数天手动工作 |
| **CV ≠ LB** | CV 改善不转化为 LB（常见 0.005-0.01 gap）|
| **CatBoost > LightGBM/XGBoost** 表格 | 原生类别处理，健壮默认值 |
| **多模型多样性 > 多 seed** | 3 个 GBDT 家族 > 15 个同家族模型 |
| **非对称混合** | 30% Silver + 70% Top-5 > 50/50（当一个家族主导时）|

---

## 🔧 MLOps 框架

`framework/` 目录提供可复用 Python 模块，已在真实 Kaggle 竞赛中验证 (H&M Recommendations LB 0.02368, S6E4 LB 0.98150)。

### 快速集成

```bash
# 复制框架到你的竞赛项目
cp -r framework/ /path/to/your-competition/

# 为你的竞赛编辑配置
cp framework/config_template.yaml config.yaml
```

### 🆕 v0.7.0: 表格比赛推荐工作流

```
Step 1: AutoGluon (5-15 min)      [新 SKILL: autogluon-first]
   ↓ 验证 OOF
Step 2: CatBoost 单模型           [新 SKILL: catboost-first-tabular]
   ↓ 对比
Step 3: 5 个 CatBoost 变体集成 (sweet spot)
   ↓ 加入 LightGBM + XGBoost (多模型多样性)
Step 4: LB 验证                   [新 SKILL: cv-lb-gap-acknowledgment]
   ↓ 不改进就停止
Step 5: AutoGluon 作为 Silver + 自定义集成  [新 SKILL: kaggle-optimal-blending]
   ↓ 提交
```

---

## 🎓 学术对齐

| 学术概念 | 我们的实践 |
|----------|------------|
| AIDE (Huang 2024) trial-and-error | 详细失败记录 |
| AutoMind (Zhang 2025) 知识库 | 120+ SKILL.md 三层架构 |
| Voyager (Wang 2023) skill library | Claudeception 自动提取 |
| CoMind (2025) 记忆架构 | 全局/项目/skill 三层记忆 |
| Reflexion (Shinn 2023) 经验反思 | 三层智慧提取 |
| NFD (Zhang 2026) 培养优先 | 核心哲学 |
| **AutoGluon (Fakoor 2020)** | **🆕 v0.7.0: 多算法集成 + stacking baseline** |
| **TabPFN (Hollmann 2023)** | **🆕 v0.7.0: 小表格 Transformer（未来方向）** |

---

## 📜 变更日志

### v0.7.0 (2026-06-14) — AutoGluon Era

**新增 3 个 skills:**
- `autogluon-first` — 表格比赛第一步跑 AutoGluon `best_quality`
- `catboost-first-tabular` — CatBoost > LightGBM/XGBoost 表格
- `cv-lb-gap-acknowledgment` — CV 改善 ≠ LB 改善

**增强 2 个 skills:**
- `ml-sweet-spot` — 新增 CatBoost-First 证据
- `kaggle-optimal-blending` — 新增 asymmetric-blending (30/70)

**新覆盖竞赛:**
- House Prices Advanced Regression (LB 0.11750)
- Spaceship Titanic (LB 0.80780)

**验证:**
- AutoGluon 在小/中等表格上 2/4 战胜手动集成
- Top-5 CatBoost > 15 模型混合集成（当一个家族主导时）
- 0.005-0.01 CV-LB gap 持续观察到

### v0.6.0 (2026-06-02) — Proactive Evolution

- 新增 3 个 Proactive Evolution 增强
- 更新培养框架
- 新增 retail-eda-framework skill

### v0.5.0 (2026-05-31) — ML Agent Code Template

- 新增 `ml-agent-code-template/` (9 hooks, 6 commands, 2 agents)
- 在 8 个 MLE-Bench 竞赛中验证 (6 Gold, 2 Silver)
- 新增 Obsidian Memory Vault 模式

### 更早版本 (v0.1.0 - v0.4.0)

- 13 个竞赛经验结晶
- 19 → 28+ skills
- 建立三层知识架构

---

## 📄 许可证

MIT License — 自由使用此框架培养您自己的 ML Agent。

## 🤝 贡献

欢迎贡献！特别是：
- **来自您自己 ML 项目的新 skill 示例**
- **改进的 SOP 或方法论**
- **主指南的翻译**
- **Skill 模板的 Bug 修复**
- **新 AutoML 工具集成** (H2O, FLAML, Auto-sklearn)

### 添加新 Skill

1. 复制 `templates/knowledge-skill.md` 到 `skills/examples/<your-skill-name>/SKILL.md`
2. 填写模板 (problem, context, solution, anti-patterns)
3. 至少在一个真实竞赛中验证
4. 更新本 README 索引
5. 提交 PR

---

<div align="center">

**最后更新**: 2026-06-14 | **版本**: 0.7.0 | **总 Skills**: 28+ | **总竞赛**: 15+

Made with ❤️ for the ML community | 用 ❤️ 制作，献给 ML 社区

</div>
