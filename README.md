# Cultivating ML Agent Expert — 培养 ML Agent 专家

> 通过知识结晶（Knowledge Crystallization）系统化训练 AI Agent，从 ML 新手到竞赛 Top 10% 的完整指南。

> A systematic guide for training AI agents from ML novice to competition Top 10% through knowledge crystallization.

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Skills](https://img.shields.io/badge/skills-43%2B-blue)](https://github.com/topprismdata/cultivating-ml-agent/tree/main/skills/examples)
[![Competitions](https://img.shields.io/badge/competitions-20%2B-success)](https://github.com/topprismdata/cultivating-ml-agent#covered-projects-15)
[![Version](https://img.shields.io/badge/version-0.9.0-orange)](https://github.com/topprismdata/cultivating-ml-agent/releases)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/topprismdata/cultivating-ml-agent/pulls)
[![Stars](https://img.shields.io/github/stars/topprismdata/cultivating-ml-agent?style=social)](https://github.com/topprismdata/cultivating-ml-agent/stargazers)

</div>

[English Version](README_EN.md) | 英文版

---

基于 **4+ 个月** 使用 Claude Code 的真实实验，覆盖 **20+ 个 ML 竞赛/项目**，涵盖 Re-ID、时序预测、表格数据、定量 Alpha、医学影像、音频分类、数学推理、游戏 AI (PTCG)、细胞追踪 (Biohub)、ONNX 网络压缩 (NeuroGolf)、地质预测 (ROGII)、LLM 推理等领域。

**核心成果**：Agent 从首个竞赛需要 **2 周**达到 Top 10%，进化到 **Top 5%** — 整整 **14x 加速**，全部归功于累积和结晶的知识。

---

## 🆕 v0.8.4 新版本 (2026-07-10) - Chronos-2 + Covariates 突破

### 🏆 Store Sales 历史最佳 LB 0.39525（AG 1.5 + Chronos-2）

`autogluon-timeseries-strategy` skill 大幅扩展，加入 AG 1.5 Chronos-2 + known_covariates 工作流：

**关键发现**：AG 1.5 的 **Chronos-2** + `known_covariates_names`（Chronos-2 原生支持 covariates）**+ 绕过 HF-mirror 错误的本地 model_path 技巧** → 0.41852 → 0.39525（**-5.6%**）

| 方法 | OOF | LB |
|------|-----|----|
| 手动 LightGBM + stacking | — | 3.0+ |
| AG 1.4 medium_quality | -0.4813 | 0.41852 |
| AG 1.5 best_quality (no Chronos) | -0.4381 | 0.40053 |
| **AG 1.5 Chronos-2 + Chronos + onpromotion covariate** | NaN | **0.39525** |

**Chronos-2 关键创新**（vs Chronos-Bolt 1.4）：
- Zero-shot forecasting SOTA (fev-bench, GIFT-Eval)
- **Native known_covariates 支持**（这是 1.5 独占能力）
- Cross-learning：跨 series 联合预测

**HF-mirror bug 解决方案**：
1. 预下载 `autogluon/chronos-2` 和 `amazon/chronos-bolt-base` 到 `~/.cache/huggingface/hub/`
2. `os.environ.pop('HF_ENDPOINT', None)`（关键！AG 1.5 默认走 hf-mirror.com 不可达）
3. 显式传本地路径作为 `hyperparameters={"Chronos2": {"model_path": LOCAL_C2}}`

### 📚 新 Skills (37 总数, v0.8.4 增强)

## 🆕 v0.8.5 新版本 (2026-07-26) - 跨算法族 Blend 突破

### 🏆 Store Sales 新最佳 LB 0.38444（Chronos-2 + darts LightGBM blend）

新 skill `store-sales-darts-chronos-blend` 记录了突破单模型天花板的方法：

**关键发现**：单个强模型会触顶，**跨算法族 blend**（神经网络 Chronos-2 + 树模型 darts LightGBM）才能继续突破。**同族 blend 无效**（Chronos v1+v2 相关性 >0.99，无增益），跨族 blend 即使各自分数相近也能大幅提升。

| 方法 | LB |
|------|-----|
| AG 1.5 Chronos-2 v2（单模型） | 0.39387 |
| darts LightGBM top-1 方法（单模型） | 0.39953 |
| **Chronos-2 + darts 几何 blend (w=0.55)** | **0.38444** (-0.012) |

**darts 关键价值**：`LightGBMModel(output_chunk_length=1).predict(n=16)` **自动正确处理递归预测**，避免了手写递归的系统性 bug（偏低 44% / 趋势外推失控 4-7x）。

**方法论要点**：
- 几何平均（log 空间线性组合）适配 RMSLE
- 读论坛优先于调参（darts top-1 方法来自比赛 discussion）
- 失败路径已记录（同族 blend、手写递归、Hybrid Ridge 外推、per-family 缩放）





新 skill `autogluon-timeseries-strategy` 解决了时序预测的特殊需求：

**关键发现**：`autogluon.tabular.TabularPredictor` ≠ `autogluon.timeseries.TimeSeriesPredictor`

**Store Sales 实证**（N=3M, 33 families × 54 stores × 1684 days）：
- AG TimeSeriesPredictor `medium_quality` 300s → **LB RMSLE 0.41852**
- 历史最佳（手动 stacking）：~0.4-0.5
- **4 分钟**达到手动 1+ 小时水平

**核心要点**：
- 必须用 `TimeSeriesDataFrame.from_data_frame()` 转换格式
- 需要指定 `freq`（'D'/'H'/'M'）
- AG 自动处理 lag features / rolling stats / 时间序列 CV
- 不要手动加 lag features（AG 内部自动生成）

**⚠️ AG 1.5 升级警告**（2025-12-19 release）：
- Python 3.10+ required（1.4 兼容 3.9）
- 1.4 训练的模型不能在 1.5 加载
- `chronos_*` preset 系列完全删除，改用 `chronos2_*`
- 80% win rate vs 1.4，10min 1.5 > 2hr 1.4

### 📚 新 Skills (37 总数, v0.8.3 新增 1 个)

### 🏆 最新成就

| 竞赛 | 最佳成绩 | 方法 |
|------|----------|------|
| **PTCG AI Battle (Simulation)** | rank 219/4164 (top 5.3%), LB 967.8 | Nithin maktha Archaludon fork + TrueSkill 策略 + CPU eval harness |
| **Playground s6e7** | **0.94942 LB** (top 500/720) | LGB+XGB+CAT blend, CatBoost-heavy 权重最优 |
| **NeuroGolf 2026** | 7228.04 LB (rank 485/2893) | ONNX 公开 bundle fork + 独立 Conv 权重求解工具链 |
| **ROGII Wellbore** | Pipeline A OOF 10.38 (forking 7159 baseline) | ravaghi artifacts + koolbox + Pipeline-A-only trim |
| **Biohub Cell Tracking** | v1 CPU kernel pushed | LB810 UNet+ILP fork (GPU 待配额) |
| **House Prices** | 0.11750 LB | 用户混合 V16+V17 (30/70) |
| **Spaceship Titanic** | 0.80780 LB | SST_v2 Top-5 CatBoost |

### 📚 新 Skills (35+ 总数, v0.8.0-v0.8.1 共新增 7 个)

**v0.8.0 竞赛特定 Skills (3个)**:

| Skill | 用途 | 验证数据 |
|-------|------|----------|
| **trueskill-simulation-competition-strategy** | Kaggle 模拟竞赛 TrueSkill 评分策略（重提交=重置收敛、latest-2规则、high-roll） | PTCG 8+ submissions, 论坛 rank 4-9 共识 |
| **code-competition-artifact-pipeline** | 代码竞赛 fork 公开 baseline 的 artifact 依赖管理 | ROGII/Biohub/NeuroGolf 3个竞赛验证 |
| **onnx-minimal-network-design** | ONNX 最小网络设计（Conv权重求解、Gather排列、sparse限制） | NeuroGolf 400 task 扫描 + 独立工具链 |

**v0.8.1 通用知识 Skills (4个)**:

| Skill | 用途 | 验证数据 |
|-------|------|----------|
| **kaggle-cognitive-cost-optimization** | Kaggle 配额认知成本优化（0.6×BEST_PUBLIC规则、3-quota-first策略、配额决策树） | 20+ 竞赛, mean ratio 0.99 |
| **kaggle-oof-lb-validation-protocol** | OOF/LB 验证协议（5种gap来源、4步确认协议、3-strike规则、不对称gap模式） | 8+ 竞赛实证表 |
| **kaggle-competition-type-strategy** | 6种竞赛类型完整策略（标准/代码/模拟/研究/Playground/LLM）+ 48h决策树 + 配额分配 | 20+ 竞赛验证 |
| **knowledge-crystallization-feedback-loop** | 知识结晶循环（实验→提取→分类→激活→遗忘）+ 3层架构 + AutoMem集成 | 109+ memories, 35 skills |

### ✏️ 增强 Skills (v0.7.0 增强 2 个)
- **ml-sweet-spot** — 新增 CatBoost-First 证据 + AutoGluon-First 对比
- **kaggle-optimal-blending** — 新增 asymmetric-blending (30/70) 原则

---

## 🚀 快速开始

### 👤 人类用户

1. 阅读 [主指南](docs/cultivating-ml-agent-expert.md) (1088 行, ~30 min)
2. 浏览 [示例 skills](skills/examples/) — **43+ skills** 覆盖表格、NLP、视觉、时序、游戏AI、ONNX、知识结晶、推荐
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
│   └── examples/                # 43+ 个真实 skills
│       ├── time-series-walk-forward-validation/  # 🆕 v0.9.0
│       ├── cross-competition-feature-transfer/   # 🆕 v0.9.0
│       ├── feature-engineering-saturation-detection/ # 🆕 v0.9.0
│       ├── autogluon-first/            # 🆕 v0.7.0
│       ├── autogluon-preset-strategy/  # 🆕 v0.8.2 — Tabular preset 选择
│       ├── autogluon-timeseries-strategy/  # 🆕 v0.8.3 — 时序专用 API
│       ├── catboost-first-tabular/     # 🆕 v0.7.0
│       ├── cv-lb-gap-acknowledgment/   # 🆕 v0.7.0
│       ├── claudeception/              # 自动 skill 提取
│       ├── three-layer-wisdom-extraction/
│       ├── agent-nurture-framework/
│       ├── ml-sweet-spot/              # ✏️ v0.7.0 更新
│       ├── kaggle-optimal-blending/     # ✏️ v0.7.0 更新
│       └── ... (43+ 总数)
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
| 16 | **PTCG AI Battle** | **博弈 RL** | **🆕 v0.8.0: V_net + 2-ply search = 59% vs rank-304 heuristic (Deep RL Phase 2) — v0.8.1: ladder-drift lesson (970→770) + meta-aware regression gate** |

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

### v0.9.0 (2026-08-05) — Time-Series & Recommendation Era

**新增 3 个 skills(基于零售 SKU 推荐项目实战经验):**
- `time-series-walk-forward-validation` — 时间序列项目标准评估规范(替代 K-fold)、8 项泄露检查清单、理论上限计算
- `cross-competition-feature-transfer` — 跨竞赛特征迁移方法论(Instacart 2017 → 推荐系统,F1 +12pp 案例)
- `feature-engineering-saturation-detection` — "特征工程已饱和"的 4 个判断信号 + 何时切换范式

**关键洞察:**
- 时间序列问题必须用 walk-forward,K-fold 会让 CV 比线上高 5-15pp
- 跨相似竞赛(top 5% 方案)迁移特征是最值钱的 ROI,可达 10x 于手工造特征
- 特征工程会饱和,连续 3-4 版无提升就应换数据/换范式,别再调参

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
- 19 → 43+ skills
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

**最后更新**: 2026-08-05 | **版本**: 0.9.0 | **总 Skills**: 43+ | **总竞赛**: 20+

Made with ❤️ for the ML community | 用 ❤️ 制作，献给 ML 社区

</div>
