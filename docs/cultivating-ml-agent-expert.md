# 培养 ML Agent 专家：从零到竞赛 Top 10% 的完整指南

> 基于 Claude Code + 120+ 技能 + 13 个竞赛/项目的实战经验
>
> 历时 2 个月（2026年2月-4月），涵盖 Re-ID、时间序列预测、表格数据、
> 量化Alpha、医学图像分割、数学推理、音频分类、篮球预测、
> 软件缺陷预测、古卷轴分割、时空图预测等多个ML领域

---

## 目录

1. [核心理念：Nurture-First Development](#1-核心理念nurture-first-development)
2. [系统架构：三层知识体系](#2-系统架构三层知识体系)
3. [培养工作流：五阶段学习闭环](#3-培养工作流五阶段学习闭环)
4. **[关键方法论与 SOP（最重要）](#5-关键方法论与-sop)**
5. [项目实战案例](#4-项目实战案例)
6. [核心技能清单与框架（参考）](#6-核心技能清单与框架)
7. [反模式与踩坑记录](#7-反模式与踩坑记录)
8. [工具链与基础设施](#8-工具链与基础设施)
9. [效果衡量与能力成长](#9-效果衡量与能力成长)
10. [实施路线图](#10-实施路线图)

> **阅读建议**: SOP和方法论（第5章）是本文最核心的内容，决定了培养体系的框架。
> 技能清单（第6章）和反模式（第7章）是参考材料，根据实际项目需要查阅。

---

## 1. 核心理念：Nurture-First Development

### 1.1 核心原则

**不是预编程，而是通过对话交互"培养"Agent。**

传统方法尝试在 system prompt 中预定义所有知识，但ML领域的知识太广、变化太快。
正确的方法是建立一套**知识结晶循环**：在实践中遇到问题 → 提取可复用模式 →
组织成结构化技能 → 在下次遇到类似问题时自动激活。

### 1.2 关键数据

| 指标 | 数值 |
|------|------|
| 培养周期 | 2个月（2026年2月-4月） |
| 累计技能数 | 120+ SKILL.md 文件（164个markdown文件） |
| 涉及竞赛/项目 | 13个 |
| 领域覆盖 | Re-ID、时序预测、表格数据、量化Alpha、医学图像、数学推理、音频分类、篮球预测、软件缺陷预测、古卷轴3D分割、时空图预测 |
| 能力成长 | 首个竞赛 2 周达 Top 10% → 最新竞赛 Top 5% → 24小时达 Top 10%（14x 加速） |
| 提取的反模式 | 25+ 个已文档化的 ML 陷阱 |
| 知识来源 | 5本数据科学经典书籍（通过LightRAG学习）+ Top方案代码分析 + arXiv论文 |

### 1.3 与学术框架的对应

| 学术概念 | 我们的实践 |
|---------|----------|
| AIDE (Huang 2024) 的试错学习 | 每个竞赛都有详细的失败记录和突破路径 |
| AutoMind (Zhang 2025) 的知识库 | 120+ SKILL.md 文件组成的三层知识架构 |
| Voyager (Wang 2023) 的技能库 | claudeception 自动提取技能的系统 |
| CoMind (2025) 的记忆架构 | 全局/项目/技能三层记忆系统 |
| Reflexion (Shinn 2023) 的经验反思 | three-layer-wisdom-extraction 技能 |

---

## 2. 系统架构：三层知识体系

### 2.1 架构图

```
┌─────────────────────────────────────────────────────────┐
│ L1: 核心能力层（稳定，极少变化）                          │
│                                                         │
│   ml-expert          — ML 全流程知识（理论+实践）         │
│   data-science-assistant — 数据分析通用技能               │
│   kaggle-competition-best-practices — 竞赛通用工作流      │
│   planning-with-files — 复杂任务的文件化规划              │
│                                                         │
│   来源: 预装 + ML书籍学习                                │
│   更新频率: 月级                                         │
├─────────────────────────────────────────────────────────┤
│ L2: 领域技能层（项目积累，持续增长）                      │
│                                                         │
│   竞赛技能:                                             │
│     kaggle-top-performer-replication                    │
│     kaggle-optimal-blending                             │
│     kaggle-mlflow-tracking                              │
│     adversarial-validation-implementation               │
│                                                         │
│   模型/训练技能:                                         │
│     ensemble-model-correlation-trap                     │
│     ml-sweet-spot                                       │
│     per-category-modeling-backfire                      │
│     catboost-multicore-config                           │
│                                                         │
│   领域专项技能:                                          │
│     ts-day-specific-forecasting                         │
│     jaguar-reid-2026-sota-framework                     │
│     brain-improve-alpha-performance                     │
│     sc-tir-mathematical-reasoning                       │
│                                                         │
│   来源: claudeception 从实战中自动提取                    │
│   更新频率: 周级                                         │
├─────────────────────────────────────────────────────────┤
│ L3: 智慧原则层（跨领域通用，抽象级别最高）                │
│                                                         │
│   three-layer-wisdom-extraction 产出:                    │
│     - "局部最优解比失败更危险"                           │
│     - "训练-推理分布不匹配导致系统性偏差"                 │
│     - "同一干预在不同系统中的效果取决于交互作用"           │
│                                                         │
│   来源: 三层抽象提取                                     │
│   更新频率: 项目里程碑时                                 │
└─────────────────────────────────────────────────────────┘
```

### 2.2 记忆管理架构

```
~/.claude/
├── memory.md                          # 全局记忆（环境配置、服务器信息）
├── skills/                            # 120+ 可复用技能
│   ├── ml-expert/SKILL.md             # ML 核心能力
│   ├── claudeception/SKILL.md         # 技能提取系统
│   ├── three-layer-wisdom-extraction/ # 智慧抽象系统
│   ├── kaggle-competition-best-practices/
│   ├── ts-forecasting-stale-lag-methodology/
│   ├── jaguar-reid-2026-sota-framework/
│   └── ... (120+ skills)
├── agents/                            # 专家Agent配置
│   ├── mentor-guide.md                # 导师Agent指南
│   ├── feature-engineer/              # 特征工程专家
│   ├── model-trainer/                 # 模型训练专家
│   └── ensemble-specialist/           # 集成学习专家
└── projects/                          # 项目级记忆
    └── kaggle-store-sales/            # 每个项目的上下文
```

### 2.3 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 技能格式 | Markdown (SKILL.md) | Agent 直接可读，无需解析 |
| 技能描述 | 仅触发条件 | 防止Agent走捷径跳过正文 |
| 技能发现 | 关键词匹配 | 描述中包含搜索关键词 |
| 知识更新 | 增量式（不覆盖） | 用See also交叉引用 |
| 技能提取 | 自动（claudeception） | 人工提取不现实（120+个） |
| 去重 | 5维重叠检测 | 防止同一问题有多个技能 |

---

## 3. 培养工作流：五阶段学习闭环

### 3.1 学习循环

```
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │  ┌─────────┐    ┌──────────┐    ┌──────────┐           │
    │  │ 1. STUDY │───▶│ 2. VERIFY│───▶│ 3. APPLY │           │
    │  │ (理论)   │    │(Notebook)│    │(实战)    │           │
    │  └─────────┘    └──────────┘    └────┬─────┘           │
    │       ▲                              │                   │
    │       │                              ▼                   │
    │  ┌─────────┐    ┌──────────┐    ┌──────────┐           │
    │  │ 5. PLAN  │◀──│ 4. EXTRACT│◀──│ Encounter│           │
    │  │ (规划)   │    │(结晶化)  │    │ Problems │           │
    │  └─────────┘    └──────────┘    └──────────┘           │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
```

### 3.2 各阶段详解

#### Stage 1: Study（理论输入）

**输入源**：
- ML教材（如《Hands-On ML》）
- Kaggle竞赛Top方案
- arXiv 论文
- 官方文档（PyTorch, LightGBM, etc.）

**实施方法**：
```bash
# 将学习内容上传到 NotebookLM 建立RAG知识库
notebooklm create "{竞赛名} 竞赛资料"
notebooklm source add /path/to/research_notes.md --wait

# 关键：必须转化为Agent可读的Markdown文件
# 而非仅存储为PDF
```

**最佳实践**：
- 每个竞赛至少收集 5 个Top方案的代码
- 用 anything-to-notebooklm 技能自动抓取网页/论文
- 学习内容要与当前项目直接相关，不盲目扩展

#### Stage 2: Verify（验证集成）

**目标**：将书本知识转化为可验证的实践知识

```python
# 在 Notebook/Colab 中验证关键概念
# 例：验证 DINOv3 的特征提取
from transformers import AutoModel
model = AutoModel.from_pretrained("facebook/dinov3-large")
# 验证特征维度、归一化方法、输出格式

# 验证后记录边界条件和 gotchas
```

**关键**：这一步将"书本知识"变为"实践知识"，是最容易被忽略但最有价值的步骤。

#### Stage 3: Apply（实战应用）

**输入**：Kaggle 竞赛、真实项目
**输出**：性能指标、Bug发现、工作流洞察

**竞赛实战 SOP**：

```
1. kaggle-data-format-first     → 先验证数据格式，避免浪费研究时间
2. kaggle-competition-best-practices → 建立知识库，研究Top方案
3. kaggle-feature-boundary       → 系统测试特征边界
4. 训练基线模型 → 分析CV/LB差距
5. 迭代改进 → 每轮提取技能
```

#### Stage 4: Extract（知识结晶）

**工具**：claudeception 技能

**触发条件**（满足任一即提取）：
- 非显而易见的解决方案（调查 >10 分钟）
- 错误解决（误导性错误信息、非显而易见的根因）
- 变通方案发现（工具/框架限制需要实验）
- 配置洞察（项目特定设置与标准不同）
- 试错成功（多次尝试后找到可行方案）

**提取流程**：

```bash
# 1. 重叠检测：搜索现有技能，避免重复
# 2. 双轨分类：
#    - Bug Fix 轨: Problem → Symptoms → Root Cause → Solution → Prevention
#    - Knowledge 轨: Context → Guidance → Why This Matters → When to Apply
# 3. 写入 SKILL.md 文件
# 4. 可发现性检查：描述包含搜索关键词
```

#### Stage 5: Plan（差距分析）

**方法**：能力矩阵评估

```markdown
| 能力维度 | 当前水平 | 目标 | 差距 | 下一步 |
|---------|---------|------|------|--------|
| 时序预测 | ★★★★☆ | ★★★★★ | Day-Specific | 研究神经网路方法 |
| Re-ID | ★★★☆☆ | ★★★★★ | 多模型集成 | 学习SOLIDER-ReID |
| 表格数据 | ★★★★☆ | ★★★★☆ | 无 | 转向其他领域 |
```

### 3.3 知识加速器

经过 2 个月的培养，Agent 的能力增长呈指数级：

| 时间点 | 竞赛 | 达到 Top 10% 所需时间 | 关键积累 |
|--------|------|---------------------|---------|
| 2月初 | S6E2 表格分类 | ~2 周 | 从零建立kaggle技能体系 |
| 2月中 | WorldQuant Brain Alpha | 持续 | Alpha表达式验证、数据字段分析 |
| 2月中 | ISEC 软件缺陷预测 | ~1 周 | SMOTE、多项式特征、Voting Ensemble |
| 2月底 | S6E3 表格分类 | ~1 周 | 对抗验证、多项式特征突破 |
| 3月初 | Jaguar Re-ID | ~1 周 | DINOv3、ArcFace、Re-ranking |
| 3月初 | Vesuvius 古卷轴 | RAG研究 | nnU-Net、数据格式验证 |
| 3月初 | BirdCLEF+ 音频分类 | 研究阶段 | PERCH预训练模型 |
| 3月中 | March Madness 篮球 | ~3 天 | Elo/Massey评分、校准预测 |
| 3月中 | AIMO3 数学推理 | ~3 天 | SC-TIR、推理网关系统 |
| 3月中 | nnU-Net 医学图像 | Apple Silicon | 3D分割、MPS训练 |
| 4月初 | Store Sales 时序 | ~3 天 | Day-Specific、Stale Lag方法论 |
| 4月中 | S6E4 表格/图 | ~1 天 | 时空图特征、全部技能复用 |
| 最新 | Store Sales R11 | Top 5% | 持续优化、全部技能积累的结果 |

---

## 4. 项目实战案例

### 4.1 Kaggle Store Sales 时序预测（LB 1.859 → 0.399）

**领域**：多步时间序列预测
**难度**：16 天预测，3M+ 训练样本
**最终成绩**：LB 0.39880（持续优化中）

#### 完整突破路径

```
Phase 1: 基线 (LB 1.859)
├── LightGBM + lag 特征 + forward-fill
├── CV=0.36 但 LB=1.86 (5x 差距!)
└── 诊断: prediction mean=40 vs actual mean=467

Phase 2: 失败的修复尝试
├── Tweedie 目标函数 → LB=1.87 (无改善)
├── 移除短lag → LB=2.84 (更差)
├── 递归预测 → mean=5.87 (错误累积)
├── TE-fill (用TE替换ffill) → mean=34 (更差!)
└── 线性混合 → LB=0.83 (改善但不优)

Phase 3: 几何平均混合突破 (LB 0.670)
├── 关键洞察: RMSLE 是对数空间指标 → 应在对数空间平均
├── 公式: expm1(α·log1p(model) + (1-α)·log1p(TE))
├── 最佳 α=0.01 (1% 模型 + 99% TE)
└── 模型提供"排序信号"，TE提供"量级信号"

Phase 4: Day-Specific 模型突破 (LB 0.399)
├── 核心创新: 训练16个独立模型，每个预测1天
├── 所有模型都使用最后训练日期的真实特征 (无ffill)
├── 无错误累积 — 每天独立预测
└── 模型本身产生正确量级 — 无需后处理
```

#### 提取的关键技能

| 技能名 | 核心洞察 |
|--------|---------|
| ts-forecasting-stale-lag-methodology | Stale lag 导致系统性欠预测的完整方法论 |
| ts-day-specific-forecasting | Day-Specific 直接预测方法（Favorita冠军方案）|
| ts-lag-stale-underprediction | 5-10x 欠预测的诊断和修复 |
| ts-lag-nan-cascade-bug | Lag 特征在测试集上的 NaN 级联问题 |
| per-category-modeling-backfire | 按类别分模型会适得其反（数据量不足时）|

### 4.2 Jaguar Re-ID 美洲豹重识别

**领域**：细粒度图像检索 / 生物特征识别
**挑战**：野外红外相机、低分辨率、遮挡、类内差异大

#### 技术路线

```
Backbone 选择:
├── DINOv3 (Meta, 7B params) — Instance Retrieval 60.7 mAP
├── EVA-02 Large — Re-ID 任务最优 backbone
└── SOLIDER-ReID (CVPR 2023) — 自监督预训练

度量学习:
├── ArcFace Loss — 角度间隔损失
├── Triplet Loss (需检测 silent failure)
└── Cosine Similarity (需正确变换)

后处理:
├── k-Reciprocal Re-ranking
├── 最优混合: 80/20 (re-ranked/raw)
└── H-Blend: 强模型 95% + 弱模型 5%
```

#### 提取的关键技能

| 技能名 | 核心洞察 |
|--------|---------|
| eva02-reid-backbone | EVA-02 是 Re-ID 最优 backbone |
| jaguar-reid-2026-sota-framework | 2026 SOTA Re-ID 完整方案 |
| arcface-normalization-bug | ArcFace 训练准确率随机 → 检查归一化 |
| triplet-loss-silent-failure | Triplet Loss 无声失败检测 |
| reid-feature-similarity-calibration | 低/负相似度修复 |
| reid-pretrained-model-protocol | 预训练模型测试协议 |
| kaggle-optimal-blending | 80/20 最优混合法则 |
| ensemble-model-correlation-trap | 模型相关性陷阱 |

### 4.3 WorldQuant Brain Alpha 量化

**领域**：Alpha因子挖掘
**平台**：WorldQuant BRAIN

#### 工作流

```
brain-nextMove-analysis     → 每日报告: 当前Alpha状态、下一步建议
brain-dataset-exploration   → 深入分析数据集字段
brain-datafield-exploration → 6种方法评估新数据字段
brain-feature-implementation → 从想法到Alpha表达式
brain-calculate-alpha-selfcorrQuick → 自相关和PPAC计算
brain-improve-alpha-performance → 5步改进工作流
brain-explain-alphas        → 解释Alpha逻辑
brain-how-to-pass-AlphaTest → 通过提交测试的完整指南
```

**Alpha改进 SOP**：
1. 收集Alpha信息（Sharpe, Fitness, Turnover）
2. 评估核心数据字段（稀疏度、频率、分布）
3. 基于arXiv理论提出改进（70%想法 + 30%参数）
4. 多参数模拟测试
5. 验证和迭代（3-5轮）

### 4.4 时空图预测竞赛

**领域**：图结构上的时空预测（如洪水、交通）
**关键洞察**：

```python
# 图聚合特征 > 单节点特征
elevation:        1179 importance
degree:            345 importance
neigh_elev_mean:  1239 importance  ← 最重要!
```

**提取技能**：`spatiotemporal-graph-feature-engineering`

### 4.5 AIMO3 数学推理

**领域**：AI数学奥林匹克
**方法**：SC-TIR（Self-Consistency with Tool-Integrated Reasoning）

```
参数: N=4-48 条推理路径, M=1-4 轮迭代
每条路径: LLM 生成 → Python 执行 → 结果验证
最终答案: 多数投票
```

**关键技能**：
- `sc-tir-mathematical-reasoning` — SC-TIR 完整算法
- `aimo3-inference-gateway-system` — 竞赛推理网关系统

### 4.6 BirdCLEF+ 2026 音频分类

**领域**：野生动物音频物种识别
**挑战**：234个物种（鸟类、两栖、哺乳、爬行、昆虫），巴西Pantanal湿地

```
任务: 从音频中识别234个野生动物物种
数据: 35,549个音频文件
评估: Macro-averaged ROC-AUC
技术: Google PERCH预训练模型 + 音频特征工程
```

**关键学习**：预训练音频基础模型（如PERCH）远优于从零训练。

### 4.7 March Madness 2026 篮球预测

**领域**：体育赛事预测（Elo/Massey评分系统）
**挑战**：Stage1（常规赛预测）+ Stage2（锦标赛预测）

```
技术栈:
├── Elo Rating System — 球队实力评分
├── Massey Rating — 考虑比赛强度的评分
├── 校准预测 — Stage2精确校准
└── GBDT 集成 — 多模型预测融合
```

### 4.8 ISEC 2026 软件缺陷预测

**领域**：静态代码指标 → 软件缺陷预测（二分类）

```
关键技术:
├── SMOTE数据增强 — 不平衡分类 (smote-data-augmentation-classification)
├── 多项式特征突破 — +0.017 LB (tabular-polynomial-features)
├── Voting Ensemble — 多模型投票集成
├── 对抗验证 — train/test分布检查
├── MLflow实验追踪 — 超参数优化记录
└── 最终方案: XGBoost + LightGBM + CatBoost 集成
```

**重要发现**：小数据集（<10K样本）上，多项式特征（degree=2）可以捕获树模型
错过的特征交互，带来显著提升。

### 4.9 Vesuvius Challenge 古卷轴分割

**领域**：3D微CT图像 → 莎草纸层分割
**挑战**：虚拟展开古代赫库兰尼姆卷轴

```
设计决策 (基于RAG研究 + 对抗Agent验证):
├── Patch Size: 128³ (96³太小无法捕获拓扑连续性)
├── Architecture: nnU-Net ResEncUNetM (3D医学分割SOTA)
├── 评估指标: TopoScore(30%) + SurfaceDice(35%) + VOI(35%)
└── 平台: Apple Silicon MPS

教训 (kaggle-data-format-first):
├── RAG研究预期: 3D TIFF stacks
├── 实际数据: 2D grayscale images (320×320)
└── 关键技能: 先验证数据格式再做研究！
```

### 4.10 表格数据竞赛 (S6E2, S6E3, S6E4)

**关键突破**：

```
S6E2: 对抗验证 + 频率+目标编码
S6E3: 多项式特征突破 (+0.017 LB)
S6E4: 时空图特征 + 全部技能复用 → 24小时达Top 10%
```

### 4.12 quantmind 量化框架 & TradingAgents

**领域**：量化交易Agent系统

```
quantmind:
├── 多模型配置 (config_models.yaml)
├── 嵌入测试 (embedding_test.py)
└── 交易策略研究

TradingAgents-CN:
├── 中文交易Agent框架
└── 基于LLM的交易决策系统
```

### 4.13 LightRAG 图书学习系统

**方法**：通过LightRAG系统让Agent学习ML经典书籍

```
已学习: 5本数据科学经典书籍
方法: LightRAG → NotebookLM → 知识提取 → 技能文件
效果: 将书本知识转化为可执行的实战技能
```

---

## 5. 关键方法论与 SOP

> **这是本文最核心的章节。** 具体的ML技巧（第6、7章）会随着项目和时代变化，
> 但方法论和SOP是跨项目通用的框架，决定了Agent能不能系统性地解决新问题。
> 一个有好的SOP但没有具体技巧的Agent，可以通过实践快速积累；
> 一个有很多技巧但没有SOP的Agent，每次遇到新问题都要从零开始。

### 5.0 方法论全景图

```
┌─────────────────────────────────────────────────────────┐
│                  培养 Agent 的核心方法论                    │
│                                                         │
│  ┌──────────────────┐  ┌──────────────────┐            │
│  │  宏观方法 (Why)    │  │  执行方法 (How)   │            │
│  │                  │  │                  │            │
│  │  • 知识结晶循环   │  │  • 竞赛启动SOP    │            │
│  │  • 三层知识架构   │  │  • 模型调试SOP    │            │
│  │  • 五阶段学习环   │  │  • 技能提取SOP    │            │
│  │  • 元认知反思    │  │  • 集成学习SOP    │            │
│  │                  │  │  • 实验管理SOP    │            │
│  └──────────────────┘  └──────────────────┘            │
│                                                         │
│  核心原则: 方法论 > 框架 > 具体技巧                       │
│  - 方法论: 决定"做什么"(方向正确)                        │
│  - 框架: 决定"用什么做"(工具选择)                        │
│  - 技巧: 决定"怎么做得好"(细节优化)                      │
│  - 技巧会过时, 方法论不会                                │
└─────────────────────────────────────────────────────────┘
```

### 5.1 竞赛启动 SOP（最重要）

```markdown
## Step 1: 数据格式验证 (kaggle-data-format-first)
- 下载数据，检查 train/test shapes
- 识别目标变量、评估指标
- 检查数据类型：时间序列？图像？表格？
- **关键**：先验证再研究，避免方向错误

## Step 2: 建立知识库 (kaggle-competition-best-practices)
- 创建 NotebookLM notebook
- 上传 Top 方案、论坛讨论
- 研究类似竞赛的获胜策略

## Step 3: 基线模型
- 最简单的可行模型
- 建立CV评估框架
- **必须**：建立CV-LB差距的基线

## Step 4: Top方案复制 (kaggle-top-performer-replication)
- 下载Top notebook，逐行分析
- 建立对比表：你的方法 vs Top方法
- 按预期影响力排序技术

## Step 5: 系统化特征边界测试 (kaggle-feature-boundary)
- 一次只变一个变量
- 记录每次实验结果
- 关注 OOF-LB 差距变化
```

### 5.2 模型调试 SOP（系统化排错）

```markdown
## 问题诊断流程

1. 检查 prediction magnitude (预测均值 vs 训练均值)
   - 如果差距 >2x → 分布不匹配问题

2. 检查 CV-LB 差距
   - CV << LB → 过拟合训练集
   - CV ≈ LB → 两者都差，模型需要改进

3. 检查特征重要性
   - Top 特征是否合理？
   - 是否有"泄露"特征？

4. 对抗验证 (adversarial-validation-implementation)
   - 检查 train/test 分布差异
   - AUC > 0.7 → 需要处理分布差异

5. 渐进式验证 (progressive-verification-debugging)
   - 从最简单的测试开始
   - 逐步增加复杂度定位问题
```

### 5.3 技能提取 SOP（claudeception 自动化知识管理）

```markdown
## 提取触发条件 (满足任一)
1. 非显而易见的解决方案 (>10分钟调查)
2. 错误解决 (误导性错误信息)
3. 变通方案 (工具/框架限制)
4. 配置洞察 (项目特定设置)
5. 试错成功 (多次尝试后成功)

## 提取流程
1. 重叠检测: 搜索现有技能的5个维度
   - 问题陈述、根因、解决方案、引用文件、预防规则
2. 分类:
   - Bug Fix: Problem → Symptoms → Root Cause → Solution → Prevention
   - Knowledge: Context → Guidance → Why This Matters → When to Apply
3. 写入: ~/.claude/skills/[descriptive-name]/SKILL.md
4. 质量检查:
   - 描述仅含触发条件 (CSO原则)
   - 解决方案经过验证 (非理论性)
   - 包含搜索关键词
```

### 5.4 实验管理 SOP（可复现的迭代）

```markdown
## 核心原则: 每次实验必须可追溯、可对比、可复现

## 1. 实验命名规范
   格式: R{round}_{approach}_{key_param}
   示例: R10_day_specific_lgb, R11_ensemble_3model, R11b_lgb_full

## 2. 实验记录 (MLflow)
   每次实验必须记录:
   - CV分数 (每个fold的详细分数)
   - LB分数 (提交后记录)
   - 使用的特征列表
   - 模型超参数
   - 数据范围 (全量 or 截断)
   - 特殊处理 (fillna方式, 后处理等)

## 3. 消融实验方法
   - 一次只改一个变量
   - 建立对比表: baseline vs 每个改进
   - 关注 OOF-LB 差距变化

## 4. 提交策略 (kaggle skill)
   - 每日限制: 5次 (Kaggle), 10次 (Playground)
   - 优先提交最有希望的方案
   - 记录每次提交的LB和参数

## 5. 迭代节奏
   - Day 1: 基线 + 数据验证
   - Day 2: Top方案分析 + 快速复制
   - Day 3-5: 系统化改进 (特征 → 模型 → 集成)
   - Day 6+: 精细调优 + 后处理
```

### 5.5 集成学习 SOP

```markdown
## 1. 检查模型相关性 (ensemble-model-correlation-trap)
   - 计算 OOF 预测的 Pearson 相关系数
   - > 0.999 → 集成无效
   - 0.99-0.995 → 可能无效

## 2. 选择混合策略
   - 低相关 (<0.99): 等权平均或加权平均
   - 高相关 (0.995-0.999): Hill-climbing 或 rank averaging
   - 特殊指标 (RMSLE等): 几何平均混合

## 3. 80/20 最优混合 (kaggle-optimal-blending)
   final = (1-ratio) * reranked + ratio * raw
   ratio = 0.20 通常最优

## 4. H-Blend 策略 (强模型主导)
   weights = {strong: 0.95, weak1: 0.02, weak2: 0.02, weak3: 0.01}

## 5. 甜点原则 (ml-sweet-spot)
   - 不是越多越好
   - 系统测试边界: 特征数、模型数、集成种子数
   - 关注 OOF-LB 差距变化
```

---

## 6. 核心技能清单与框架

### 6.1 ML 核心技能

| 技能 | 类别 | 核心洞察 |
|------|------|---------|
| ml-expert | 基础 | ML全流程知识体系 |
| ml-sweet-spot | 方法论 | "过犹不及"——存在最优点 |
| ml-research-validation | 方法论 | 验证ML研究声明的真实性 |
| ensemble-model-correlation-trap | 反模式 | 模型相关性 >0.999 时集成无效 |
| per-category-modeling-backfire | 反模式 | 数据不足时分类别建模适得其反 |
| domain-knowledge-constraints-trap | 反模式 | 领域知识约束可能损害性能 |
| adversarial-validation-implementation | 技术 | 正确实现对抗验证（train vs test） |
| tabular-polynomial-features | 技术 | 小数据集上多项式特征突破 |
| smote-data-augmentation-classification | 技术 | SMOTE数据增强用于不平衡分类 |

### 6.2 时序预测技能

| 技能 | 核心洞察 |
|------|---------|
| ts-forecasting-stale-lag-methodology | Stale lag → 欠预测的完整方法论 |
| ts-day-specific-forecasting | Day-Specific 直接预测（Favorita冠军方案）|
| ts-lag-stale-underprediction | 5-10x 欠预测诊断修复 |
| ts-lag-nan-cascade-bug | 测试集lag特征的NaN级联 |

### 6.3 Re-ID / 计算机视觉技能

| 技能 | 核心洞察 |
|------|---------|
| eva02-reid-backbone | EVA-02 Large 是Re-ID最优backbone |
| jaguar-reid-2026-sota-framework | 2026 SOTA Re-ID完整方案 |
| arcface-normalization-bug | ArcFace准确率随机→检查归一化 |
| triplet-loss-silent-failure | Triplet Loss 无声失败检测 |
| reid-feature-similarity-calibration | 低/负相似度修复 |
| tta-normalization-trap | TTA破坏归一化 |
| tta-pretrained-model-ineffectiveness | 预训练模型上TTA无效 |
| pose2id-training-free-reid | 无训练Re-ID (Pose2ID) |

### 6.4 量化/Alpha技能

| 技能 | 用途 |
|------|------|
| brain-nextMove-analysis | 每日Alpha状态分析报告 |
| brain-dataset-exploration | 深入数据集分析 |
| brain-datafield-exploration | 6种方法评估数据字段 |
| brain-feature-implementation | 想法→Alpha表达式 |
| brain-calculate-alpha-selfcorrQuick | 自相关/PPAC计算 |
| brain-improve-alpha-performance | 5步Alpha改进工作流 |
| alpha-expression-verifier | Alpha语法验证 |

### 6.5 元认知/系统技能

| 技能 | 用途 |
|------|------|
| claudeception | 自动从实战中提取技能 |
| three-layer-wisdom-extraction | 三层抽象提取跨域智慧 |
| agent-nurture-framework | Agent培养的完整方法论 |
| claude-code-memory-management | 跨项目记忆管理 |
| memory | Zettelkasten 知识图谱 |
| claude-skills-manager | 技能清单管理 |
| skill-refresh | 技能更新维护 |

### 6.6 基础设施技能

| 技能 | 用途 |
|------|------|
| wsl-remote-execution | 远程WSL执行Python脚本 |
| wsl-remote-tmux | WSL持久化tmux会话 |
| wsl-ml-crash-prevention | 防止ML训练崩溃 |
| mac-to-windows-ml-migration | Mac→Windows ML迁移 |
| pytorch-gpu-utilization-optimization | GPU利用率优化 |
| python-output-buffering | Python输出缓冲修复 |
| huggingface-mirror-acceleration | HuggingFace镜像加速 |

### 6.7 多Agent团队框架

```yaml
# ~/.claude/agents/ 配置
mentor-guide.md:
  - 导师Agent: 协调多Agent团队
  - 派发任务、审核结果、培养子Agent

feature-engineer:
  - 特征工程专家Agent
  - Target Encoding, 频率编码, 交互特征, 特征选择

model-trainer:
  - 模型训练专家Agent
  - XGBoost/LightGBM/CatBoost训练、调参、评估

ensemble-specialist:
  - 集成学习专家Agent
  - Blending, Stacking, Hill-climbing, 权重搜索

researcher:
  - 研究Agent
  - 文献调研、方案分析、代码搜索
```

---

## 7. 反模式与踩坑记录

### 7.1 ML 核心反模式

| 反模式 | 症状 | 修复 |
|--------|------|------|
| **Stale Lag** | CV=0.36 LB=1.86 | Day-Specific 模型 |
| **Ensemble Correlation Trap** | 集成不如单模型 | 先检查OOF相关性 |
| **Per-Category Backfire** | 分类别LB更差 | 数据量阈值: >200K行 |
| **Domain Constraints Trap** | 加约束后性能下降 | 检查对抗验证AUC变化 |
| **TE Leakage** | CV虚高LB差 | Fold内计算TE |
| **Oil Rolling Bug** | 跨组污染 | 先按日期去重计算 |
| **fillna(0) Trap** | 模型误认为零销量 | 层级填充: sf_dow→sf→f_dow→family |
| **TTA on Pretrained** | TTA无改善或更差 | 预训练模型不需TTA |
| **OOF Evaluation Bug** | OOF分数4.0+ | 只计算非零OOF样本的分数 |
| **Python Buffering** | 日志为空 | flush=True 或 python -u |
| **Data Truncation** | 1年数据不如4年 | 除非计算限制，用全量数据 |
| **CatBoost CPU Slow** | 训练极慢 | 配置 thread_count=-1 |

### 7.2 Agent 培养反模式

| 反模式 | 问题 | 修复 |
|--------|------|------|
| **过度提取** | 每个小技巧都变成技能 | 只提取非显而易见的解决方案 |
| **描述含工作流** | Agent跳过正文走捷径 | 描述仅含触发条件(CSO) |
| **技能重复** | 同一问题多个技能 | 5维重叠检测，更新而非创建 |
| **叙事性写作** | "在2026年2月3日我们发现..." | 结构化知识，不含时间线 |
| **不验证就提取** | 理论性方案不可用 | 只提取经过验证的解决方案 |

### 7.3 时间序列特有反模式

| 尝试 | 结果 | 为什么失败 |
|------|------|-----------|
| Tweedie目标函数 | 无改善 | 不修复stale特征 |
| 移除短lag | 更差 | 丢失太多信号 |
| 递归预测 | 错误累积 | Lag特征传播错误 |
| TE替换ffill | 更差 | 模型期望噪声lag，平滑TE是OOD |
| 对Day-Specific做几何混合 | 灾难性 | Day-Specific已在正确量级，混合破坏精度 |

---

## 8. 工具链与基础设施

### 8.1 核心工具栈

| 工具 | 用途 | 关键技能 |
|------|------|---------|
| Claude Code | 主Agent平台 | claude-code-memory-management |
| LightGBM / XGBoost / CatBoost | GBDT模型 | catboost-multicore-config |
| PyTorch | 深度学习 | pytorch-gpu-utilization-optimization |
| MLflow | 实验追踪 | kaggle-mlflow-tracking |
| Kaggle API | 竞赛提交 | kaggle (限10次/天) |
| NotebookLM | RAG知识库 | anything-to-notebooklm |
| nnU-Net v2 | 医学图像分割 | nnunet-apple-silicon-training |

### 8.2 多环境协作

```
Mac (macOS)          → 日常开发、代码编写、小型实验
Windows WSL (GPU)    → 大规模训练、Re-ID训练
Kaggle Notebooks     → 提交、GPU推理
```

**跨环境工具链**：
- `wsl-remote-execution` — SSH到WSL执行脚本
- `wsl-remote-tmux` — 持久化远程会话
- `wsl-large-file-upload` — 大文件传输
- `ssh-windows-file-transfer` — SCP文件传输
- `mac-to-windows-ml-migration` — 环境迁移

### 8.3 知识管理工具

```
claudeception               → 自动技能提取
three-layer-wisdom-extraction → 智慧抽象
anything-to-notebooklm      → 多源内容→NotebookLM
lightweight-knowledge-graph → 本地知识图谱
claude-skills-manager       → 技能清单管理
skill-refresh               → 技能更新维护
```

---

## 9. 效果衡量与能力成长

### 9.1 能力矩阵（2个月后）

| 能力维度 | 2月初 | 4月中 | 增长 |
|---------|-------|-------|------|
| 表格数据竞赛 | ★★☆☆☆ | ★★★★☆ | +2 |
| 时序预测 | ☆☆☆☆☆ | ★★★★☆ | +4 |
| Re-ID / CV | ☆☆☆☆☆ | ★★★☆☆ | +3 |
| 量化Alpha | ★☆☆☆☆ | ★★★☆☆ | +2 |
| 医学图像 | ☆☆☆☆☆ | ★★☆☆☆ | +2 |
| 数学推理 | ☆☆☆☆☆ | ★★☆☆☆ | +2 |
| 音频分类 | ☆☆☆☆☆ | ★★☆☆☆ | +2 |
| 体育预测 | ☆☆☆☆☆ | ★★☆☆☆ | +2 |
| Agent培养 | ★☆☆☆☆ | ★★★★☆ | +3 |
| 跨域迁移 | ★☆☆☆☆ | ★★★★☆ | +3 |

### 9.2 关键指标

| 指标 | 测量方法 |
|------|---------|
| 技能数量 | `find ~/.claude/skills -name "SKILL.md" \| wc -l` |
| 技能覆盖 | 按领域统计技能数 |
| 竞赛成绩 | Kaggle排行榜位置 |
| 提取速率 | 每周新增技能数 |
| 复用率 | 被触发的技能数 / 总技能数 |
| 竞赛加速 | 新竞赛达到Top 10%的时间 |

### 9.3 成长里程碑

```
Week 1-2: 基础技能建立
  - ml-expert, data-science-assistant
  - 第一个竞赛 (S6E2)
  - 提取 ~15 个技能

Week 3-4: 方法论形成
  - kaggle-top-performer-replication
  - adversarial-validation-implementation
  - ensemble-model-correlation-trap
  - 提取 ~30 个技能

Week 5-8: 领域扩展
  - Re-ID (DINOv3, ArcFace)
  - 时空图预测
  - 医学图像分割
  - 提取 ~60 个技能

Week 9-12: 深度突破
  - Store Sales 时序预测 (LB 1.859→0.399)
  - AIMO3 数学推理
  - 元认知技能 (three-layer-wisdom-extraction)
  - 提取 ~120 个技能

持续: 知识结晶化
  - 技能去重、更新、整合
  - 跨域迁移能力增强
  - 新竞赛24小时达Top 10%
```

---

## 10. 实施路线图

### 10.1 第一周：基础搭建

```bash
# 1. 安装 Claude Code
# 2. 创建技能目录
mkdir -p ~/.claude/skills

# 3. 安装核心技能
# 从以下核心技能开始:
# - ml-expert (ML全流程知识)
# - kaggle-competition-best-practices (竞赛工作流)
# - claudeception (自动技能提取)

# 4. 配置记忆管理
# 创建 ~/.claude/memory.md 全局记忆文件
```

### 10.2 第二周：首个竞赛

```bash
# 1. 选择一个适合入门的竞赛
#    推荐: Kaggle Playground Series (表格数据)

# 2. 执行竞赛启动 SOP (见 5.1)

# 3. 每次解决问题后运行 claudeception

# 4. 目标: 提取 10-15 个领域技能
```

### 10.3 第三-四周：方法论形成

```bash
# 1. 参加第二个竞赛 (不同领域)
# 2. 验证技能的跨域复用性
# 3. 安装 three-layer-wisdom-extraction
# 4. 在竞赛里程碑时运行智慧提取
# 5. 目标: 累计 30+ 技能
```

### 10.4 第二个月：深度与广度

```bash
# 1. 挑战更高难度竞赛
# 2. 配置多Agent团队 (mentor-guide + feature-engineer + model-trainer)
# 3. 建立自动化实验追踪 (MLflow)
# 4. 目标: 累计 60+ 技能, 竞赛Top 10%
```

### 10.5 持续：知识结晶化

```bash
# 定期运行:
# - skill-refresh: 清理过时技能
# - claudeception: 提取新技能
# - three-layer-wisdom-extraction: 跨域智慧
# - 竞赛能力矩阵评估
```

---

## 附录 A: 技能模板

### Bug Fix 模板
```markdown
---
name: [descriptive-kebab-case-name]
description: |
  Use when [triggering conditions ONLY].
---
# [Skill Name]

## Problem
[What broke]

## Symptoms
[Observable symptoms]

## Root Cause
[Technical cause]

## Solution
[Step-by-step fix with code examples]

## Prevention
[How to avoid recurrence]

## References
[Optional: URLs to docs]
```

### Knowledge 模板
```markdown
---
name: [descriptive-kebab-case-name]
description: |
  Use when [triggering conditions ONLY].
---
# [Skill Name]

## Context
[What situation prompted this]

## Guidance
[Recommended practice with examples]

## Why This Matters
[Impact of following/not following]

## When to Apply
[Conditions and boundary cases]

## Notes
[See also: links to related skills]
```

## 附录 B: 关键命令速查

```bash
# 查看所有技能
find ~/.claude/skills -name "SKILL.md" | sort

# 统计技能数量
find ~/.claude/skills -name "SKILL.md" | wc -l

# 搜索特定领域技能
grep -r "Re-ID" ~/.claude/skills/*/SKILL.md

# 提交Kaggle
kaggle competitions submit -c <comp> -f submission.csv -m "message"

# 创建NotebookLM知识库
notebooklm create "竞赛资料"
notebooklm source add research.md --wait

# 检查远程GPU
ssh <user>@<gpu-server> "nvidia-smi"
```

---

> **总结**: 培养 ML Agent 专家的关键不是预定义所有知识，而是建立一套
> **知识结晶循环**——在实战中遇到问题 → 自动提取可复用模式 →
> 组织成结构化技能 → 在下次类似问题中自动激活。2 个月 120+ 技能的积累
> 使 Agent 从需要 2 周达到 Top 10% 进化到 24 小时完成同样目标。
>
> **方法论 > 框架 > 技巧**: SOP决定方向，框架提供工具，技巧优化细节。
> 技巧会随时间过时，但系统化的方法论（竞赛启动SOP、模型调试SOP、
> 技能提取SOP）是跨项目通用的核心能力。
>
> 本文档本身就是一个活的文档——随着新项目和技能的积累，应持续更新。
