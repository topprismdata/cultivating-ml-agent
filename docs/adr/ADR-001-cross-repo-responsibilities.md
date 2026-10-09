---
adr: "001"
title: 跨仓库职责与禁区（Cross-Repo Responsibilities & No-Go Zones）
status: Proposed
date: 2026-10-09
plan_baseline: /Users/ghb/Downloads/TopPrism_ML_Agent_Ontology_Upgrade_v0.1.md
related: [ADR-002-decision-trace-boundary.md]
---

# ADR-001：跨仓库职责与禁区

## 状态

**Proposed**（2026-10-09）

本 ADR 对应计划基线 v0.1（`/Users/ghb/Downloads/TopPrism_ML_Agent_Ontology_Upgrade_v0.1.md`，下称"计划 v0.1"）§12 第一批工程任务中的 "ADR-001 确定跨仓库职责和禁区"。主要依据计划 v0.1 §2「项目边界与仓库职责」、§3「跨领域语义设计」、§13「参考仓库与待核事项」。状态为 Proposed，等待各仓库维护方评审后转为 Accepted。

## 背景

计划 v0.1 提出 cultivating-ml-agent 本体化升级，但其 §13 自述为"设计基线……非已经实施的事实声明"，并列出待核事项。在起草本 ADR 时，以下仓库事实已核验：

1. **visit 谱系澄清**：计划 v0.1 §13 所列 `visit-scheduling-optimizer` 在 GitHub 上已 404（仓库不存在）。实际的 visit 谱系由三个仓库构成：
   - **VisitIR**（`github.com/topprismdata/visit-ir`）：合同-相位本体，零求解器依赖，纯函数风格；
   - **VisitModel**（`visitmodel`）：L2 求解器；
   - **Territory-IR**：同属 IR 层的领域本体仓库。
   这一谱系证明"IR 与求解器分离"在本组织已有落地先例，ML Agent 的升级应当沿用同一模式，而不是让 ML Agent 吸收求解职能。
2. **prism-ontology**（HEAD `9d93a99`）：`core.ttl` 已发布类为 Entity / Role / Organization / Event / Activity / Observation / DerivedEstimate / InformationObject / Evidence / Policy / Constraint / Eligibility / Decision / Plan / SpatialGeometry。
3. **agent-nurture-framework**（ANF）：已发布 4 个 schema——`capability-record` / `experience-record` / `evidence-envelope` / `policy-record`。
4. **skill-tester**、**topprismwiki**、**cultivating-ml-agent**：参与协作的功能仓库（测试门禁、知识发布、ML 实验实现）。

计划 v0.1 §3 列出 14 项跨领域共享核心（WorldStateRef、Task/Goal、Constraint、Candidate、Plan、Execution、Observation、Estimate、Evidence、Claim、Decision、Outcome、Capability），并要求"均须逐项核对现有注册表，未发布者视为提案"。若不先钉死各仓库职责边界与禁区，升级过程中会出现概念重复治理、职责漂移、以及 ML Agent 越权改写上层本体的风险。

## 决策

### 1. 职责矩阵

按计划 v0.1 §2，各仓库职责与明确不负责的事项如下：

| 仓库 | 职责 | 明确不负责 |
|---|---|---|
| **prism-ontology** | 概念治理、SHACL 校验、版本兼容、已发布通用语义与领域 Profile；新概念必须经过提案审查（已有 MS-PROP-021、MS-PROP-022 先例） | 不承载 ML 领域的实验执行逻辑，不直接采纳任何仓库单方面上推的概念 |
| **cultivating-ml-agent** | ML 领域的 ExperimentIR、ML Profile、实验规划与执行适配、MLflow 映射、DecisionTrace 生成 | 不直接向 prism-ontology 推概念；不充当通用业务求解器 |
| **agent-nurture-framework** | 能力候选、验证、迁移、治理与授权规则（围绕已发布的 4 个 schema） | 不承担训练运行时 |
| **skill-tester** | 触发测试、负例、跨任务迁移与回归测试 | 不定义本体概念，不发布知识 |
| **topprismwiki** | 只收录经过核验的组织知识及其来源 | 不将实验日志未经核验直接发布为知识 |
| **PJP / 仓储等领域引擎** | 消费经验证的预测/估计，执行自身约束与优化，反馈决策结果 | 不要求 ML Agent 代执行领域业务约束 |
| **VisitIR / VisitModel / Territory-IR** | visit 谱系参照：VisitIR（IR 层，纯函数、零求解器依赖）与 VisitModel（L2 求解）分离 | 不与 ML Agent 共享求解器或存储 |

### 2. 禁区（负面清单）

1. **ML Agent 不得直接向 prism-ontology 推送概念**。一切新概念（含净新增与既有类语义变更）必须走 prism-ontology 提案流程，先例为 MS-PROP-021 / MS-PROP-022。
2. **ML Agent 不得变成通用业务求解器**。PJP（频次、间隔、容量、路线）与仓储的求解器保持独立；ML Agent 只输出经验证的估计及其置信范围。此模式与 VisitIR（IR）/ VisitModel（求解）的既有分离同构。
3. **MLflow 是执行事实来源之一，不是本体注册表**。MLflow 记录不自动产生组织结论，也不直接映射为 prism-ontology 类。
4. **ANF 不承担训练运行时**；训练执行属于 cultivating-ml-agent 及其适配层。
5. **skill-tester 的测试结果不直接改写本体**；测试结论作为能力候选门禁的证据输入。
6. **topprismwiki 不直接发布实验日志**；只有经核验的知识摘要可进入 wiki（计划 v0.1 §2、§6 第 10 步）。

### 3. 计划共享核心 vs prism-ontology core.ttl 映射表

对计划 v0.1 §3 的 14 项共享核心逐项对照 core.ttl（HEAD `9d93a99`）已发布类：

| 计划共享核心概念 | core.ttl 对应 | 结论 |
|---|---|---|
| Observation | `prism-core:Observation` | 可映射 |
| Estimate | `prism-core:DerivedEstimate` | 可映射 |
| Evidence | `prism-core:Evidence` | 可映射 |
| Constraint | `prism-core:Constraint` | 可映射 |
| Decision | `prism-core:Decision` | 可映射 |
| Plan | `prism-core:Plan` | 可映射 |
| WorldStateRef | 无已发布类 | **净新增**，需提案批次 |
| Task | 无已发布类 | **净新增**，需提案批次 |
| Goal | 无已发布类 | **净新增**，需提案批次 |
| Candidate | 无已发布类 | **净新增**，需提案批次 |
| Execution | 无已发布类 | **净新增**，需提案批次 |
| Claim | 无已发布类 | **净新增**，需提案批次 |
| Outcome | 无已发布类 | **净新增**，需提案批次 |
| Capability | 无已发布类 | **净新增**，需提案批次 |

汇总：**6 项可映射**（Observation、Estimate、Evidence、Constraint、Decision、Plan），**8 项净新增**（WorldStateRef、Task、Goal、Candidate、Execution、Claim、Outcome、Capability）。

处置规则：

- 6 项可映射概念在 ML Profile / ExperimentIR / DecisionTrace 中直接引用对应 `prism-core:` 类，不得另造同义类。
- 8 项净新增概念以**一个提案批次**整体提交 prism-ontology（一次提案、逐项审查、统一版本锁定），不逐个临时上推；批次通过前，这些概念仅存在于 ML 局部命名空间（计划 v0.1 §3 "ML 局部扩展"同名规则：先局部 Profile，再申请晋升）。
- ML 局部扩展概念（MLTask、DatasetSnapshot、FeatureSet、ValidationProtocol、MetricDefinition、Hypothesis、ExperimentPlan、ExperimentRun、EvaluationResult、ModelArtifact、TransferEvidence）属 cultivating-ml-agent 的 ML Profile 命名空间，默认不申请进入 core.ttl。

## 备选方案

1. **每个 ML 概念随需单独提案** —— 否决：碎片化提案使版本兼容审查成本线性增长，且无法对 ExperimentIR 的依赖集合做一次性版本锁定，违反计划 v0.1 G0 门禁（"全部外部 Profile/依赖均有 tag 与 SHA"）。
2. **ML Agent 自建并行本体、日后合并** —— 否决：同一语义概念会出现互斥定义，直接违反计划 v0.1 §10 G0（"同一语义概念不能出现互斥定义"），且合并成本远高于先行提案。
3. **以 ANF 已发布的 4 个 schema 作为跨仓库共享核心** —— 否决：ANF schema 面向能力治理记录（capability/experience/evidence/policy），不是通用概念本体；让它承担 core.ttl 职能即越过 ADR 第 1 节职责矩阵（ANF 管能力治理，prism-ontology 管概念治理）。
4. **维持计划 v0.1 §13 的 visit-scheduling-optimizer 引用不变** —— 否决：该仓库已 404，继续引用会造成集成任务（ADAPTER-002 等）指向不存在的依赖；必须改写为 VisitIR / VisitModel / Territory-IR 真实谱系。

## 后果

**正面**

- 职责边界与禁区一次性成文，P1–P5 各阶段（计划 v0.1 §9）不再逐次重新谈判边界。
- 8 项净新增概念有明确的提案批次路径与先例（MS-PROP-021/022），避免"边实现边上推"。
- visit 谱系澄清后，跨域适配任务指向真实仓库；"IR/求解分离"有组织内先例可援引。

**负面 / 成本**

- 8 项净新增概念在提案批次通过前只能停留在 ML 局部命名空间，跨域共享语义（如 PJP 消费 Outcome）需要等提案周期。
- 职责矩阵要求跨仓库对齐与评审，沟通成本高于单仓库自行决定。

**中性**

- MLflow 地位不变（执行事实源之一），不新增注册表职能，现有训练 CLI 与 MLflow 记录不受影响。

## 回滚

本 ADR 只约束职责边界与提案路径，不引入运行时依赖，因此回滚成本极低：

- 若提案批次被 prism-ontology 否决：ML 局部命名空间中的概念继续以 ML Profile 草案形式存在，6 项可映射概念仍引用 `prism-core:` 类；无数据迁移、无代码回退。
- 现有训练 CLI、MLflow 记录与 Skill 检索保持可用（与计划 v0.1 §11 回滚条款一致）；本 ADR 不改变任何默认执行路径。
- 若职责矩阵在评审中被修订：只需更新本 ADR 并追加新决议编号，不产生运行时副作用。
