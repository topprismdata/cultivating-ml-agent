# SPEC-002: ML Profile 概念与映射规格（v0.1.0 草案）

- 状态: draft（命名空间 local / pending_proposal）
- 日期: 2026-10-09
- 关联:
  - SPEC-001: ExperimentIR 实验契约（`docs/specs/SPEC-001-*.md`，ExperimentIR 线并行交付；字段拼写以其落地版为权威）
  - ADR-001: 共享核心概念净新增提案批次（仅引用，本文档不修改）
  - ADR-002: DecisionTrace 与 ANF experience-record 的边界（`docs/adr/ADR-002-decision-trace-boundary.md`）
  - BOM: `docs/ontology/compatibility.bom.yaml`（G0：外部依赖全部锁定 tag/SHA）
  - 样板: prism-ontology@9d93a99 `profiles/outlet-insight/`

## 1. 目的与范围

SPEC-002 定义 cultivating-ml-agent 机器学习实验域的受管概念层（ML Profile）：概念清单、
到共享核心 prism-core 的映射、以及与 ExperimentIR（SPEC-001）的字段引用关系。
本规格只做**声明式建模**：概念是纯数据契约，零求解器依赖；执行语义归 ExperimentIR
编译产物，家法（VisitIR 风格：纯函数、只能证伪不能证真的快筛分层）约束概念的生命周期设计。

工件清单（`docs/ontology/ml-profile/0.1.0/`）：

| 工件 | 内容 |
| --- | --- |
| `concepts.yaml` | 概念清单与局部命名空间声明 |
| `mappings.yaml` | 概念 → 共享核心映射 + 8 项共享核心净新增提案 |
| `context.jsonld` | 最小 JSON-LD 上下文 |
| `ml.shacl.ttl` | 最小 SHACL 形状（pyshacl 未接入，本版仅语法级自检） |
| `README.md` | 草案定位、样板对齐说明、晋升路径 |

## 2. 概念定义

命名空间：`prism-ml: <prism://ontology/ml/>`，`status: local`、`governance_status: pending_proposal`。

### 2.1 实体概念（绑定共享核心父类）

| 概念 | 中文名 | 父类 | 定义要点 |
| --- | --- | --- | --- |
| `MLTask` | 机器学习任务 | `prism-core:Activity` | 以泛化预测为目标的任务意图；只圈定问题边界，不携带执行状态。 |
| `DatasetSnapshot` | 数据集快照 | `prism-core:InformationObject` | 数据的不可变指纹视图（行集、特征列、时间窗、切分冻结点）；变更产生新快照，禁止原地修改。 |
| `FeatureSet` | 特征集 | `prism-core:InformationObject` | 派生特征及其构造定义；构造必须纯函数、可重放。 |
| `ValidationProtocol` | 验证协议 | `prism-core:InformationObject` | 切分策略与折结构声明（四策略，见 §5）；防泄漏硬门禁的声明载体。 |
| `MetricDefinition` | 度量定义 | `prism-core:InformationObject` | 受管度量、聚合方式与硬门禁阈值。 |
| `ExperimentPlan` | 实验计划 | `prism-core:Plan` | 实验的完整编译单元（任务、快照、特征集、协议、度量、候选假设的绑定）；编译后不得被自然语言代理隐式修改，变更必须产生新版本。 |
| `ExperimentRun` | 实验运行 | `prism-core:Activity` | 计划的一次确定性执行实例；产出 OOF 帧、模型工件与证据信封，不做计划外决策。 |
| `ModelArtifact` | 模型工件 | `prism-core:InformationObject` | 运行落盘的模型对象及元数据（训练配置、指纹、父计划引用）；只读、可复演。 |
| `TransferEvidence` | 迁移证据 | `prism-core:Evidence` | 跨任务/跨竞赛复用经验的依据记录；必须可回指 DecisionTrace（`decision_trace_ref`，ADR-002）。 |

### 2.2 特殊处置一：Hypothesis（假设）

- 定义：对「某改动会改善某度量」的**可证伪**陈述；进入 `ExperimentPlan` 候选集合，
  运行后由证据支持或证伪——只能证伪，不能证真（家法）。
- 特殊性：其目标父类 `prism:core/Claim` **尚不在共享核心**。Claim 属共享核心
  净新增提案（`mappings.yaml: core_proposals`，归属 **ADR-001 提案批次**）。
- 处置：`Hypothesis` 的 `governance_status` 标 `proposed_local`，随 Claim 提案一并走
  治理；治理通过前 Hypothesis 仅在本 Profile 与 ExperimentIR 草案内使用。

### 2.3 特殊处置二：EvaluationResult（评估结果，显式不设实体）

- **本 Profile 不设 EvaluationResult 实体**（ADR-002 决议）。
- 评估结果一律落成 **ANF evidence-envelope**（agent-nurture-framework@33f444b
  `schemas/evidence-envelope.schema.json`）；完整决策保真唯一归属 DecisionTrace，
  ML Profile 侧只投影 `decision_trace_ref` 指针。
- SHACL 侧以 `prism-ml:ProhibitEvaluationResultShape` 禁止实例化（沿用 outlet-insight
  样板的执行禁令写法：`sh:path rdf:type ; sh:maxCount 0`）。
- 理由：评估结果本质是「证据 + 指针」，不是需要独立生命周期的业务实体；
  设实体会诱导系统把分数当成可改写状态，破坏 append-only 证据链。

## 3. 映射理由

| 概念 | 目标父类 | 理由摘要 |
| --- | --- | --- |
| MLTask | Activity | 有时间边界、有意图产出的活动；不落 InformationObject，避免把意图物化成文档。 |
| DatasetSnapshot | InformationObject | 快照是关于数据集的**陈述**（指纹+冻结点），不是数据本身。 |
| FeatureSet | InformationObject | 构造定义是信息对象；纯函数可重放，定义与数据分离。 |
| ValidationProtocol | InformationObject | 切分/折结构声明是可校验的信息对象，与具体数据解耦以便复用。 |
| MetricDefinition | InformationObject | 受管定义；与样板 `prism:insight/MetricDefinition` 定位同构，便于日后对齐。 |
| ExperimentPlan | Plan | 意图与步骤的绑定，编译后版本化；不落 Activity 以区分计划与执行。 |
| ExperimentRun | Activity | 执行实例：消耗 Plan、产出工件与证据，有起止有簿记。 |
| ModelArtifact | InformationObject | 只读、可复演的工件信息对象；引用指纹而非内嵌数据。 |
| TransferEvidence | Evidence | 迁移复用本质是证据，须满足证据可回指性（ADR-002）。 |
| Hypothesis | Claim（提案） | 可证伪主张；父类属净新增提案，随 ADR-001 批次治理。 |

共享核心 8 项净新增（`WorldStateRef / Task / Goal / Candidate / Execution / Claim /
Outcome / Capability`）全部标 `proposed_local`，归属 ADR-001 提案批次；
治理通过前禁止跨 Profile 复用。其中 `Execution` 显式为**纯函数式执行簿记**、
`WorldStateRef` 显式为**只读引用**——两者共同保证 ML Profile 不引入世界状态写入语义。

## 4. 与共享核心的引用约束

- 本 Profile 引用的 prism-core 既有实体以 prism-ontology@9d93a99 为准（BOM 锁定）。
- 8 项净新增在 prism-ontology 治理通过前，任何跨 Profile 复用视为违规
  （`concepts.yaml` 命名空间 note 与 `mappings.yaml` core_proposals note 均已声明）。

## 5. 与 ExperimentIR（SPEC-001）的字段引用关系

ExperimentIR 是 ML 域实验契约：编译后不得被自然语言代理隐式修改，变更必须产生新版本；
任一硬门禁未满足必须拒绝执行（计划 G1）。ML Profile 概念是 ExperimentIR 运行时字段的
**声明式投影**，引用关系如下（SPEC-001 落地后字段拼写以其为权威）：

| ML Profile 概念 | ExperimentIR / 框架锚点 | 引用关系 |
| --- | --- | --- |
| `ExperimentPlan` | ExperimentIR 编译单元（experiment/plan 标识、版本号） | Plan 的 `plan_ref` 指向编译产物标识；版本化重编译产生新 Plan 实例，不覆写。 |
| `ExperimentRun` | ExperimentIR 运行信封（`decision_id`、run 标识） | `runId` 与运行信封一一对应；`decision_id` 是 DecisionTrace 寻址键（ADR-002）。 |
| `ValidationProtocol` | `framework/src/pipeline/splits.py::make_folds` | `splitStrategy ∈ {stratified, kfold, time_based, group}`（SHACL sh:in 封闭词表）；`n_folds`、`random_state`、`val_size_weeks`、`group_col` 为协议参数。 |
| `DatasetSnapshot` | 数据指纹与切分冻结点 | `datasetFingerprint`（sha256）由 ExperimentIR 编译期门禁消费；快照冻结先于任何运行。 |
| `FeatureSet` | 特征构造定义（纯函数管线） | `featureCount` 与构造函数指纹登记；特征列变更 = 新 FeatureSet，禁止原地改。 |
| `MetricDefinition` | `framework/src/pipeline/validate.py` 硬门禁 | `metricName` 与门禁阈值绑定；阈值不满足即拒绝执行（G1 语义在 SPEC-001）。 |
| EvaluationResult（不设实体） | `framework/src/pipeline/oof.py::build_oof_frame`（列 `id` / `target` / `oof_pred`） | OOF 帧是评估结果的物理载体之一；度量值与门禁判定落 ANF evidence-envelope，完整轨迹在 DecisionTrace（ADR-002）。 |
| `ModelArtifact` | run 产物（模型对象 + 元数据） | `artifactUri` 指向落盘工件；元数据含父 Plan 与快照/特征集指纹，保证可复演。 |
| `Hypothesis` | ExperimentIR 候选假设登记（candidate） | 假设进入 Plan 的候选集合；运行后只能被证伪或被证据支持，不允许「证真」表述。 |
| `TransferEvidence` | DecisionTrace 导出的 experience-record（ADR-002） | `decisionTraceRef` 必须为 `er-<decision_id>@<status>` 规范形式（SHACL pattern）；跨竞赛复用经验必须携带。 |

约定：

1. 本表是**契约级引用**：字段拼写以 SPEC-001 落地版为权威；冲突时以 SPEC-001 为准并回灌本表。
2. ML Profile 概念实例由 ExperimentIR 编译/运行流程程序化生成，自然语言代理只能引用、
   不能隐式修改（ ExperimentIR 变更必须走版本化编译）。
3. `decision_id` 的生命周期、状态词表与导出规则见 ADR-002 决议记录（2026-10-09）。

## 6. 治理与晋升

晋升路径（详见 `docs/ontology/ml-profile/0.1.0/README.md`）：
**MS-PROP 提案批次 → prism-ontology 治理评审（命名空间授予 / pyshacl 接入）→ dist 发布**。
治理通过前，本 Profile 全部概念 `status: local`、`governance_status: pending_proposal`
（Hypothesis 及 8 项核心净新增为 `proposed_local`）。
