---
adr: "002"
title: DecisionTrace 语义边界（Decision Trace Semantic Boundary）
status: Proposed
date: 2026-10-09
plan_baseline: /Users/ghb/Downloads/TopPrism_ML_Agent_Ontology_Upgrade_v0.1.md
related: [ADR-001-cross-repo-responsibilities.md]
---

# ADR-002：DecisionTrace 语义边界（本 ADR 第一决议）

## 状态

**Proposed**（2026-10-09）

本 ADR 对应计划基线 v0.1（`/Users/ghb/Downloads/TopPrism_ML_Agent_Ontology_Upgrade_v0.1.md`，下称"计划 v0.1"）§12 第一批工程任务中的 "ADR-002 定义共享 DecisionTrace 语义边界"，是其第一项决议。主要依据计划 v0.1 §5「DecisionTrace v0.1 契约」、§6「关键运行流程」、§10 门禁 G3；涉及仓库职责的部分以 [ADR-001](./ADR-001-cross-repo-responsibilities.md) 职责矩阵为前提。状态为 Proposed，等待 prism-ontology、ANF、cultivating-ml-agent 三方维护方评审。

## 背景

1. 计划 v0.1 §5 定义了 DecisionTrace v0.1 最小字段（decision_id、state_snapshot_ref、intent_ref、alternatives、constraint_check、baseline_ref、selected_option、rejected_options_and_reasons、evidence_refs、decision_actor、authorization_ref、execution_ref、outcome_refs、uncertainty、created_at），并规定：允许 `status=proposed/approved/executed/evaluated/rejected`、采用**不可变事件记录**、修正通过新事件链接旧事件、禁止直接覆盖历史判断。
2. ANF 已发布 `experience-record` schema，其真实字段（已核验）为：`record_id`、`task`、`context`、`decision`、`decision_summary`、`alternatives_considered`、`action`、`execution_trace_refs`、`outcome`、`correction`、`related_skill_ids`、`classification`、`trust_level`、`timestamp`；其中必填（required）为：`record_id`、`task`、`decision`、`decision_summary`、`action`、`outcome`、`classification`、`trust_level`、`timestamp`。
3. ANF 已发布的 `evidence-envelope` schema 强制要求 `measurement_protocol`、`protocol_version`、`metric_version`、`provenance_chain_id`，且其 `source_system` 枚举包含 `'cultivating'`；`evidence_type` 枚举为 `task_success / task_failure / negative_transfer / correction / contradiction / dispute`。
4. 按计划 v0.1 §6 第 8–9 步，DecisionTrace 生成后要经过 Skill Tester 与 ANF 的独立门禁决定是否形成能力候选；ANF 的 `experience-record` 与 ML 的 DecisionTrace 字段高度相邻。若不裁定谁是事实源，两仓库极易演成双向互写，造成同一决策两份不一致记录。

## 决策

### 决议 1：事实源裁定

**DecisionTrace（ML 域，由 cultivating-ml-agent 生成）是决策的唯一事实源。** 其语义为：

- **不可变事件链**：每条 DecisionTrace 是 append-only 的事件序列；
- **状态机**：`status ∈ {proposed, approved, executed, evaluated, rejected}`，状态转移权限按计划 v0.1 §5 明确约定；
- **修正语义**：修正 = 新事件链接旧事件，禁止直接覆盖历史判断。

### 决议 2：experience-record 是派生投影，禁止双向写

ANF `experience-record` 是 DecisionTrace 的**派生投影**（derived projection）：

- 只允许 **DecisionTrace → experience-record** 单向导出，且必须经过一个实现了下述显式字段映射的导出适配器；
- **禁止双向写**：ANF 侧不得回写 DecisionTrace；ANF 自有治理字段（如 `trust_level`、`classification` 的治理语义）不回流事实源。

### 决议 3：逐字段投影映射表

方向：DecisionTrace → ANF `experience-record`。

| # | DecisionTrace 字段 | experience-record 字段 | 投影规则 |
|---|---|---|---|
| 1 | `decision_id` | `record_id`（必填） | 派生映射：`record_id` 由 `decision_id` 生成（如 `er-<decision_id>`），保留按 `decision_id` 反查能力；两者主键不同 |
| 2 | `selected_option` | `decision`（必填） | 直接映射 |
| 3 | `rejected_options_and_reasons` | `decision_summary`（必填） | 聚合映射：与 `uncertainty` 合并为决策摘要文本 |
| 4 | `uncertainty` | `decision_summary`（必填） | 聚合进第 3 行摘要，不单独成字段 |
| 5 | `alternatives` | `alternatives_considered` | 直接映射 |
| 6 | `execution_ref` | `execution_trace_refs` | 引用透传（如 MLflow run 引用） |
| 7 | `outcome_refs` | `outcome`（必填） | 结果引用/摘要透传 |
| 8 | `created_at` | `timestamp`（必填） | 直接映射 |
| 9 | `evidence_refs` | —（配套 `evidence-envelope` 实例） | 不落入 experience-record 单一字段；由 evidence-envelope 实例承载（见决议 5），`context` 可携带引用摘要 |
| 10 | `status` | `classification`（必填） | 派生映射：导出时写入链状态快照；后续状态转移经再次前向导出更新，永不从 ANF 侧发起 |
| 11 | `decision_actor` | `context` | 写入 `context` 说明段（ANF 无独立字段） |

**投影损失**（ANF experience-record 表示不了的字段，映射后丢失，仅可回溯源记录查询）：

| DecisionTrace 字段 | 说明 |
|---|---|
| `constraint_check` | ANF 无对应字段；约束校验细节只存在于事实源 |
| `baseline_ref` | ANF 无对应字段 |
| `authorization_ref` | ANF 无对应字段；其**值**可参与派生 `trust_level`，但字段本身不落入 ANF |
| `state_snapshot_ref` | ANF 无对应字段 |
| `intent_ref` | ANF 无对应字段 |

投影损失字段的处理纪律：**禁止为补齐这五个字段而单方面修改 ANF schema**；列入本 ADR Open Questions，经与 ANF 维护方协商（schema 扩展提案或由关联记录承载）后另行决议。

**ANF 独有字段**（非 DecisionTrace 来源，注明填充方，防止事实源越界代填）：

| experience-record 字段 | 填充方 |
|---|---|
| `task`（必填） | 导出适配器从 `intent_ref` / `state_snapshot_ref` 解析出的任务标识填充 |
| `action`（必填） | 导出适配器从 `execution_ref` 指向的执行事实中提取动作摘要 |
| `correction` | 由 DecisionTrace 修正事件（新事件链接旧事件）生成，记录被修正事件的引用 |
| `related_skill_ids` | 由 ANF 能力候选流程填充，ML Agent 不得代填 |
| `trust_level`（必填） | 由 ANF 治理/授权状态派生；`authorization_ref` 的值可作为输入之一 |

导出适配器契约：experience-record 的 9 个必填字段（`record_id`/`task`/`decision`/`decision_summary`/`action`/`outcome`/`classification`/`trust_level`/`timestamp`）任一无法非空填充时，**导出失败**，不得写空值或占位值。

### 决议 4：evidence-envelope 复用（EvaluationResult 不新建实体）

ML 的 `EvaluationResult` **不新建实体**，落成 ANF `evidence-envelope` 实例：

- evidence-envelope schema 已强制 `measurement_protocol` / `protocol_version` / `metric_version` / `provenance_chain_id`，天然满足计划 v0.1 §10 G2（可复现：固定数据哈希、代码版本、环境与随机种子）对证据链的要求；
- `source_system` 枚举已含 `'cultivating'`，无需扩展；
- `evidence_type` 枚举 `task_success / task_failure / negative_transfer / correction / contradiction / dispute` 已覆盖实验评估语义（含负例迁移与修正），无需扩展；
- DecisionTrace 的 `evidence_refs` 即指向这些 evidence-envelope 实例（经 `provenance_chain_id` 关联）。

### 决议 5：与 ADR-001 职责矩阵的一致性

本裁定与 [ADR-001](./ADR-001-cross-repo-responsibilities.md) 一致：DecisionTrace 生成属 cultivating-ml-agent；ANF 围绕 experience-record 做能力候选、验证、迁移与授权，不承载决策事实；Prism 本体侧的 `prism-core:Decision` 映射与提案批次按 ADR-001 第 3 节处理。

## 备选方案

1. **以 ANF experience-record 作为决策事实源** —— 否决：它缺不可变事件链（修正语义无法表达，只能覆盖或另建记录）、无 `constraint_check`、无 `baseline_ref`，无法满足计划 v0.1 §5 契约与 G3 门禁（100% 被执行实验有批准引用、约束检查与基线）；且按 ADR-001 职责矩阵，决策事实生成属 cultivating-ml-agent，不是 ANF 职责。
2. **双向同步（DecisionTrace ⇄ experience-record 互写）** —— 否决：事实源不唯一，两侧各自演化后必然漂移；ANF 侧的写回无法满足"新事件链接旧事件"的不可变修正语义；违反计划 v0.1 §1 架构原则与 §10 G0（同一语义概念不能出现互斥定义）。
3. **将 DecisionTrace 并入 ANF 单记录（由 ANF 扩展 schema 承载全部 15 字段）** —— 否决：越职责（ADR-001 矩阵：DecisionTrace 生成属 cultivating-ml-agent，ANF 管能力治理）；把 ML 执行细节塞入能力治理记录会造成 ANF schema 膨胀（计划 v0.1 §11 明示"与 ANF 职责重叠"是风险项）。

## 后果

**正面**

- 决策事实源唯一，审计路径清晰：experience-record 经 `record_id` 反查 DecisionTrace 事件链，任何决策都可回放（G3）。
- 投影损失五字段显式成文，避免"以为 ANF 记录是完整决策"的误用。
- EvaluationResult 复用 evidence-envelope 后立即获得 measurement_protocol / provenance_chain_id 等强制治理字段，无需新建实体，满足 G2 可复现门禁。

**负面 / 成本**

- 投影损失五字段在 ANF 侧不可查，跨仓库审计必须回源；对只读 ANF 仓库的消费者，约束检查与基线信息不可见。
- 导出适配器成为必维护组件：字段映射需随两侧 schema 演化同步更新，并需要映射回归测试。

**中性**

- evidence_type 枚举已覆盖 negative_transfer / correction 等实验语义，本决议不要求任何 schema 扩展即可落地。

## 回滚

- **导出适配器可整体停用**：停用后 DecisionTrace 仍为事实源并继续记录，仅不再产生新投影；已导出的 experience-record 保留（`record_id` 内嵌 `decision_id` 可反查），不做删除或改写。
- evidence-envelope 实例为 append-only 证据，停用导出不涉及删除。
- 事实源裁定本身不依赖任何运行时组件；若未来 ANF schema 扩展补齐五个投影损失字段，只需升级决议 3 的映射表，裁定与方向不反转。
- 与计划 v0.1 §11 回滚条款一致：旧训练 CLI、MLflow 记录与现有 Skill 检索保持可用，本 ADR 不改变默认执行路径。

## Open Questions

1. 投影损失五字段（`constraint_check` / `baseline_ref` / `authorization_ref` / `state_snapshot_ref` / `intent_ref`）应向 ANF 提案扩展 experience-record schema，还是由 evidence-envelope / 关联记录承载？待与 ANF 维护方协商。
2. DecisionTrace 状态后续转移（如 `evaluated → rejected` 的修正链）发生时，已导出 experience-record 的更新策略（版本化重导出 vs 原地更新）需与 ANF 侧确认，且必须保持前向单向。
3. `record_id` 派生规则（`er-<decision_id>` 前缀约定）需与 ANF 维护方确认无既有 id 命名冲突。
