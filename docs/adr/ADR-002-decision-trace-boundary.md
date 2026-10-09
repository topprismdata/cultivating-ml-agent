# ADR-002: DecisionTrace 与 ANF experience-record 的边界

- 状态: 已接受 (Accepted)
- 日期: 2026-10-09
- 关联: ADR-001（共享核心概念净新增提案批次，本文档不修改之）、SPEC-002（`docs/specs/SPEC-002-ml-profile.md`）、`docs/ontology/compatibility.bom.yaml`（agent-nurture-framework@33f444b）

## 背景

cultivating-ml-agent 的实验域（ExperimentIR，SPEC-001）与 ML Profile（SPEC-002）需要完整、
可回放的决策轨迹（DecisionTrace）：候选、依据、状态迁移全程簿记。同时，经验沉淀复用
agent-nurture-framework（ANF）的 experience-record 作为跨系统交换格式。

两者的粒度与保真度不同：DecisionTrace 是**完整保真**的决策簿记；experience-record 是
面向复用的**单条经验快照**。必须划清边界，否则会出现两种失败模式：

1. 把 DecisionTrace 整体塞进 ANF 记录——schema 膨胀，ML 域与通用经验域强耦合；
2. 投影时丢失指针——决策不可回指，证据链断裂，复用失去依据。

## 决策

- 完整保真唯一归属 DecisionTrace；ANF experience-record 只承载**有损投影 + 指针**。
- ML Profile 显式不设 EvaluationResult 实体；评估结果一律以 ANF evidence-envelope 落地。
- 投影载体、版本化与 record_id 约束的具体机制，见文末「决议记录（2026-10-09，Open Questions 处置）」。

## Open Questions

- OQ1: 投影是否需要扩展 ANF experience-record schema，以容纳决策轨迹字段？
  ——已决议，见文末。
- OQ2: 决策状态每前进一步，导出的经验记录如何版本化？消费方如何取最新？
  ——已决议，见文末。
- OQ3: ANF `experience-record.schema.json` 的 `record_id` 存在哪些约束？`er-` 前缀
  规范 id 能否直接写入该字段？
  ——已决议，见文末。

---

## 决议记录（2026-10-09，Open Questions 处置）

### OQ1 — 不扩 ANF schema

- **决议**：不扩展 ANF experience-record schema。决策轨迹以**指针**方式携带：
  `experience-record.context`（string 类型）以 `decision_trace_ref=<规范 id 或 dt:// URI>`
  形式嵌入 DecisionTrace 回指。
- **投影保持有损**：投影只保留 decision 摘要与结果；完整候选、依据与状态迁移的
  保真唯一归属 DecisionTrace，任何消费方需要完整轨迹时凭指针回源。

### OQ2 — 状态每前进一步版本化重导出

- **决议**：决策状态每前进一步，即**版本化重导出**一条新经验记录，
  规范 `record_id = er-<decision_id>@<status>`；**append-only**，不覆盖、不改写旧记录。
- 消费方**按前缀取最新**：以 `er-<decision_id>@` 为前缀切片，取最新状态版本；
  `@` 分隔符保证 decision_id 含 `-` 时切片仍无歧义。
- 状态词表由 DecisionTrace 生命周期定义（当前草案：draft → verified → consolidated）。

### OQ3 — record_id 约束核验（核验结果与计划假设不符，已据此更正）

- **核验**（2026-10-09，对 agent-nurture-framework@33f444b 的
  `schemas/experience-record.schema.json` 实际读取）：
  - `record_id` **存在** pattern 约束：`^exp-[A-Za-z0-9._-]+$`；
  - 无 `format` 关键字。
  - 计划假设「record_id 无 pattern/format 约束」**与实际不符**，本决议按实际核验结果更正。
- **决议**：仍采纳 `er-` 前缀写入**导出适配器契约**，拆为两层，保证 append-only 与
  前缀取最新语义不变，同时满足 ANF schema：
  - **规范 id**：`er-<decision_id>@<status>`——DecisionTrace 与消费方寻址、append-only
    台账、按前缀 `er-<decision_id>@` 取最新，一律以规范 id 为准；由 OQ1 的
    `context.decision_trace_ref` 指针携带。
  - **导出投影**：ANF 落盘的 `record_id` 字段由导出适配器机械投影为 pattern 合法形式
    `exp-er-<decision_id>-<status>`（映射规则：`er-` → `exp-er-`，`@` → `-`）。
    前置条件：`decision_id` 字符集限定 `[A-Za-z0-9._-]+`（由 DecisionTrace 分配规则保证，
    SPEC-001 为权威）。状态取封闭词表且互非前缀（draft / verified / consolidated），故导出
    形式按前缀 `exp-er-<decision_id>-` 取最新与规范 id 语义一致。
- **影响**：OQ2 的版本化重导出与消费规则不变；变化仅在 ANF 落盘字段使用投影形式，
  规范 id 由 context 指针承载。导出适配器必须同时输出两层，并保证投影可逆。
