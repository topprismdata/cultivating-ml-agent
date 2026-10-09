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
- OQ2: 决策状态每前进一步，导出的经验记录如何版本化？消费方如何取最新？
- OQ3: ANF `experience-record.schema.json` 的 `record_id` 存在哪些约束？`er-` 前缀
  规范 id 能否直接写入该字段？
