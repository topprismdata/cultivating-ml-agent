# GATE-STATUS — 治理门禁状态（G0–G5）

**分支**：`upgrade/p5-release-docs` · **生成日期**：2026-10-09 · **计划基线**：v0.1 §10

对应计划 v0.1 §10 的六道门禁逐门登记：状态、证据（文件 / 测试 / 命令路径）、缺口。
本文件只陈述事实，不是门禁本身；每道门的执行体是所列测试与代码路径。

**状态词汇**：

| 状态 | 含义 |
| --- | --- |
| `enforced` | 机器强制：违规输入在测试/代码路径被拒绝，CI（`.github/workflows/ci.yml`，`python -m pytest tests/ -q`）覆盖 |
| `manual` | 人工步骤：依赖人执行并留痕，无机器把守 |
| `partial` | 混合：部分机器强制，部分人工；或落地中（标注转入 enforced 的条件） |

**铁律（计划 v0.1 §10，适用于全部门禁）**：

- 阈值必须**预注册**，禁止事后为通过门禁修改口径（G4 策略文件一经提交即锁定，修改需新版本 policy，supersedes 语义）。
- G4 禁止一次成功自动晋升：激活永远是人/权限（A2）的动作，接口只产出 draft 与建议（ANF policy-record authority 约束 "Not promotion"）。

---

## G0 — 外部依赖 BOM 与 schema 一致性 — `partial`

**定义**：全部外部 Profile / 依赖带 tag/commit 与锁定日期；同一语义概念不得出现互斥定义。

**证据**：

- BOM：`docs/ontology/compatibility.bom.yaml`（v1.0.0，locked_at 2026-10-09；prism-ontology `9d93a99`、agent-nurture-framework `33f444b`、skill-tester `c5022b8`、VisitIR/VisitModel/Territory-IR 各自锁定；本地领域路径以 `untagged-local` 如实标注，不伪装成锁定项）
- schema 一致性：`docs/ontology/ml-profile/0.1.0/`（concepts.yaml / mappings.yaml / context.jsonld 与 outlet-insight 样板骨架对齐；ANF 4 schema 经 jsonschema 适配器校验）

**机器强制部分**：ANF evidence-envelope/experience-record 导出适配器的 schema 校验（`tests/test_adapters_pjp.py::test_schema_examples_valid_and_identity_consistent`、`tests/test_replay_reproducibility.py::test_evidence_anf_required_fields`）。

**人工部分**：BOM 锁定值与外部仓库实际 HEAD 的周期性比对为人工复核；ML Profile YAML 结构自检为提交前手动解析（`ml-profile/0.1.0/README.md` §校验现状）。

**缺口**：无自动化 CI 步骤在每次提交时重新核验 BOM ref ↔ 外部仓库实际 ref；pyshacl 语义级校验未接入（仅 `ml.shacl.ttl` 语法级目检），待 prism-ontology 治理 CI 接入。

**命令**：`python -m pytest tests/test_adapters_pjp.py tests/test_replay_reproducibility.py -q`

---

## G1 — ExperimentIR 六硬门 + 拒载 — `enforced`

**定义**：ExperimentIR 契约经 schema 校验 + content_hash 校验 + 六道硬门（固定次序）；任一 fail 即拒绝执行，篡改文档一律拒载。

**证据**：

- 实现：`framework/src/ir/experiment_ir.py`（`_GATE_ORDER`：G1_TEMPORAL_AVAILABILITY → G2_SPLIT_SEPARATION → G3_METRIC_DEFINITION → G4_BUDGET_BOUNDS → G5_BASELINE_SAME_PROTOCOL → G6_RUN_AUTHORIZATION；`load_ir` 对 `IRSchemaError`/`ContentHashMismatch` 拒载；`run_gates`/聚合 verdict）
- 测试：`tests/test_experiment_ir.py`（`test_invalid_sample_is_schema_legal_but_gate_blocked`、`test_tampered_body_rejected`、`test_tampered_hash_field_rejected`、`test_unknown_top_level_field_rejected`、`test_supersedes_chain`、`test_runtime_module_imports_no_mlflow_or_sklearn`）

**命令**：`python -m pytest tests/test_experiment_ir.py -q`（CI 覆盖）

**缺口**：无。规格见 `docs/specs/SPEC-001-experiment-ir.md`。

---

## G2 — 确定性复演（双跑同哈希 + golden 回归） — `enforced`

**定义**：同一 IR + 同一数据指纹双跑产出逐字节一致结果；golden 样例回归锁定输出形状。

**证据**：

- 实现：`framework/src/ir/runner.py`（确定性执行；被 IR 硬门阻断的 IR 拒绝执行）
- 测试：`tests/test_replay_reproducibility.py`（`test_double_run_identical`、`test_golden_regression`、`test_blocked_ir_refused_execution`、`test_fixture_tamper_raises_data_hash_mismatch`、`test_ir_content_hash_tamper_rejected`、`test_runner_imports_no_mlflow`）

**命令**：`python -m pytest tests/test_replay_reproducibility.py -q`（CI 覆盖）

**缺口**：无。规格见 `docs/specs/SPEC-003-replay-and-evidence.md`。

---

## G3 — DecisionTrace 状态机 + 四眼 + 负例拒绝 — `enforced`

**定义**：决策只允许合法状态转移；禁止自批（四眼原则）；劣于基线/低于阈值的候选全量拒绝；追加式防篡改链。

**证据**：

- 实现：`framework/src/ir/decision_trace.py`（状态机、追加式 hash 链、防篡改载入）、`framework/src/ir/decide.py`（胜出契约、负例拒绝、family 一致性）
- 测试：`tests/test_decision_trace.py`（`test_illegal_transitions_rejected`、`test_self_approval_rejected_four_eyes`、`test_tampered_stored_line_rejected_on_load`、`test_load_detects_broken_forward_link`）、`tests/test_decide.py`（`test_worse_than_baseline_rejects_everything`、`test_below_min_improvement_threshold`、`test_gate_fail_marks_all_candidates_constraint_gate_failed`、`test_family_mismatch_raises`）

**命令**：`python -m pytest tests/test_decision_trace.py tests/test_decide.py -q`（CI 覆盖）

**缺口**：无。规格见 `docs/specs/SPEC-004-decision-trace.md`、`docs/specs/SPEC-005-candidate-decision-and-mlflow.md`、边界决议 `docs/adr/ADR-002-decision-trace-boundary.md`。

---

## G4 — 晋级门禁（负例 + 四眼 + skill gate） — `partial`（本轮 A 线落地）

**定义**：能力候选晋级必须满足预注册阈值（min_task_success>=3、min_task_families>=2、require_negative_case=true、min_negative_transfer>=1、max_evidence_age_days=180、require_no_dispute=true、activation_authority="A2"）；激活是授权方人为动作，接口只产出 draft 与建议，一次成功不自动晋升。

**证据（A 线交付物，分支 `upgrade/p5-governance`，合入本仓库后路径生效）**：

- 策略 schema：`schemas/governance/promotion-policy.schema.json`（阈值预注册载体）
- 默认策略：`framework/src/governance/policies/default-policy.json`（提交即锁定；修改需新版本 policy，supersedes 语义）
- 测试：`tests/test_governance_*.py`（负例要求、四眼、skill gate、禁止自动晋升）
- 规格：`docs/specs/SPEC-007-governance-gates.md`

**现状**：本仓库 main 线尚无 G4 执行体；A 线合入后翻转为 `enforced`（届时 CI 的 `pytest tests/` 自动覆盖 `tests/test_governance_*.py`）。

**缺口**：A 线未合入前，晋级判定只能人工按 ANF capability-record / policy-record 逐条核对（不产生机器裁决记录）；合入后本条目更新为 enforced 并回填测试名。

---

## G5 — 跨域估计交换（四列分列 + 真求解器往返） — `enforced`

**定义**：跨域报告四列严格分列（不混列、空格渲染 NA 而非 0）；适配器零求解器 import，以 JSON 信封为界；真求解器往返真验（环境守卫）。

**证据**：

- 实现：`framework/src/adapters/report.py`（四列分列 + 确定性输出）、`framework/src/adapters/pjp.py`、`framework/src/adapters/estimates.py`（信封导出、量化规则、sigma 基准钉死）
- 测试：`tests/test_cross_domain_report.py`（`test_report_has_exactly_four_columns`、`test_columns_do_not_mix`、`test_null_cells_render_as_na_not_zero`、`test_report_deterministic_same_input_same_output`）；`tests/test_adapters_pjp.py`（`test_adapters_never_import_solvers`、`test_sigma_basis_pinned_one_based`、`TestRealSolverRoundTrip`——`VISITMODEL_PATH` 守卫）
- 规格：`docs/specs/SPEC-006-cross-domain-estimate-exchange.md`

**命令**：`python -m pytest tests/test_cross_domain_report.py tests/test_adapters_pjp.py -q`（CI 覆盖；真验部分 CI 无 `VISITMODEL_PATH` 自动 skip）

**缺口**：真求解器往返真验依赖本机领域仓 checkout（BOM `local_domain_paths`：VisitModel `/Users/ghb/VisitModel`，untagged-local，`VISITMODEL_PATH` 守卫）；引擎/领域仓发布可锁 ref 后须走新 BOM 版本转正，届时真验方可入 CI。

---

## 汇总表

| 门禁 | 状态 | 一句话证据 |
| --- | --- | --- |
| G0 | `partial` | BOM 全锁定 + untagged-local 如实标注；BOM↔外部 ref 复核与 pyshacl 仍人工/缺位 |
| G1 | `enforced` | 六硬门 + `load_ir` 拒载；`tests/test_experiment_ir.py` |
| G2 | `enforced` | 双跑同哈希 + golden 回归；`tests/test_replay_reproducibility.py` |
| G3 | `enforced` | 状态机 + 四眼 + 负例拒绝；`tests/test_decision_trace.py` + `tests/test_decide.py` |
| G4 | `partial` | 预注册阈值策略已定稿；A 线 `upgrade/p5-governance` 合入后转 enforced |
| G5 | `enforced` | 四列分列 + 零求解器 import；真验需 `VISITMODEL_PATH`（CI 自动 skip） |
