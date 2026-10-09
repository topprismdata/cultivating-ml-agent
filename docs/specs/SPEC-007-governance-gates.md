# SPEC-007: 治理双线门禁 GOV-001 晋级接口 + Skill Tester 门禁 + 决策账本（0.1.0）

状态：已实现
版本：0.1.0
基线：origin/main @ 7fd22ec（SPEC-001~006 已合入）
实现：`schemas/governance/promotion-policy.schema.json`、`framework/src/governance/`（promotion / skill_gate / ledger / errors / policies/default-policy.json）
测试：`tests/test_governance_promotion.py`、`tests/test_governance_skill_gate.py`
关联：SPEC-003（evidence 形状）、SPEC-004（DecisionTrace 与链哈希家法）、GOV-001、ADR-002（ANF 投影）、ADR-001（authority 平面，"Not promotion"）、计划基线 v0.1 §9/§10/§12、G4 门禁

## 1. 目标与铁律

P5 治理线交付一件事：**接口只产出 draft 与建议，激活永远是人/权限的动作**。评估可以自动化，晋升不能自动化。

铁律（计划 §12，G4）：

1. **禁止一次成功自动晋升**：`evaluate_promotion` 全部阈值通过后产出的是 ANF capability-record 且 `lifecycle_status` 恒为 `"draft"`（代码字面量，非参数），决策状态为 `approved_for_draft`——批准的是"可以起草"，不是"已激活"。
2. **阈值必须预注册**（计划 §10）：所有阈值来自哈希钉死的策略文件，代码里不存在可漂移的默认口径；策略一经提交即锁定，改阈值 = 新版本 policy + `supersedes` 语义，禁止事后为通过门禁修改口径。
3. **激活无代码路径**：`activate_capability` 函数存在但恒抛 `GovernanceError("激活需 A2+ 人工授权，接口不自动晋升")`——给它一个可 grep 的明确拒绝点，而不是静默什么都不做。
4. **Skill Tester 负例缺失一票否决**：负例 ≥ 1 是 G4 在技能侧的投影，负例缺失 → fail，且不给结晶建议。
5. **结晶仅 suggest**：门禁通过后的结晶建议只是决策对象里的一个字段，接口绝不写 `skills/`。

## 2. 晋级策略与预注册（schemas/governance + policies/default-policy.json）

schema draft 2020-12，`additionalProperties: false`。默认策略 `pol-promotion-default.v1`：

| 字段 | 预注册值 | 语义 |
|---|---|---|
| `thresholds.min_task_success` | **3** | task_success 证据最少条数 |
| `thresholds.min_task_families` | **2** | task_success 证据去重 task_family 最少族数 |
| `thresholds.require_negative_case` | **true** | 负例缺失即拒（G4） |
| `thresholds.min_negative_transfer` | **1** | negative_transfer 证据最少条数 |
| `thresholds.max_evidence_age_days` | **180** | 证据保鲜期（天），超龄即 stale |
| `thresholds.require_no_dispute` | **true** | 未解决 dispute/contradiction 即拒 |
| `activation_authority` | **A2** | draft→active 所需最低 ANF 权威级别（仅记录，不执行） |

锁定机制：

- `policy_content_hash` = `sha256(canonical(policy − {policy_content_hash, locked_at}))`（canonical 与 `ir.decision_trace.canonical_bytes` 同一实现：sorted keys、无冗余空白、UTF-8）。加载时重算，失配即 `PolicyError`——**policy 哈希篡改拒载**。
- `locked_at` 是易失字段，永不入哈希（镜像 DecisionTrace `created_at` 纪律）：钉住的是阈值口径，不是锁定的时刻。
- `policy_id` 含版本后缀（`pol-promotion-default.v1`），字符集与 ANF policy-record 的 `^pol-[A-Za-z0-9._-]+$` 一致，因此可直接落 `capability-record.authority_constraints`。
- `supersedes`：首版为 null；换阈值 = 新 policy_id 新版本 + supersedes 指向旧版，旧文件永不改写。

## 3. GOV-001 晋级接口（promotion.py）

### 3.1 决策流：evidence → decide → draft（激活留人工）

```
evidence_list (ANF evidence-envelope 形状)
        │  逐条校验：evidence_id/evidence_type/timestamp 等
        ▼
evaluate_promotion(capability_id, evidence_list, policy,
                   risk_level=…, scope=…, now=…)     ← now 可注入，默认墙钟但永不入哈希
        │  逐项评估预注册阈值（全部评估，逐条 reason）
        ├── 任一不满足 → PromotionDecision(status="rejected",
        │                               reasons=[{reason ∈ 固定枚举, detail}, …])
        ▼
全部满足 → PromotionDecision(status="approved_for_draft")
        + draft_capability_record（ANF capability-record，lifecycle_status="draft"）
        │
        ▼
activate_capability(...)  ──► 恒抛 GovernanceError（A2+ 人工授权，接口外）
```

### 3.2 阈值 → reason 固定枚举（逐条上报）

| 阈值检查 | 失败 reason（封闭枚举 `PROMOTION_REASONS`） |
|---|---|
| task_success 计数 ≥ 3 | `insufficient_task_success` |
| 去重 task_family ≥ 2 | `insufficient_task_families` |
| 负例缺失（require=true 且 negative_transfer=0） | `missing_negative_case` |
| negative_transfer < 1（非缺失场景） | `insufficient_negative_transfer` |
| 任一证据超龄 > 180d | `stale_evidence` |
| 未解决 dispute | `unresolved_dispute` |
| 未解决 contradiction | `unresolved_contradiction` |

纪律：一次评估报告**所有**未达标项（逐条），不是首个失败即停；reason 词表封闭，detail 只含计数与证据 id——**禁止把计算出的证据年龄写进 detail**（那会把墙钟 smuggle 进 decision_id 哈希）。dispute/contradiction 证据带 `resolved: true` 视为已解决（envelope additionalProperties 通道）。

### 3.3 输入输出契约

- 证据 dict 用 ANF evidence-envelope 字段名（`evidence_type`/`timestamp`/`metric_name`/`task_family`/`evidence_id`…）；本接口只强制它消费的字段，所以真实 ANF envelope 原样可通过（测试用真实上游 schema 验证 fixture）。
- `risk_level`/`scope` 由调用方声明（ANF 枚举 low/medium/high；personal/project/team/organization）。
- 通过路径产出的 capability-record：`evidence_ids` 为去重排序的输入证据 id；`provenance_chain_id = "prov-" + sha256(canonical(sorted(evidence_ids)))[:16]`（从证据链派生，证据变则链变）；`freshness.last_verified` 取最新证据；`authority_constraints = [policy_id, "<policy_id>.activation_floor_<authority>"]`——**authority 约束随记录走**。
- `decision_id = "gvd-" + sha256(canonical(决策内容 − {decision_id, evaluated_at}))[:12]`：同内容同 id（时钟平移不换 id，测试钉死），`evaluated_at` 是易失审计字段。
- `draft_experience_projection(projection)`：校验 P3 `ir.decision_trace.project_to_anf` 输出（ANF 九必填字段 + `outcome.result` 封闭词表），再按**固定声明映射**转录为 evidence 源 dict（`success→task_success`、`failure→task_failure`、`partial→correction`；`evidence_id`/`provenance_chain_id` 均为 sha256 派生）。机械转录，不做二次语义加工；`task_family` 不在 P3 投影里，由调用点补齐。

## 4. Skill Tester 门禁（skill_gate.py）

`evaluate_skill_gate(skill_id, tester_report, policy) -> GateDecision`。`tester_report` 是 skill-tester run 输出 dict：`tests` 数组，每项含 `case_id`（非空）、`case_type ∈ {positive, negative}`、`status ∈ {pass, fail}`——形状违约是 `GovernanceError`（契约错误），与"规则不满足 → fail"严格分列。

规则（固定，非逐次可调）：

| 规则 | 失败 reason | 来源 |
|---|---|---|
| 负例 ≥ 1 | `missing_negative_case` | G4；复用预注册阈值 `require_negative_case` |
| 无失败用例 | `failed_case_present`（detail 列失败 case_id） | 质量底线 |
| skill_id 非空 | `empty_skill_id` | 身份底线 |

GateDecision：`status ∈ {pass, fail}`、`reasons`（封闭词表 `SKILL_GATE_REASONS`）、`counts`、`report_digest = sha256(canonical(tester_report))`（报告指纹，审计可回指）、`policy_id`+`policy_content_hash`（门禁运行在哪个口径下）、`crystallization`（仅 suggest：`suggested` 仅在 pass 且 skill_id 非空时为 true，note 明示"不自动写文件"；测试钉死调用后 cwd 无 skills/ 写入）。`decision_id = "gvg-" + 12hex`，门禁决策零易失字段，完全确定。

## 5. 决策账本（ledger.py）

复用 P3 `ir.decision_trace` 链哈希家法，存储根 `outputs/governance/`（`DEFAULT_LEDGER_DIR`）：

- 每个决策一个 `<decision_id>.jsonl`，一行一个 canonical JSON 事件；首事件 `prev_event_id="genesis"`。
- `event_id = "gev-" + sha256(canonical(event − {event_id, recorded_at}))`：`recorded_at` 是易失审计字段，不入哈希（镜像 `created_at`）。
- `append_decision`：写前重验决策身份；已存在文件逐行字节比对，既有记录被改动 → `GovernanceLedgerConflictError`（append-only，存储历史永不改写；同内容重复 append 幂等 no-op）。
- `load_decision`：重算每个 `event_id`、校验 prev 链、重验内嵌决策的 sha256 身份——任何一层失配 → `GovernanceLedgerTamperError`（**篡改拒绝**；只改 `recorded_at` 不断链，这是设计而非漏洞，测试如实钉死）。

## 6. G4 映射表

| G4 条款 | 实现点 | 测试 |
|---|---|---|
| 禁止一次成功自动晋升 | `lifecycle_status` 字面量 `"draft"`；无 draft→active 代码路径 | `test_full_pass_produces_draft_not_active` |
| 激活是人/权限动作 | `activate_capability` 恒抛（含传 actor/authorization_ref、传 `{}`） | `test_activate_capability_always_raises_g4` |
| 阈值预注册禁事后改口径 | policy 哈希钉死 + 加载重算 | `test_threshold_change_changes_hash`、`test_tampered_policy_rejected_on_load` |
| 负例缺失一票否决（晋级侧） | `missing_negative_case` reason | `test_missing_negative_case_rejected_g4` |
| 负例缺失一票否决（技能侧） | gate `missing_negative_case` → fail 且不给结晶建议 | `test_missing_negative_case_fails_g4` |
| 结晶不自动写文件 | `crystallization` 仅 suggest 字段 | `test_no_side_effects_on_skills_dir` |

## 7. 与 GOV-001 / ADR-002 的关系

- **GOV-001** 是本规格实现的门禁本身：晋级评估（§3）+ 技能门禁（§4）是其两条入口；账本（§5）为其提供可审计的决策留痕。
- **ADR-002**（ANF 投影 + 指针）：P3 的 `project_to_anf` 产 experience-record（经验平面），本规格的 `draft_experience_projection` 机械复用它作为证据源；产出的 capability-record 是能力平面记录，两平面只经 evidence-envelope 形状的字段对接。authority 平面按 ADR-001 只读：接口消费 `activation_authority` 并写进 `authority_constraints`，**从不代表任何 authority 做动作**——capability-record 上的 `authority_constraints` 字段语义是"约束它何时能跑"，不是"晋升记录"（"Not promotion"）。
- 与 SPEC-004 的关系：链哈希、canonical bytes、易失字段排除、append-only 拒改写均沿用同一实现（`ir.decision_trace.canonical_bytes` 直接复用），不建立第二套哈希家法。

## 8. 依赖与测试分层

- 零新 pip 依赖：运行时纯 stdlib（`dataclasses`/`hashlib`/`json`/`re`/`pathlib`/`datetime`）；jsonschema 仅测试路径（`importorskip`）。
- 零 uuid4、零墙钟入哈希：身份一律 sha256 派生；`evaluated_at`/`recorded_at`/`locked_at` 三个易失字段全部有"不入哈希"测试。
- 真实 ANF schema 验证：capability-record 投影与 evidence fixture 用 `/tmp/anf-skilltester-check` 的上游 schema 验证（不可得则 clone，再不可得按 P3 模式 skip 并说明）。
- 架构守卫同 P4 家法（`test_governance_never_imports_heavy_or_forbidden_modules`）：正则扫描 `framework/src/governance/**/*.py` 的 import 语句，命中 `mlflow|sklearn|ortools|numpy|pandas|torch` 即红——governance 运行时纯 stdlib（内部只依赖同为纯 stdlib 的 `ir.decision_trace`）。
