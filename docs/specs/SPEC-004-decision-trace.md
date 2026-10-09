# SPEC-004: 决策轨迹层（0.1.0）

状态：已实现
版本：0.1.0
基线：origin/main @ e756797（SPEC-003 已合入）
决议：`docs/adr/ADR-002-decision-trace-boundary.md`（已接受）
实现：`framework/src/ir/decision_trace.py`、`framework/src/ir/__init__.py`（静态导出）
Schema：`schemas/decision-trace/0.1.0/decision-trace.schema.json`（draft 2020-12，
`additionalProperties: false`）+ `examples/`（全生命周期 / rejected / correction）
测试：`tests/test_decision_trace.py`

## 1. 目标与不变量

ExperimentIR（SPEC-001）把实验编译为不可变契约，重放与证据层（SPEC-003）把执行
固化为可复核证据；本规范定义其上的**决策轨迹层**：候选、依据、状态迁移全程簿记，
并把完整保真轨迹**有损投影 + 指针**输出为 ANF experience-record（ADR-002）。

不变量：

1. **不可变事件记录**：决策是事件链，追加式（append-only）；任何修正通过新事件
   链接旧事件（`correction_of`），禁止直接覆盖历史判断。
2. **确定性标识**：`decision_id` / `event_id` 一律 sha256 派生；禁用 uuid4。
   `created_at` 不进入任何哈希——注入时钟、重放均不改变身份。
3. **纯 stdlib**：`ir.decision_trace` 不 import mlflow/sklearn/jsonschema；
   jsonschema 仅用于 schema 文件与测试。
4. **文档 = 事件链的折叠**：文档字段由首事件 payload 快照 + 后续事件提升
   （`executed`→`execution_ref`、`evaluated`→`outcome_refs`）+ 末事件 status
   机械重建，内存文档与 `load_decision` 重建文档逐字段相等（有测试断言）。

## 2. 字段表

### 2.1 文档级（DecisionTrace）

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `schema_version` | const `"0.1.0"` | ✓ | 契约版本；语义升级须新建 schema 目录 |
| `decision_id` | string `^dt-[0-9a-f]{12}$` | ✓ | `dt-` + sha256(canonical{intent_ref, selected_option, alternatives, baseline_ref}) 前 12 hex；内容定，时钟无关 |
| `intent_ref` | string | ✓ | 决策服务的意图引用 |
| `state_snapshot_ref` | object | ✓ | `{ir_content_hash, data_sha256, code_version}`，三者均非空字符串 |
| `alternatives` | array | ✓ | 候选项 `{candidate_id, family, params(JSON 可序列化对象), metric_value}` |
| `constraint_check` | array | ✓ | 硬门禁结果 `{gate_id, status}`，gate_id 取 G1–G6 词表（§8） |
| `baseline_ref` | object | ✓ | `{baseline_id, protocol_ref(object), score(number)}` |
| `selected_option` | object | ✓ | 同候选项结构 |
| `rejected_options_and_reasons` | array | ✓ | `{candidate_id, reason, detail}`；reason 封闭词表 `worse_than_baseline \| below_min_improvement \| lost_to_selected \| constraint_gate_failed` |
| `evidence_refs` | array[string] | ✓ | 提案证据指针（SPEC-003 evidence-envelope） |
| `decision_actor` | string | ✓ | 提案人；批准人不得与其相同（§4） |
| `authorization_ref` | string | ✓ | 授权记录指针（对应 ExperimentIR G6 authorization） |
| `execution_ref` | string | — | 执行清单/轨迹指针；由 `executed` 事件提升 |
| `outcome_refs` | array[string] | — | 评估结果指针；由 `evaluated` 事件提升 |
| `uncertainty` | object | ✓ | `{metric_std(number\|null), margin(number\|null), note(string)}` |
| `created_at` | string(date-time) | ✓ | 首事件时间；**不入任何哈希** |
| `status` | enum | ✓ | `proposed \| approved \| executed \| evaluated \| rejected`；恒等于末事件 status |
| `events` | array, minItems 1 | ✓ | 追加式事件链 |

共享契约 `DecisionOutcome`（B 线产出 → A 线消费）中 `metric_name`、
`metric_direction` 在文档 schema（`additionalProperties: false`）无顶层槽位，
随 `record_from_decision_outcome` / `new_decision` 写入首事件 `payload`，
ANF 投影与 G3 映射从 payload 读取（§7、§8）。

### 2.2 事件级（traceEvent）

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `event_id` | string `^ev-[0-9a-f]{64}$` | ✓ | `ev-` + sha256(canonical(事件内容，含 `prev_event_id`)) 全长 hex；排除 `event_id` 本身与 `created_at` |
| `prev_event_id` | `^(genesis\|ev-[0-9a-f]{64})$` | ✓ | 首事件固定 `genesis`，否则前一事件 id |
| `event_type` | enum | ✓ | `proposed \| approved \| executed \| evaluated \| rejected \| correction` |
| `status` | enum | ✓ | 该事件后决策状态；correction 事件重复当前 status |
| `actor` | string | ✓ | 操作者 |
| `created_at` | string(date-time) | ✓ | 注入时钟；不入哈希 |
| `authorization_ref` | string | — | 如批准时引用的授权 |
| `execution_ref` | string | — | `executed` **必填**（schema if/then + 模块双重强制） |
| `evidence_refs` | array, minItems 1 | — | `evaluated` **必填且非空** |
| `outcome_refs` | array, minItems 1 | — | `evaluated` 可选；存在即提升到文档级 |
| `reason` | string | — | `rejected` / `correction` **必填** |
| `correction_of` | `^(genesis\|ev-[0-9a-f]{64})$` | — | `correction` **必填**：被更正事件 id |
| `payload` | object | — | 首事件携带文档快照（§1 不变量 4）；correction 携带更正内容 |

## 3. 状态机

转移表（`_TRANSITIONS`，非法转移抛 `IllegalTransition`）：

```
proposed ──▶ approved ──▶ executed ──▶ evaluated（终态）
   │            │
   └──▶ rejected ◀┘                    （终态）
```

| 当前 | 合法后继 |
| --- | --- |
| proposed | approved, rejected |
| approved | executed, rejected |
| executed | evaluated |
| evaluated | （终态，无后继） |
| rejected | （终态，无后继） |

状态转移权限：只有 `append_event(decision, event_type, *, actor, ...)` 能推进
状态，且 `event_type` 即目标状态；`correction` 不在转移表内，必须经
`correct()`（不改 status）；`append_event(…, "correction")` 直接抛
`IllegalTransition`。终态后任何 `append_event` 均拒绝。

## 4. 职责分离规则

- **四眼原则**：`approved` 事件的 `actor` 不得等于 `decision_actor`
  （提案人不得自行批准），违者抛 `SeparationOfDutiesError`。
- **强制随附**：`executed` 必带 `execution_ref`；`evaluated` 必带非空
  `evidence_refs`；`rejected` 必带 `reason`。缺失抛 `MissingEventFieldError`。
- **未知字段拒绝**：事件仅允许 schema 列出的扩展字段，其余 kwargs 抛 `TypeError`。

## 5. 不可变与修正语义

- **链哈希**：`event_id = ev-<sha256(canonical(事件内容含 prev_event_id))>`；
  每次 `append_event` / `correct` / `new_decision` 后全链重算并校验前向链接；
  `load_decision` 逐事件重验（防篡改）。
- **存储**：`save_decision(decision, decisions_dir)` 写
  `<decision_id>.jsonl`，每行一事件（canonical JSON）。append-only：文件已存在
  时逐行与内存事件比对，既有行被改动或文件超前于内存文档即抛 `LedgerWriteError`
  拒绝写入；仅追加缺失行。
- **修正**：`correct(decision, event_id, reason, *, actor, payload=None)` 追加
  `correction` 事件：`correction_of=event_id`、必带 `reason`、附新 `payload`；
  **不改 status、不改任何既有事件**（测试断言 `events[:-1]` 逐字节相等）。
  `correction_of` 指向不存在的事件抛 `UnknownEventError`。
- **篡改拦截**：手改账本任一行的实质字段（actor / execution_ref / payload /
  prev_event_id）后 `load_decision` 抛 `ChainHashMismatch`。`created_at` 被设计
  为不入哈希（时钟可注入），其改动不在拦截范围。

## 6. 标识与哈希纪律

- `decision_id = dt-<sha256(canonical{intent_ref, selected_option,
  alternatives, baseline_ref})[:12]>`：仅覆盖选择定义性内容，同一决策内容
  两次生成必得同 id（测试断言），不同内容必得不同 id。
- `event_id = ev-<sha256(canonical(事件内容))>`：含 `prev_event_id` 与全部
  业务字段；排除 `event_id`、`created_at`。零 uuid4、零墙钟入哈希（源码纯度
  测试 + 双时钟同 id 测试双重把守）。
- canonical 形式与 SPEC-001 一致：`json.dumps(sort_keys=True,
  separators=(",",":"), ensure_ascii=False).encode("utf-8")`。

## 7. 共享契约入口与 ANF 投影（ADR-002）

`record_from_decision_outcome(outcome, *, ir_content_hash, clock=None)` 是
`new_decision` 的别名入口：按共享契约字段逐一映射，缺字段抛 `KeyError` 并指名
（如 `'metric_name'`）；`ir_content_hash` 固化进 `state_snapshot_ref`，与既有
值冲突抛 `ValueError`。

按 ADR-002 决议（OQ1/OQ2/OQ3），`project_to_anf(decision)` 产出有损投影 +
指针；**不扩展 ANF schema**，完整保真唯一归属 DecisionTrace，消费方凭
`context` 指针回源，按前缀取最新：

| ANF 字段 | 投影值 | 依据 |
| --- | --- | --- |
| `record_id` | `exp-er-<decision_id>-<status>`（机械投影：`er-`→`exp-er-`、`@`→`-`；decision_id 字符集 `[a-z0-9._-]` 与 status 封闭词表保证匹配 ANF pattern `^exp-[A-Za-z0-9._-]+$`） | OQ3：ANF record_id 实测有 pattern、无 format，规范 id 不能直接落盘 |
| `context` | `decision_trace_ref=er-<decision_id>@<status>`（规范 id，append-only 台账与前缀取最新以此为准） | OQ1：指针方式回指，不扩 schema |
| `task` | `intent_ref` | ANF 必填 minLength 1 |
| `decision` | selected_option 摘要：`selected <candidate_id> [<family>] <metric_name>=<metric_value> (<direction>)` | OQ1：仅保留 decision 摘要 |
| `decision_summary` | rejected + uncertainty 聚合（`rejected: <id>(<reason>), …; uncertainty: metric_std=…, margin=…, note=…`），超 2000 字符截断 | ANF maxLength 2000 |
| `alternatives_considered` | `["<candidate_id> [<family>]", …]` | array[string] |
| `action` | `execute_experiment(<selected candidate_id>)` | — |
| `execution_trace_refs` | `[execution_ref]`（缺失为 `[]`） | 引用非全文 |
| `outcome` | `outcome_refs` 非空 → `{result: "success", metric_name, metric_value, notes}`；否则 `evidence_refs` 非空 → `{result: "partial", notes}`；否则 `{result: "failure"}` | OQ1：结果只留摘要；ANF `result` 枚举 success/failure/partial |
| `classification` | `"internal"` | 枚举词表 |
| `trust_level` | `executed`/`evaluated` → `verified_execution`；其余（proposed/approved/rejected）→ `agent_inference` | 枚举词表 |
| `timestamp` | `created_at`（ISO-8601 date-time） | — |

状态每前进一步即版本化重导出一条新记录（`record_id` 随 status 变化），
append-only，不覆盖旧记录；消费方以 `er-<decision_id>@` 为前缀取最新
（OQ2）。九个 ANF 必填字段（record_id、task、decision、decision_summary、
action、outcome、classification、trust_level、timestamp）投影后必须非空，
测试以 `re.fullmatch` 断言 pattern、以真实上游 schema
（agent-nurture-framework `schemas/experience-record.schema.json`，
`jsonschema.validate`）验证投影结果；schema 不可得时 `pytest.skip` 并注明。

## 8. G3 门禁映射

`constraint_check[*].gate_id` 取 `ir.experiment_ir` 的六个硬门禁 id；G3
（`G3_METRIC_DEFINITION`）在决策轨迹层的对应关系：

| G3 要素（SPEC-001） | DecisionTrace 落点 |
| --- | --- |
| 指标定义存在且 direction 合法 | 首事件 `payload.metric_name` / `payload.metric_direction`（共享契约必填，`metric_direction ∈ {maximize, minimize}`，违者 `TraceSchemaError`） |
| 门禁结论 | `constraint_check` 中 `{"gate_id": "G3_METRIC_DEFINITION", "status": …}` 行，随提案固化 |
| 指标不可比导致的淘汰 | `rejected_options_and_reasons[*].reason = "constraint_gate_failed"`（或 G5 失效场景 `"worse_than_baseline"`） |
| 评估证据 | `evaluated` 事件非空 `evidence_refs`（SPEC-003 evidence-envelope）；提升后 `outcome_refs` 指向评估产物 |
| ANF 投影 | `outcome.metric_name` 取自 payload `metric_name`；`outcome.metric_value` 取 `selected_option.metric_value`（仅 success） |

G1/G2/G4/G5/G6 门禁结论同样以 `gateCheck` 行随提案簿记；G6 授权对应
`authorization_ref`。门禁不通过不是 schema 违规（schema 只管形状），而是
决策层淘汰候选或拒绝提案的依据——与 SPEC-001「gate 级违规由
run_gates/authorize_execution 捕获」分层一致。

## 9. 依赖纪律

`ir.decision_trace` 纯 stdlib（hashlib/json/re/datetime/pathlib/typing）；
`ir/__init__.py` 静态导出其公开 API（`new_decision`、`append_event`、`correct`、
`save_decision`、`load_decision`、`record_from_decision_outcome`、
`project_to_anf`、`verify_decision` 与异常族）；runner 的懒导出机制不受影响，
`import ir.decision_trace` 保持零 numpy/sklearn。
