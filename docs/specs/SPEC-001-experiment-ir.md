# SPEC-001: ExperimentIR 实验契约（0.1.0）

状态：已实现
版本：0.1.0（契约）
基线：main @ 6e58f52
实现：`framework/src/ir/experiment_ir.py`、`schemas/experiment-ir/0.1.0/`
测试：`tests/test_experiment_ir.py`

## 1. 目标与不变量

ExperimentIR 是 ML 域的**实验契约**：一次实验在执行前必须先编译为一份
ExperimentIR 文档（JSON）。它是自然语言代理与执行环境之间唯一的交接界面，
遵循 VisitIR 家法：

- **纯函数、零求解器依赖**：运行时模块 `framework/src/ir/experiment_ir.py`
  只用 stdlib（`jsonschema` 可选安装；缺失时回退内置最小结构检查），
  禁止 import mlflow/sklearn，并有回归测试守护
  （`test_runtime_module_imports_no_mlflow_or_sklearn`）。
- **编译后不可被隐式修改**：文档一经编译即冻结（`content_hash` 锁定）。
  任何改动——无论来自代理还是人——都会导致 `load_ir` 抛出
  `ContentHashMismatch` 拒绝加载。修正的唯一合法路径是**产生新 revision**
  并以 `supersedes` 指向前版 `content_hash`。
- **只能证伪，不能证真的快筛分层**：`soft_preferences` 仅供代理权衡，
  永不参与门禁；`hard_constraints` 必须挂接可机器判定的 `check` 谓词。
- **任一硬门禁未满足必须拒绝执行**（计划 G1）：`authorize_execution`
  汇总裁决，任何 gate `fail` 即 blocked。

## 2. 加载协议（load_ir 流水线）

```
JSON 文件
  │
  ├─ ① schema 校验   jsonschema(draft 2020-12) 优先；
  │                  未安装 → _fallback_validate（内置最小结构检查）
  │                  失败 → IRSchemaError（schema 级非法，文档不可加载）
  │
  ├─ ② ISO-8601 严检  created_at / as_of / label_cutoff / granted_at
  │                  （draft 2020-12 的 format 仅是注解，不能依赖
  │                    jsonschema extras，故由 stdlib 统一裁决）
  │                  失败 → IRSchemaError
  │
  ├─ ③ content_hash   sha256(canonical_bytes(文档去掉 content_hash))
  │                  不符 → ContentHashMismatch（编译后被篡改，拒载）
  │
  └─ 返回 dict ──► run_gates(ir) ──► authorize_execution(ir)
                                      ok=False ⇒ 拒绝执行（计划 G1）
```

关键 API：

| 函数/类 | 语义 |
| --- | --- |
| `load_ir(path) -> dict` | ①②③ 全部通过后返回文档；否则抛 `IRSchemaError` / `ContentHashMismatch` |
| `canonical_bytes(ir) -> bytes` | 确定性 JSON 字节：`sort_keys=True`、分隔符 `","`/`":"`、UTF-8 |
| `compute_content_hash(ir) -> str` | sha256 hexdigest，**排除 `content_hash` 字段本身** |
| `run_gates(ir) -> list[GateResult]` | 固定顺序跑六门，每门返回 `{gate_id, status, reason}` |
| `authorize_execution(ir) -> AuthorizationVerdict` | `ok` / `failed_gates` / `reasons` / `gates`；`blocked == not ok` |
| `GateResult.status` | `pass` / `fail` / `not_applicable`（仅 fail 阻塞） |

异常层级：`IRError` ← `IRSchemaError`、`ContentHashMismatch`。

## 3. 字段表（schema_version = "0.1.0"）

### 3.1 顶层

schema 根与所有子对象均为 `additionalProperties: false`（未知字段一律拒绝）。

| 字段 | 类型 | 必填 | 约束 / 语义 |
| --- | --- | --- | --- |
| `schema_version` | string | ✔ | 常量 `"0.1.0"`，与 schema 目录一一对应 |
| `experiment_id` | string | ✔ | 非空；实验全局唯一标识 |
| `task_ref` | string | ✔ | 非空；任务/比赛定义引用 |
| `world_state_ref` | string | ✔ | 非空；编译时世界状态快照（既往实验、决策） |
| `objective` | string | ✔ | 非空；本实验要验证什么 |
| `dataset_snapshot_ref` | object | ✔ | 见 3.2 |
| `feature_set_ref` | string | ✔ | 非空；不可变特征集引用 |
| `validation_protocol_ref` | object | ✔ | 见 3.3 |
| `metric_definition_ref` | object | ✔ | 见 3.4 |
| `baseline_ref` | object | ✔ | 见 3.5 |
| `hypothesis_ref` | string | ✖ | 非空；被检验假设的引用（可选） |
| `candidates` | array | ✔ | 元素见 3.6 |
| `hard_constraints` | array | ✔ | 元素见 3.7；可为空数组 |
| `soft_preferences` | array | ✖ | 可选；代理偏好，永不门禁 |
| `budget` | object | ✔ | 见 3.8 |
| `expected_artifacts` | array | ✔ | 非空字符串数组；执行必须产出的工件 |
| `authorization` | object | ✔ | 见 3.9 |
| `source_ref` | string | ✔ | 非空；溯源：谁/什么从哪里编译出本文档 |
| `created_at` | string | ✔ | ISO-8601 编译时间戳（加载时严检） |
| `content_hash` | string | ✔ | `^[0-9a-f]{64}$`；见 §5 版本纪律 |
| `revision` | integer | ✔ | ≥1；文档修订号 |
| `supersedes` | string | ✖ | `^[0-9a-f]{64}$`；被取代前版的 `content_hash`（可选，rev≥2 应填） |

### 3.2 `dataset_snapshot_ref`

| 字段 | 类型 | 必填 | 约束 / 语义 |
| --- | --- | --- | --- |
| `uri` | string | ✔ | 非空；不可变数据快照地址 |
| `sha256` | string | ✔ | `^[0-9a-f]{64}$`；数据快照摘要 |
| `rows` | integer | ✔ | ≥0 |
| `as_of` | string | ✔ | ISO-8601；数据截止时间 |
| `label_cutoff` | string | ✖ | ISO-8601；`strategy=time_based` 时由 G1 要求存在且 ≤ `as_of` |

### 3.3 `validation_protocol_ref`

| 字段 | 类型 | 必填 | 约束 / 语义 |
| --- | --- | --- | --- |
| `strategy` | enum | ✔ | `stratified` \| `kfold` \| `time_based` \| `group`（封闭词表，schema 级强制；G2 防御性复检） |
| `n_folds` | integer | ✔ | ≥1；`time_based` 为单折前向切分，此字段不参与切分但须 ≥1 |
| `time_col` | string | ✖ | `time_based` 时 G1 要求非空 |
| `val_size_weeks` | integer | ✖ | `time_based` 时 G2 要求 >0 |
| `group_col` | string | ✖ | `group` 时 G2 要求非空 |

### 3.4 `metric_definition_ref`

| 字段 | 类型 | 必填 | 约束 / 语义 |
| --- | --- | --- | --- |
| `name` | string | ✔ | 非空（schema minLength 1，G3 防御性复检） |
| `direction` | string | ✖* | *合法值 `maximize` \| `minimize` **由 G3 门禁强制，schema 刻意放开为任意字符串**。这样非法方向属于"门禁级非法"（加载成功 → 拒绝执行 → 新 revision 修正），而不是 schema 级非法（无法加载、无法追溯）。这是对计划 §4 中"`direction: maximize|minimize`"字样的有意豁免，动机见 §7。 |

### 3.5 `baseline_ref`

| 字段 | 类型 | 必填 | 约束 / 语义 |
| --- | --- | --- | --- |
| `baseline_id` | string | ✔ | 非空 |
| `protocol_ref` | object | ✔ | 与 3.3 同构；**必须与顶层 `validation_protocol_ref` 全等（G5）** |
| `score` | number | ✔ | 基线在 `protocol_ref` 口径下的得分 |

### 3.6 `candidates[]`

| 字段 | 类型 | 必填 | 约束 |
| --- | --- | --- | --- |
| `candidate_id` | string | ✔ | 非空 |
| `family` | string | ✔ | 非空；模型族（lightgbm/xgboost/…） |
| `params` | object | ✔ | 模型参数，结构开放 |

### 3.7 `hard_constraints[]`

| 字段 | 类型 | 必填 | 约束 |
| --- | --- | --- | --- |
| `type` | string | ✔ | 非空（如 `leakage`） |
| `description` | string | ✔ | 非空；人读描述 |
| `check` | string | ✔ | 非空；可机器判定的谓词引用，由执行器评估 |

### 3.8 `budget`

| 字段 | 类型 | 必填 | 约束 |
| --- | --- | --- | --- |
| `max_train_hours` | number | ✔ | **正值由 G4 强制**（schema 只约束数值类型，非法样本保持门禁级） |
| `max_runs` | number | ✔ | 同上 |

### 3.9 `authorization`

| 字段 | 类型 | 必填 | 约束 |
| --- | --- | --- | --- |
| `authorized_by` | string | ✔ | 非空由 G6 强制（schema 允许空串，保持门禁级非法） |
| `authorization_ref` | string | ✔ | 同上 |
| `granted_at` | string | ✔ | ISO-8601（加载时严检） |

## 4. 六门语义

门以固定顺序执行：G1 → G2 → G3 → G4 → G5 → G6（`gate_id` 是契约的一部分，
不得重命名、重排）。

| gate_id | 检查内容 | pass | fail（阻塞） | not_applicable |
| --- | --- | --- | --- | --- |
| `G1_TEMPORAL_AVAILABILITY` | `strategy=time_based` 时：`time_col` 非空、`val_size_weeks` 存在、`label_cutoff` 存在（**缺即 fail**）且 ≤ `as_of`（可解析） | 时序要素齐备且无未来标签泄漏 | 缺任一要素 / `label_cutoff` 晚于 `as_of` / 时间戳不可解析 | **非时序策略**（stratified/kfold/group）一律不阻塞 |
| `G2_SPLIT_SEPARATION` | `strategy` ∈ 四策略；`time_based` → `val_size_weeks > 0`；`group` → `group_col` 非空 | 对应策略的分离要素合规 | 未知策略 / `val_size_weeks` ≤ 0 或缺失 / `group_col` 空 | —（四策略内总有明确结论） |
| `G3_METRIC_DEFINITION` | `name` 非空且 `direction` ∈ {maximize, minimize} | 指标定义合法 | 名称为空 / 方向非法（如 `maximise`） | — |
| `G4_BUDGET_BOUNDS` | `max_train_hours > 0` 且 `max_runs > 0` | 预算合规 | 任一 ≤ 0 或非数值 | — |
| `G5_BASELINE_SAME_PROTOCOL` | `baseline_ref.protocol_ref` 与顶层 `validation_protocol_ref` **深度全等** | 基线与候选可比 | 口径不一致（哪怕只差一个字段） | — |
| `G6_RUN_AUTHORIZATION` | `authorized_by` 与 `authorization_ref` 非空 | 有执行授权 | 缺授权人 / 授权引用 | — |

裁决规则：

- `run_gates` 返回全部六门结果；`authorize_execution` 汇总。
- **任一 `fail` ⇒ `AuthorizationVerdict(ok=False, …)`，执行被拒绝**（计划 G1）。
- `not_applicable` 永不阻塞（非时序策略在 G1 上天然豁免）。
- 门禁是**防御性自包含**的：即使 schema 已约束的字段（如 `strategy` 枚举、
  `name` 非空），门内仍复检——回退校验路径或未来调用方绕过 schema 时，
  门禁依然 fail-closed。

## 5. 版本纪律

1. **契约版本**：`schema_version` 常量 `"0.1.0"`，与 schema 目录
   `schemas/experiment-ir/<version>/` 一一对应。语义不兼容变更必须开新目录
   （如 `0.2.0/`）并新增 schema 文件，不得原地改旧版 schema。
2. **文档修订**：对已编译文档的任何修正 → `revision` 加一（≥2），
   `supersedes` 指向**前版的 `content_hash`**，重算并覆写自身
   `content_hash`。测试 `test_supersedes_chain` 验证了多级链
   rev1 → rev2 → rev3。
3. **完整性**：`content_hash = sha256(canonical_bytes(文档去掉 content_hash))`；
   `canonical_bytes` 为排序键紧凑 UTF-8 JSON，保证跨进程确定性
   （`test_canonical_bytes_deterministic`）。`load_ir` 重算不符即抛
   `ContentHashMismatch` 拒载——"隐式修改"在加载层面即被击穿。
4. **禁止事项**：禁止原地改 IR；禁止绕过 `authorize_execution` 直接执行；
   禁止把门禁失败"修"进加载器（如放宽 hash 校验）。

## 6. 与 pipeline.splits / config.ValidationConfig 的对齐

`validation_protocol_ref` 的词表与字段名必须与运行时切分实现 1:1 对齐，
代理不得发明运行时不认识的策略名。

| ExperimentIR | `config.ValidationConfig` | `pipeline.splits` | 备注 |
| --- | --- | --- | --- |
| `strategy: stratified\|kfold\|time_based\|group` | `strategy: str = "stratified"`（同值域） | `make_folds` 按同名常量分派四分支，未知值 `ValueError` | 值域封闭，两端一致 |
| `n_folds ≥ 1` | `n_folds: int = 5` | `stratified_folds` / `kfold_folds` / `group_folds` 的折数 | `time_based` 是**单折前向切分**，`n_folds` 不参与切分（schema 仍要求 ≥1，仅作占位） |
| `time_col` | `time_col: str = ""` | 执行器按列名物化 `time` 数组传入 `make_folds(time=…)` | IR 记录列名；物化是执行器职责 |
| `val_size_weeks > 0`（G2） | `val_size_weeks: int = 20` | `time_based_folds(time, val_size_weeks)`：以浮点周为单位的 cutoff，不变式 `max(train) < min(val)` | G1/G2 保证该参数存在且为正，切分才不退化 |
| `group_col` 非空（G2） | `group_col: str = ""` | 执行器按列名物化 `groups` 数组；`GroupKFold` 保证组隔离 | `splits.group_folds` 要求组数 ≥ n_folds |

对齐的两条推论：

- 通过 G1/G2 的协议，交给 `make_folds` 时不会再因缺 `time`/`groups`
  列或非正 `val_size_weeks` 而 `ValueError`——门禁在编译期拦截运行期故障。
- 未来给 `ValidationConfig` 或 `make_folds` 增删策略时，必须同步：
  schema `strategy` 枚举、G2 词表、本表（三处一起改，测试守护值域）。

## 7. 计划 G1 → 代码映射

计划 G1："ExperimentIR 编译后不得被自然语言代理隐式修改，变更必须产生新
版本；任一硬门禁未满足必须拒绝执行。"

| 计划条款 | 代码落点 |
| --- | --- |
| 编译后不可隐式修改 | `content_hash` + `load_ir` 的 `ContentHashMismatch`（§5.3） |
| 变更必须产生新版本 | `revision` + `supersedes` 纪律（§5.2，`test_supersedes_chain`） |
| 任一硬门禁未满足必须拒绝执行 | `run_gates` 六门 + `authorize_execution` blocked（§4） |
| 拒绝执行的出口 | 调用方检查 `verdict.ok`；`ok=False` 时以 `verdict.reasons`（`Gx_…: 原因`）拒绝启动训练并回报 |

## 8. 示例与测试的防漂移契约

`schemas/experiment-ir/0.1.0/examples/` 内的文档是测试的**唯一门禁级
夹具**（`tests/test_experiment_ir.py` 直接加载文件驱动）：

| 文件 | 期望 |
| --- | --- |
| `valid-minimal.json` | 过 schema；六门全 pass；`authorize_execution().ok == True` |
| `invalid-g1-time-col.json` | 过 schema；仅 `G1_TEMPORAL_AVAILABILITY` fail（time_based 缺 `time_col`） |
| `invalid-g3-direction.json` | 过 schema；仅 `G3_METRIC_DEFINITION` fail（`direction: "maximise"`） |
| `invalid-g4-budget.json` | 过 schema；仅 `G4_BUDGET_BOUNDS` fail（`max_train_hours: 0`） |
| `invalid-g5-baseline-protocol.json` | 过 schema；仅 `G5_BASELINE_SAME_PROTOCOL` fail（基线 `n_folds=5` ≠ 顶层 `1`） |
| `invalid-g6-authorization.json` | 过 schema；仅 `G6_RUN_AUTHORIZATION` fail（授权字段为空串） |

规则：

- **invalid 样本必须是门禁级非法，不是 schema 级非法**：它们通过 JSON
  schema、`content_hash` 与自身一致、`load_ir` 成功，只在门禁上被拒。
  因此每个非法样本的 `content_hash` 都按其（非法）内容独立计算——篡改
  检测与门禁检测是两个正交层次。
- 每个非法样本**恰好**触发一扇门（断言 `failed_gates` 等于单元素），
  防止样本之间缺陷互相污染。
- 示例与 schema 的同步由测试保证：任何一边改动导致样例不再过 schema、
  或不再触发目标门，测试立即红。修改示例必须同步修改对应断言。
