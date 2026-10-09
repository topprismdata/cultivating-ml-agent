# SPEC-005: 候选比较与负例拒绝 + MLflow 追踪适配（0.1.0）

状态：已实现
版本：0.1.0
基线：origin/main @ e756797（SPEC-003 已合入）
实现：`framework/src/ir/decide.py`、`framework/src/pipeline/mlflow_adapter.py`
测试：`tests/test_decide.py`、`tests/test_mlflow_adapter.py`
关联：SPEC-001/003、ADR-002、ADAPTER-001、计划基线 v0.1 §5/§10/§12

## 1. 目标与不变量

P3 决策层双线的 B 线（本仓，分支 `upgrade/p3-decide-mlflow`）交付两件事：

1. **决策函数** `decide(ir, results)`：在授权 ExperimentIR 与逐候选执行
   结果之上做候选比较与负例拒绝，产出共享契约 **DecisionOutcome** dict
   （计划 §5/§12；边界由 ADR-002 钉死——完整保真归 A 线 DecisionTrace，
   本 dict 是 B 产出、A 消费的唯一接口）。
2. **追踪适配** `log_experiment_run(...)`：把单次执行（manifest + evidence
   + OOF 工件）落进 MLflow（ADAPTER-001）。

不变量：

1. **字段契约封闭**：输出恰为共享契约 13 字段
   （`DECISION_CONTRACT_FIELDS`），一字段不多不少；
   `decision_id` / `execution_ref` / `outcome_refs` / `created_at` 等
   DecisionTrace 事件簿记字段全部归 A 线，绝不出现在本层输出。
2. **门禁先行**：`run_gates(ir)` 任一 fail ⇒ 不做任何排名，全体候选标
   `constraint_gate_failed`，`selected_option=None`。
3. **阈值预注册**（计划 §10）：`min_improvement_over_baseline` 阈值只从
   IR 的 `soft_preferences` 读取（随 `content_hash` 冻结）；decide 绝不在
   决策时现定阈值。
4. **确定性**：胜者并列取 IR 候选序在前者；拒绝规则链固定顺序；无 set
   迭代序、无 uuid、无墙钟入决策。
5. **零 A 线依赖**：`decide.py` 运行时纯 stdlib（`RunResult` 仅
   `TYPE_CHECKING`，manifest 鸭子类型访问），不 import `decision_trace`；
   依赖仅限已并入 main 的 `experiment_ir` / `runner` 等模块。
6. **fail-closed 且如实输出**：结果键未知/缺失、候选与 manifest 的
   family 不一致、manifest 缺 IR 指标 ⇒ `ValueError`；胜者被拒不产生
   选中项，决策整体 rejected 的语义由上游（A 线状态机）处理，decide
   只如实输出事实。

## 2. 决策流程

```
decide(ir, results) -> dict          # results: {candidate_id: RunResult}
  │
  ├─ ① 输入校验   候选非空、candidate_id 无重复；结果键 ∈ IR 候选集
  │               （门禁通过后要求全覆盖；门禁失败路径允许无结果）
  │
  ├─ ② 硬门禁     run_gates(ir) 全量 6 门 → constraint_check 原样登记
  │               （pass/fail/not_applicable）；任一 fail ⇒ 短路：全体
  │               候选 constraint_gate_failed，selected=None，不排名
  │
  ├─ ③ 一致性     每候选 manifest 的 model_family 必须等于 IR 声明、
  │               metrics 必须含 IR 指标名，否则 ValueError
  │
  ├─ ④ 胜者       按 metric_definition_ref.direction 取最优值；并列取
  │               IR 序在前者（minimize 取最小，maximize 取最大）
  │
  ├─ ⑤ 负例拒绝   对每个候选（胜者同样过链，见 §3）按序判定
  │               worse_than_baseline / below_min_improvement /
  │               lost_to_selected
  │
  └─ ⑥ 组装       state_snapshot_ref 取胜者 manifest；uncertainty 按
                  §4 规则；evidence_refs 取各候选 oof_sha256
```

`execute_experiment` 只执行 `candidates[0]`（SPEC-003），因此双候选比较
= 每候选一次执行：第二候选以**旋转候选序的新 revision** 执行（候选序
重排 → `content_hash` 重算 + `revision` 递增，绝不原地改 IR）。

## 3. 拒绝规则表（含预注册阈值语义，计划 §10）

按序判定，命中即停：

| # | 条件 | reason | detail |
| --- | --- | --- | --- |
| 0 | `run_gates(ir)` 任一 fail | `constraint_gate_failed`（**全体候选**） | 失败 gate 的 `gate_id: reason` 拼接 |
| 1 | 按 direction **严格劣于** `baseline_ref.score`（minimize: value > baseline；maximize: value < baseline） | `worse_than_baseline` | 指标值 vs 基线值 + 方向 |
| 2 | 改进幅度 < 预注册阈值（改进幅度 = direction 下相对基线的带符号改进：minimize 为 `baseline−value`，maximize 为 `value−baseline`） | `below_min_improvement` | 改进量 vs 阈值 + 方向 |
| 3 | 其余败者 | `lost_to_selected` | 与胜者差距 `gap`（按方向为正）+ 双方值 |

规则链对**每个候选统一适用**（含胜者）：胜者若劣于基线或低于阈值，则
`selected_option=None`（最优者尚且不过线 ⇒ 无可选）；胜者过链即选中，
其余候选按链归类。

**预注册阈值语义**（计划 §10；G3/G4 保证指标契约与预算合法后才进入
比较）：

- 载体：`soft_preferences` 条目
  `{"type": "min_improvement_over_baseline", "threshold": <number>}`。
  schema（`schemas/experiment-ir/0.1.0/experiment-ir.schema.json`）对
  `soft_preferences` **不约束条目形状**（自由数组），故 decide 防御式
  读取：`type` 不匹配或 `threshold` 非有限数值的条目**忽略**（schema
  合法文档永不使决策崩溃）。
- 判定：改进幅度 **≥ 阈值达标，< 阈值拒绝**；与基线并列（改进 = 0）
  无阈值时归 `lost_to_selected`，有阈值时改进 0 < 阈值 ⇒
  `below_min_improvement`。
- 多条匹配条目取**最严**（最大阈值）——决策须同时满足全部预注册偏好。

## 4. DecisionOutcome 契约（与 A 线一致，字段所有权）

恰 13 字段，B 线产出 → A 线消费；A 线（`upgrade/p3-decision-trace`，
SPEC-004 / ADR-002）只消费本 dict，`decision_id`（sha256 派生）、status
状态机、`execution_ref` / `outcome_refs` / `created_at` 等事件簿记全部
归 A 线，不做二次推断：

| 字段 | 所有权/来源 |
| --- | --- |
| `intent_ref` | `ir.hypothesis_ref`，缺省回退 `experiment_id`（B 线如实投影） |
| `state_snapshot_ref` | 胜者 `manifest.canonical` 的 `ir_content_hash` / `data_sha256` / `code_version`；无胜者回退首个可用 manifest（IR 序），再回退 IR 自身声明 + `"unknown"` |
| `alternatives` | 各候选 `{candidate_id, family, params, metric_value}`（IR 序） |
| `constraint_check` | `run_gates` 全量 `{gate_id, status}`（pass/fail/not_applicable，固定 G1→G6 序） |
| `baseline_ref` | `ir.baseline_ref` 原样（`baseline_id` / `protocol_ref` / `score`） |
| `selected_option` | 胜者 `{candidate_id, family, params, metric_value}`；被拒为 `null` |
| `rejected_options_and_reasons` | `{candidate_id, reason, detail}`（§3 词表封闭） |
| `uncertainty` | `margin = |selected − baseline|`（未选中为 null）；`metric_std` = 胜者 `manifest.canonical.fold_metrics[metric_name]` 的样本标准差（≥2 折才计算，否则 null）；`note` 说明缺口 |
| `decision_actor` | `ir.authorization.authorized_by` |
| `authorization_ref` | `ir.authorization.authorization_ref` |
| `evidence_refs` | 各候选 `oof_sha256`（IR 序） |
| `metric_name` / `metric_direction` | `ir.metric_definition_ref` |

## 5. 计划 G3/G4 → 代码映射

| 计划条款 | decide 落点 |
| --- | --- |
| G3_METRIC_DEFINITION（指标名 + 方向合法） | 门禁先行（§2 ②）：direction 非法 ⇒ `constraint_gate_failed`、从不排名；方向合法后驱动胜者规则与基线/阈值比较；`metric_direction` 字段原样输出 |
| G4_BUDGET_BOUNDS（预算为正） | 门禁先行（§2 ②）：预算非法 ⇒ `constraint_gate_failed`；decide 自身不测耗时/成本，如实拒绝并留 detail |

## 6. MLflow 映射表（ADAPTER-001）

`log_experiment_run(run_result, evidence, ir, *, tracking_uri=None) -> run_id`：

| 来源 | MLflow 目标 | 说明 |
| --- | --- | --- |
| `ir["experiment_id"]` | experiment（`set_experiment` 按名）+ run_name | 一次执行 = 该实验下一条 run |
| `manifest["canonical"]`（拍平） | params | 嵌套映射键点连（`metrics.rmsle`）、列表按索引段（`fold_sizes.0.train`）、映射键排序遍历、叶子确定性字符串化（str 原样；bool/None 走 JSON 拼写）；上限 **500 键**，超出 `ValueError`——绝不静默截断 |
| `manifest["canonical"]["metrics"]` | metrics | `{metric_name: value}` 原样 float |
| `evidence` | artifacts `evidence/evidence.json` | `sort_keys` + indent、UTF-8；内容哈希回读可对账 |
| `RunResult.oof_path` | artifacts `evidence/<experiment_id>.csv` | 字节拷贝，sha256 不变 |
| `evidence["source_system"]` / `manifest.canonical["schema_version"]` | tags `source_system=cultivating` / `schema_version` | `decision` 标签待 P3-A（DecisionTrace）接入后补，现在刻意缺席 |

import-guard：mlflow 缺失时模块可正常导入；调用 `log_experiment_run`
抛 `RuntimeError` 并提示 `pip install mlflow`。追踪是可选能力，执行层
（SPEC-003）保持零 mlflow。

## 7. 复现性贯穿与 CI 说明

- **复现性贯穿到追踪层**：G2 双跑（同 IR + 同数据 + 同代码 + 同种子）
  ⇒ canonical manifest 全等 ⇒ 追踪层 params 全等、metrics **逐位相等**
  （`test_double_run_metrics_bitwise_equal_in_store` 以 sqlite 真跑证明）。
- **CI 里 mlflow 跳过说明（本地真验，CI 轻量）**：
  `tests/test_mlflow_adapter.py` 以 `pytest.importorskip("mlflow")` 开头——
  CI 未装 mlflow 时整模块 skip，不拖慢流水线；本地装有 mlflow 3.17.0，
  用例以 tmp_path 下 sqlite tracking_uri **真跑**（db 与 artifacts 均落
  临时目录，不污染仓库）。测试内在 import mlflow 前先设
  `os.environ["MLFLOW_DISABLE_AGENT_HINT"]="1"`，抑制 mlflow 3.17 的
  agent-hint 提示。decide 本体零 mlflow 依赖，CI 全量可跑。

## 8. 依赖纪律

- `ir/decide.py`：运行时纯 stdlib（`copy` / `math` / `statistics`）+
  `ir.experiment_ir.run_gates`；`RunResult` 仅 `TYPE_CHECKING` 引入，
  manifest 按鸭子类型读取；不 import mlflow/sklearn/numpy。
- `pipeline/mlflow_adapter.py`：stdlib + 可选 mlflow（guard 后导入）；
  不 import sklearn/numpy/pandas。
- 零对 A 线文件（`schemas/decision-trace/**`、
  `framework/src/ir/decision_trace.py`）的依赖——两线仅以 §4 的 dict
  契约相接。
