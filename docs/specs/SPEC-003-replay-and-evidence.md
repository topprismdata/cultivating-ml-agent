# SPEC-003: 重放执行与证据层（0.1.0）

状态：已实现
版本：0.1.0
基线：origin/main @ 9f56c53（SPEC-001 已合入）
实现：`framework/src/ir/runner.py`、`framework/src/ir/__init__.py`（懒导出）、`replays/`
测试：`tests/test_replay_reproducibility.py`

## 1. 目标与不变量

SPEC-001 把实验编译为不可隐式修改的 ExperimentIR 契约；本规范定义其上的
**执行与证据层**，并用两个提交在库内的重放用例证明计划基线 v0.1 §9 的
G2 复现门禁：

> 同 ExperimentIR + 同数据字节 + 同代码版本 + 同种子
> ⟹ **规范哈希（canonical manifest hash）逐字节一致，指标位型相等。**

三条不变量：

1. **易变物只进 volatile 区**：墙钟（`created_at` / `timestamp`）、主机名、
   Python 版本只出现在 `manifest["volatile"]` 与 `evidence["timestamp"]`，
   永不进入任何内容哈希。既往教训（任何 `uuid4()`/时间戳字段破坏内容哈希
   复现）由此结构性排除——所有标识符从内容派生
   （`evidence_id = ev-<ir_content_hash[:12]>`），禁止 uuid。
2. **canonical 区封闭**：manifest 的 canonical 字段集合固定（§3.1），
   序列化采用与 ExperimentIR 相同的规范化 JSON（排序键、紧凑分隔符、
   UTF-8，见 `experiment_ir.canonical_bytes`）。
3. **fail-closed**：门禁 blocked、IR 被篡改（`ContentHashMismatch`）、数据
   哈希不符（`DataHashMismatch`）一律拒绝执行；执行器在运行前还会重算
   `compute_content_hash(ir)`，传入已被原地改动的 dict 同样拒载。

## 2. 执行流程

`execute_experiment(ir, project_root, *, seed=42, data_root=None) -> RunResult`：

```
ExperimentIR dict
  │
  ├─ ① 授权门禁   authorize_execution(ir)；任一 gate fail ⇒ ExecutionBlocked
  │               （六个用例 tests 里 parametrize 了全部 invalid-g* 样例）
  │
  ├─ ② IR 完整性  compute_content_hash(ir) == ir["content_hash"]，否则
  │               ContentHashMismatch（防调用方加载后再原地篡改）
  │
  ├─ ③ 数据校验   dataset_snapshot_ref.uri 解析（data_root 覆盖，否则相对
  │               project_root）→ 逐块 sha256 全文件 → 与 IR 记录不符 ⇒
  │               DataHashMismatch；通过后 pd.read_csv 载入
  │
  ├─ ④ 标签过滤   标签列固定为 `target`（P2 重放约定，与 pipeline.oof 默认
  │               一致）；label 缺失行在切分前整体剔除（S6E5 风格用例的
  │               缺失标签场景），保留行 reset_index 保证位置稳定
  │
  ├─ ⑤ 切分       validation_protocol_ref → make_folds（四策略同名分派）：
  │               time_based 时按 time_col 物化 float 周数组（ISO 日期 →
  │               toordinal()/7），group 时物化 group_col 字符串数组，
  │               random_state=seed
  │
  ├─ ⑥ 训练       特征列 = 除保留集 {"id", "target", time_col} 外的全部
  │               CSV 列（按 CSV 列序，禁 set 迭代序）；非数值列按
  │               np.unique（排序）整数编码；逐折训练，折内种子 =
  │               (seed + fold_idx) mod (2^31-1) 确定性派生；
  │               candidate[0] 为默认执行候选，family 必须在 MODEL_FAMILIES
  │               注册（ridge：sklearn Ridge，candidate.params 透传（含
  │               alpha）；hgb：HistGradientBoostingRegressor，params 透传
  │               且 random_state 缺省取折种子）
  │
  ├─ ⑦ 指标       metric_definition_ref.name → utils/metrics 注册表按 OOF
  │               预测计算（本层不重复实现指标；rmsle/rmse/mae 均为既有
  │               实现）。metric_version 常量
  │               METRIC_VERSION = "framework-utils-metrics/1.0.0" 钉住该
  │               实现版本——若未来改指标实现，必须 bump 此常量并重造 golden
  │
  ├─ ⑧ OOF 落盘   build_oof_frame(oof_pred, ids, targets)（含真值）→
  │               project_root/outputs/oof/<experiment_id>.csv；写盘用
  │               float_format="%.6f"、LF 行尾、UTF-8、无索引列；未评分行
  │               （time_based 的训练段）oof_pred 记 NaN，指标只在有预测的
  │               行上计算
  │
  └─ ⑨ 产出       manifest（canonical + volatile）与 evidence（ANF
                  evidence-envelope 投影）→ RunResult
```

## 3. 产出物字段表

### 3.1 `RunResult.manifest["canonical"]`（全部进入内容哈希）

| 字段 | 语义 |
| --- | --- |
| `schema_version` | manifest 契约版本，常量 `"0.1.0"` |
| `ir_content_hash` | 所执行 IR 的 `content_hash` |
| `code_version` | 本模块所在仓库 `git rev-parse HEAD`；git 不可用 → `"unknown"`。刻意从框架仓库根解析而非 `project_root`（后者只是输出位置） |
| `data_sha256` | 数据文件实测 sha256（= IR 记录值） |
| `seed` | 全局种子 |
| `strategy` | `validation_protocol_ref.strategy` |
| `n_folds` | 实际折数（`time_based` 为单折前向切分 = 1） |
| `fold_sizes` | 每折 `{"train": n, "val": m}` 数组 |
| `metrics` | `{metric_name: OOF 指标值(float)}` |
| `oof_sha256` | OOF CSV 字节摘要 |
| `model_family` | `candidates[0].family` |
| `model_params` | `candidates[0].params` |

`manifest["volatile"]`：`created_at`（UTC ISO 墙钟）、`host`
（`platform.node()`）、`python_version`——永不参与任何哈希。

内容哈希：`manifest_canonical_hash(manifest) = sha256(canonical_json(manifest["canonical"]))`。

### 3.2 `RunResult.evidence`（ANF evidence-envelope 投影）

对齐 ADR-002 决议「评估结果一律落成 ANF evidence-envelope，ML Profile 不设
EvaluationResult 实体」。必填字段（测试守护）：

| 字段 | 值 |
| --- | --- |
| `evidence_id` | `ev-<ir_content_hash[:12]>`（内容派生，禁 uuid4） |
| `capability_id` | `"ml-experiment"` |
| `skill_id` | `"experiment-run"` |
| `evidence_type` | `"task_success"` |
| `source_system` | `"cultivating"` |
| `measurement_protocol` | `validation_protocol_ref` 的规范化 JSON 字符串（canonical_bytes，与 IR 哈希同规范） |
| `protocol_version` | `"0.1.0"` |
| `metric_name` / `metric_value` / `metric_version` | IR 指标名 / OOF 值 / `framework-utils-metrics/1.0.0` |
| `executor` | `"framework/src/ir/runner.py"` |
| `provenance_chain_id` | `pch-<ir_content_hash[:12]>` |
| `task_id` | `experiment_id` |
| `artifact_refs` | `[{"path": "outputs/oof/<experiment_id>.csv", "sha256": …}]`（相对 project_root） |
| `timestamp` | UTC ISO 墙钟——**唯一 volatile 字段** |

`evidence_content_hash(evidence) = sha256(canonical_json(去掉 volatile 字段))`；
`EVIDENCE_VOLATILE_FIELDS = ("timestamp",)`。篡改 `timestamp` 不改变内容哈希
（测试断言）。

### 3.3 `compare_replays(a, b, *, tolerance=1e-9) -> ReplayComparison`

逐项布尔 + 总结论 `identical`（全部布尔取合取）：

`canonical_hash_equal`、`metrics_within_tolerance`（容差只作用于该标志）、
`data_sha256_equal`、`ir_content_hash_equal`、`seed_equal`、`code_version_equal`。

注意：容差放宽能让指标标志转绿，但 canonical 哈希包含指标字节——任何实质
差异都会使总结论保持 False。重放相等按位成立，容差仅用于诊断近似数值漂移。

## 4. 确定性纪律清单

| # | 纪律 | 落点 |
| --- | --- | --- |
| 1 | 规范化 JSON：排序键、紧凑分隔符、UTF-8、`ensure_ascii=False` | `canonical_json` / `experiment_ir.canonical_bytes`，两处同规范 |
| 2 | uuid4/墙钟/主机名/解释器版本禁入 canonical 与一切内容哈希 | §1 不变量 1、§3 字段表 |
| 3 | 内容派生 ID：`ev-`/`pch-` + `ir_content_hash[:12]` | evidence 构造 |
| 4 | CSV fixture 固定格式：`%.6f` 浮点、LF 行尾、UTF-8、固定列序、固定类别值 | 两个 `make_fixtures.py` |
| 5 | fixture 数值流：`numpy` PCG64（`default_rng(42)`），遵循 numpy 流兼容政策 | 同上 |
| 6 | 折内种子由 `(seed, fold_idx)` 确定性派生 | `_fold_seed` |
| 7 | 类别编码按 `np.unique`（排序）映射，禁 set 迭代序 | `_feature_matrix` |
| 8 | 特征列按 CSV 列序（列表推导 dict/ndarray 迭代），保留集为显式集合常量 | `execute_experiment` |
| 9 | OOF 落盘 `%.6f` + LF + 无索引列，`oof_sha256` 即字节摘要 | OOF 写盘 |
| 10 | 双跑不等式只允许出现在 volatile 区 | 测试 `test_double_run_identical` |

## 5. 重放用例与 golden 回归

`replays/` 内两个提交在库的用例（fixture、IR、golden 全部提交）：

| 用例 | 形态 | 规模 | 协议 | 指标 | candidates |
| --- | --- | --- | --- | --- | --- |
| `s6e5-style/` | 表格时序回归（rig），7 特征（5 数值 + 2 类别），温和时变 + 年周期，~1% 缺失标签 | 2000 日行 | time_based, val_size_weeks=4 | rmsle | ridge → hgb |
| `store-sales-style/` | 多序列时序（3 店 × 180 天），promo/weekday/stock 预制特征 + 周季节性 | 540 行 | time_based, val_size_weeks=4 | rmse | ridge → hgb |

约定：`time_based` 切分语义沿用 `pipeline.splits.time_based_folds`——
`cutoff = max(time) - val_size_weeks`，`train = time < cutoff`、
`val = time ≥ cutoff`（日粒度数据上验证窗为 `val_size_weeks×7+1` 个日历日，
多序列用例中按全局截止日对每条序列同时截尾）。

`golden/manifest.canonical.json` 为双跑后的回归金标：

- `canonical`：规范 manifest，其中 `code_version` 以 `@CODE_VERSION@` 占位。
  **不钉死具体 commit**——钉死会让每次提交都使金标失效；占位符之外的每个
  字节（指标位型、fold_sizes、data_sha256、oof_sha256、model_params 等）
  由 `test_golden_regression` 逐字节回归。
- `canonical_sha256`：占位态 canonical 的内容哈希（测试同时验证金标自洽）。
- `metrics`：指标值冗余一份，便于人工审阅。
- `evidence_stable` / `evidence_content_sha256`：去掉 volatile `timestamp`
  后的证据信封及其内容哈希。

重造金标的流程：跑 `/tmp` 一次性脚本（或复跑两次 `execute_experiment` 后
按上述占位规则手工写入）→ 确认 `test_double_run_identical` 与
`test_golden_regression` 双绿 → 提交。任何「会改变 canonical 的」改动（改
fixture、改 IR、改模型参数、bump METRIC_VERSION）都必须同步重造 golden，
否则回归测试红——这正是防漂移设计。

## 6. 真实 Kaggle 数据接入

零网络、零下载内建于执行器；接入真实数据只需两步，**不改执行器代码**：

1. 本地取得数据（kaggle CLI 或人工下载）后，写一份新 revision 的 IR：
   - `dataset_snapshot_ref.uri` 填相对路径（相对 `data_root` 或
     `project_root`；绝对路径与 `file://` 亦可）；
   - `dataset_snapshot_ref.sha256` 填真实文件摘要（`sha256sum`）；
   - `rows`、`as_of`、`label_cutoff`（time_based 必填且 ≤ as_of）同步更新；
   - 重算 `content_hash`，`revision` 加一并以 `supersedes` 指向前版哈希
     （SPEC-001 §5 版本纪律）。
2. 执行时传 `data_root=Path("/path/to/kaggle-data")` 覆盖数据根：

```python
run = execute_experiment(ir, project_root=Path("outputs"),
                         seed=42, data_root=Path("~/data/store-sales").expanduser())
```

`data_root` 只覆盖**数据解析根**；OOF 产物仍落在 `project_root/outputs/oof/`。
数据一字节之差即 `DataHashMismatch` 拒绝执行——IR 里的 sha256 就是唯一准许
的字节指纹。

## 7. 计划 G2 → 代码映射

计划 G2（基线 v0.1 §9）：「同 IR + 同数据哈希 + 同代码版本 + 同种子 ⟹ 证据
规范哈希逐字节一致，指标差异为 0。」

| 计划条款 | 代码落点 |
| --- | --- |
| 同 IR | `ir_content_hash` 进 canonical（§3.1）；执行前重算哈希拒篡改（§2 ②） |
| 同数据哈希 | 执行前全文件 sha256 对账，不符 `DataHashMismatch`（§2 ③） |
| 同代码版本 | `code_version = git rev-parse HEAD` 进 canonical；git 不可用降级 `"unknown"`（此时所有运行皆 `"unknown"`，哈希仍互相一致） |
| 同种子 | `seed` 进 canonical；折种子 `(seed, fold_idx)` 派生（§4.6） |
| 证据规范哈希逐字节一致 | `manifest_canonical_hash` / `evidence_content_hash`；双跑全等由 `compare_replays` 逐项布尔裁决（§3.3） |
| 指标差异为 0 | 位型相等断言（`test_double_run_identical`）；容差仅诊断用（§3.3） |
| 端到端重放证明 | 两个提交用例 × 双跑（`tests/test_replay_reproducibility.py`）+ golden 回归（§5） |

## 8. 依赖纪律

- 执行器只用 stdlib + numpy/pandas/sklearn；**零 mlflow**（连可选适配都不
  做，P2 不接 mlflow）、零网络。测试 `test_runner_imports_no_mlflow` 在
  mlflow 被 import 阻断的环境下强制 `import runner` 成功。
- `ir/__init__.py` 对 runner 的再导出为**懒导出**（PEP 562
  `__getattr__`）：`import framework.src.ir.experiment_ir` 必须保持零
  numpy/sklearn（SPEC-001 纯净性回归守护）。
- 指标复用 `framework/src/utils/metrics.py` 既有实现。该模块经
  `importlib` 按文件路径加载：`framework.src.utils` 包的 `__init__` 会连带
  导出 submission 助手，其遗留顶层导入（`from pipeline.validate import …`）
  仅在把 `framework/src` 本身放上 sys.path 时才可解析——运行时采用的
  `framework.src.*` 包导入模式下不可用。按路径加载复用既有实现而不分叉；
  该遗留导入问题是 `utils/submission.py` 的既有缺陷，不在本层修复范围。
- 零新外部依赖：`docs/ontology/compatibility.bom.yaml` 无需变更。
