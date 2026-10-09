# SPEC-006: 跨域估计交换信封 EstimateEnvelope + 求解器适配（0.1.0）

状态：已实现
版本：0.1.0
基线：origin/main @ 9759eb1（SPEC-005 已合入）
实现：`schemas/estimate-envelope/0.1.0/`、`framework/src/adapters/`（estimates / pjp / warehouse / report）
测试：`tests/test_adapters_pjp.py`、`tests/test_adapters_warehouse.py`、`tests/test_cross_domain_report.py`
关联：SPEC-001/003/004/005、ADR-002、ADAPTER-001/002/003、计划基线 v0.1 §7/§12、G5 门禁

## 1. 目标与铁律

P4 跨域交付一件事：**ML 侧只产估计（点估计 + 残差尺度 + 量化声明），求解器独立执行硬约束**。两边之间只有 JSON 信封与 kwargs 拼装，没有代码依赖。

铁律（计划 §12）：

1. **ML 只产估计 + 置信范围**：信封携带 `value`（点估计）、`sigma`（残差 std）、可选 `confidence`；量化只把连续估计翻译成离散通道（星期几集合），从不产生"决策"。
2. **求解器独立执行硬约束**：义务次数 `k_c`、每日期恰一列、`sigma_budget` 全部由 `sp_solve_ip` 施加；适配器只拼形状，永不替求解器做约束推理。
3. **ML 零求解器依赖**：`framework/src/adapters/**` 禁止 import ortools/求解器/VisitModel/VisitIR/opticore；以 JSON 信封为界（架构守卫见 §7）。
4. **业务 KPI 是领域上报事实**：`import_domain_outcome` 把领域上报 JSON 折进 ANF evidence-envelope 时，业务 KPI 只落 `domain_kpis` 字段，**永不写入 `metric_name`/`metric_value`**（该 ML 指标通道在本路径钉死为 null）。
5. **店身份用稳定编码**（D7）：spec 层永远是稳定编码字符串；`int idx` 只在求解器传输形状里短暂存在，不配当身份、不进持久状态。

## 2. EstimateEnvelope 契约（schemas/estimate-envelope/0.1.0）

draft 2020-12，`additionalProperties: false`。样例：`examples/pjp-service-weekday.json`（PJP，含量化载荷）、`examples/warehouse-demand.json`（仓储，未量化）。

| 字段 | 类型 | 语义 |
|---|---|---|
| `schema_version` | `"0.1.0"` | 契约版本；语义变更新开目录 |
| `envelope_id` | `env-<12hex>` | sha256 派生：`"env-" + content_hash[:12]`；无 uuid4、无时钟 |
| `estimate_kind` | 枚举 | `pjp_service_weekday` \| `warehouse_demand` \| `warehouse_processing_time` |
| `created_at` | ISO 字符串 | **易失，永不入哈希**（时钟注入不改变身份） |
| `model_provenance` | 对象 | `{ir_content_hash(64hex), evidence_id(ev-<12\|64hex>), decision_id?(dt-<12hex>)}` — 回指 ExperimentIR 与 run 证据 |
| `units[]` | 数组 | `{unit_id, value, sigma, confidence?, quantized?}`，每店/每 SKU 一条 |
| `quantization_rule` | string\|null | 声明量化规则（如 `p80_weekday_coverage`）；null = 未量化 |
| `value_unit` | string | 信封级单位语义（对齐 visit-ir 家法）：`minutes`/`km`/`units_per_day` 等；`value` 与 `sigma` 共用 |
| `content_hash` | 64hex | `sha256(canonical(envelope − {created_at, content_hash, envelope_id}))`；canonical = sorted keys、无冗余空白、UTF-8（与 `ir.decision_trace.canonical_bytes` 同一实现） |

unit 约束：`unit_id` 非空字符串且**禁止裸数字**（schema 层 `^(?!\d+$).+$` 拒绝 `"123"`——D7 的第一道闸）；`sigma ≥ 0`（OOF 残差 population std，ddof=0）；`quantized.weekdays` 是 1..7 内不重复整数（见 §3）；`quantized.rule` 必须与信封级 `quantization_rule` 一致。

构建：`estimates.envelope_from_oof(oof_csv_path, group_col, ir, evidence, *, estimate_kind=..., value_unit="units_per_day", quantization="p80_weekday_coverage", quantile=0.8, clock=None)`。输入是 `build_oof_frame` 产物落盘的 CSV（必含 `group_col`/`target`/`oof_pred`，可选 `time`）；逐组 `value = round(mean(oof_pred), 6)`、`sigma = round(std(oof_pred − target, ddof=0), 6)`；`time` 列可用才产量化载荷（声明保留、载荷缺省）。仓储侧薄封装 `warehouse.warehouse_envelope_from_oof(...)` 固定 `estimate_kind="warehouse_demand"`、不量化。

身份算法（`estimates.verify_envelope_identity` 复算，篡改即拒）：`content_hash` 与 `envelope_id` 均为内容 sha256 派生，与 SPEC-001 `content_hash`、SPEC-004 `decision_id` 同一家法。

## 3. sigma 三义陷阱（0/1 基对照表）

"sigma" 在三个层面**同名不同义**，这是本规格最高频的错误源：

| 层面 | 所在 | 类型 | 基制 | 语义 |
|---|---|---|---|---|
| 残差尺度 σ | 信封 `units[].sigma` | number ≥ 0 | —（无基制） | OOF 残差标准差，估计不确定度 |
| 合同星期几 | `VisitContract.sigma`（VisitIR `visit_semantic_api/__init__.py:171`：`sigma: int  # 星期几, 0=周一`） | int | **0 基**（0=周一..6=周日） | 合同槽位星期几 |
| 预测服务日集合 | `sp_solve_ip(sigma=...)`（VisitModel `formulation.py:114`） | `{store_idx: set(int)}` | **1 基 ISO**（1=周一..7=周日） | 预测该店应被服务的星期几集合；列中店 w∉集合计违规 |

纪律：

1. **信封内部只说 1 基 ISO**（`quantized.weekdays ∈ 1..7`）。schema 层 `minimum: 1` + 适配器双重拒绝：`build_solver_inputs` 遇 0 抛 `AdapterError("...只接受 1 基 ISO 星期几...")`——0 出现即 0 基泄漏（`VisitContract.sigma` 语义跨过边界），必须先 +1。
2. 转换发生点唯一：`quantize_p80_weekday_coverage` 内 `pandas weekday()`（0 基）`+1`。其余代码不做任何基制运算。
3. 测试钉死：`test_sigma_basis_pinned_one_based`（OOF 周一 → 信封 1）、`test_zero_based_weekday_leak_rejected`（信封含 0 → 拒绝）。

## 4. 量化规则（p80_weekday_coverage）

残差 σ → (weekday 集合, budget) 的量化发生在 ML 侧适配器（VisitModel 无店级连续 sigma 通道）。规则声明式定义：

- 逐店逐 weekday 取 `oof_pred` 均值 `m_gw`；
- 店阈值 `θ_g = np.percentile({m_gw}, 80)`（线性插值，确定性）；
- `weekdays_g = {w : m_gw ≥ θ_g}`，升序输出，1 基 ISO。

性质：纯函数（同帧同输出 → 幂等）；最大均值 ≥ 任意分位数 ⇒ 集合非空（构造保证，dropna 后无 NaN 路径）。载荷 `{"rule", "weekdays", "quantile"}` 随 unit 落盘；`time` 列缺失时声明保留、载荷缺省（部分覆盖的求解后果见 §5）。`quantization=None`（仓储路径）不出载荷。

## 5. 求解器适配（ADAPTER-002, pjp.py）

### 5.1 build_solver_inputs（只拼形状，不调用求解器）

```
build_solver_inputs(envelope, code_to_idx, dates, pool, k_c, *, sigma_budget, timeout_s=60.0)
  -> {"dates", "k_c", "pool", "sigma", "sigma_budget", "timeout_s"}   # sp_solve_ip 精确 kwargs
```

- `dates`/`pool` 求解器原生：pool 列 `(date, route[int idx], km)`——`km` 是估计进入求解的唯一数值成本通道（语义为公里，目标 `int(round(km*1000))` 毫公里整数化）。
- `k_c` 按稳定编码传入，适配器经 `code_to_idx` 映射为 `{idx: count}`（身份纪律：调用方在本边界只说编码）。
- `sigma`：信封 `quantized.weekdays`（1 基）→ `{idx: set(1..7)}`。
- **部分覆盖拒绝**：k_c 中任一店缺量化集合即抛错——求解器把不在 sigma 里的店视为恒违规（`w not in sigma.get(c, set())`），部分覆盖是静默约束脚枪。
- `sigma_budget` 必填（与 sigma 同时给才生效的求解器语义，在此固化为契约）；校验 `dates ⊇ pool 日期`、route idx ⊆ 映射值域、km 非负有限、idx 唯一。
- `code_to_idx` 由调用方每次传入，适配器不读写持久状态（D7；测试钉死调用后映射不变、信封无 idx 通道）。

### 5.2 summarize_solve_result（确定性字段锁）

```
summarize_solve_result(result, *, sigma=None) -> {"objective_milli", "status", "violations"}
```

输入是 `sp_solve_ip(..., return_diagnostics=True)` 的 `(km, days, diagnostics)` 三元组。

**锁字段清单**：

| 字段 | 定义 | 锁定理由 |
|---|---|---|
| `objective_milli` | `int(round(km*1000))`（formulation.py:199 同款整数化；CP-SAT 目标本身是整数毫值，与 `diagnostics.objective_value_milli` 交叉核验，偏差 >1 抛错） | 位级确定 |
| `status` | diagnostics.status（OPTIMAL/FEASIBLE/INFEASIBLE/PRECHECK_INFEASIBLE/…） | 求解器状态字 |
| `violations` | `count_sigma_violations(days, sigma)`：每选中列按 `date.weekday()+1` 对列内每店查 `w ∈ sigma[c]`，违规店数线性可加（每店每列恰覆盖一次 ⇒ 可加性成立，与 formulation.py viol_terms 同构） | 由锁定的 days 内容决定；sigma 缺省时为 None，**不伪造** |

**不锁：selected days**。原因：CP-SAT `num_search_workers=8` 且无 random_seed（formulation.py:206），并列最优列的选择可在多线程调度下漂移——days 是解的内部表示，不是契约。测试断言两次运行 objective/status 相等，不比较 days。

超时契约：达 `timeout_s` 返回 `FEASIBLE + optimality_proven=False`；`optimality_proven` 不入锁字段（由 status 蕴含）。

### 5.3 import_domain_outcome（领域上报 → ANF evidence-envelope）

```
import_domain_outcome(payload, *, source) -> dict   # ANF evidence-envelope 形状
```

- `source` 调用方传入（`'visitmodel'` / `'warehouse-engine'`），原样落 `source_system`/`executor`；
- `payload` 白名单 `{envelope_id, outcome, domain_kpis, solver_status, task_id, timestamp}`，未知键拒绝（import 边界严格性）；
- `outcome ∈ {success, failure, task_success, task_failure}` → `evidence_type`；其余拒绝；
- **业务 KPI 只落 `domain_kpis`（逐字转录），`metric_name`/`metric_value`/`metric_version` 钉死 null**——领域上报事实永不冒充 ML 指标（测试做结构断言）；
- `evidence_id = "ev-" + sha256(canonical(内容 − timestamp))`，无 uuid4、无墙钟：import 路径全确定（timestamp 是领域上报事实，原样保留，不入哈希）；
- provenance 双通道回指信封：`provenance_refs.envelope_id` 逐字 + `provenance_chain_id = "pch-" + sha256(canonical({envelope_id}))[:12]`；`task_id` 缺省取 envelope_id。

## 6. G5 报告映射（report.py）

`cross_domain_report(cases) -> markdown`（内嵌 canonical JSON 块；`cross_domain_report_json` 单出 JSON）。每案例四列**严格分列、跨列零运算**：

| 列 | 来源 | 纪律 |
|---|---|---|
| ML 指标 | `ml_metric {name, value}` | ML 数字的唯一出现位置 |
| 业务 KPI | `domain_kpis`（键字典序转录） | 领域上报事实，不与 ML 指标混算 |
| 约束违规数 | `violations`（int\|None） | 求解域；None 不伪造为 0（渲染 `n/a`） |
| 计算成本 | `compute_cost.wall_time_s` | 求解域 |

case 白名单 `{case_id, ml_metric, domain_kpis, violations, compute_cost}`，未知/缺失键拒绝；float 一律 6 位舍入；纯函数（同输入逐字节同输出；case 顺序即报告顺序，无隐藏重排）。计划 G5 映射：**ML 指标列 = OOF/信封统计；业务 KPI 列 = `domain_kpis`；约束违规数 = `summarize_solve_result().violations`；计算成本 = 求解 wall time**。

## 7. 零求解器依赖纪律与测试分层

1. **架构守卫**（`test_adapters_never_import_solvers`）：正则扫描 `framework/src/adapters/*.py` 的 import 语句，命中 `ortools|visitmodel|visit_ir|visit_semantic_api|opticore` 即红（仿 VisitIR 家法的架构测试思路；不查散文——docstring 里出现名字合法）。
2. **求解器往返真验只在 tests**：`TestRealSolverRoundTrip` 以 `VISITMODEL_PATH` 环境变量守卫——**先判 `os.environ.get("VISITMODEL_PATH", "")` 非空，再判 `Path(...).exists()`**（空字符串 `Path("").exists()` 恒 True 的历史教训）。守卫通过后把 `<VISITMODEL_PATH>/src` 与同级 `VisitIR/src`、`OptiCore/src`（存在才加）插到 `sys.path` 再 import；未设置则整组 skip（CI 无本地领域路径自动跳过）。真验内容：4 店×4 日 fixture（VisitModel `tests/test_linkage_v2.py:22-35` 同形），baseline `sigma=None` vs ML sigma 两次求解——objective_milli/status 可复现；违规计数随 sigma_budget 单调不增（0..3 INFEASIBLE → ≥4 OPTIMAL/4 违规）；空 set 边界（每列恒违规，budget=0 INFEASIBLE、budget=8 违规=8）；`sigma_budget=0` 边界如实钉死（求解器语义：零容忍 ⇒ 对齐预测时可行、错位预测时 INFEASIBLE）。
3. **σ 路径边界钉死**：VisitModel 原有测试不覆盖 sigma 路径；本规格测试补齐空 set、budget=0、budget 单调三边界。
4. `import_domain_outcome` 的结构测试放 `tests/test_adapters_warehouse.py`（函数体在 pjp.py——按任务分工落位，SPEC 如实记录）。

## 8. 仓储适配（ADAPTER-003, warehouse.py）与 blocked-on-engine-repo

- 同一 EstimateEnvelope，`warehouse_demand` 种（§2 薄封装）。
- `stub_decision_engine(envelope, *, capacity) -> dict`：**reference stub**，输出自标 `{"engine": "reference-stub/0.1.0", "stub": true, "deterministic": true}`——按稳定编码序贪心装箱到 `capacity`（units_per_day），返回 accept/reject + 容量违规数（被拒 unit 计数）。纯函数：同输入逐字节同输出。它是接口占位，**不是仓储业务决策**。
- **真实引擎接入点：blocked-on-engine-repo**。仓储决策引擎在独立仓库（未定 tags/ref），在 BOM 记录其锁定 ref 之前，本仓只保留 stub 与信封契约；引擎仓库就绪后以新 BOM 版本锁定，替换点即 `stub_decision_engine` 的调用方（签名不变）。

## 9. 依赖与 BOM

- 零新 pip 依赖：适配器运行时仅 stdlib + numpy/pandas（既有基线，`pipeline/oof.py` 同级）；示例的 jsonschema 校验仅测试路径（`importorskip`）。
- VisitModel/VisitIR/OptiCore 是**本地路径测试依赖**（untagged-local）：仅守卫测试的往返真验使用，不入编译/执行路径，已记录于 `docs/ontology/compatibility.bom.yaml` 的 `local_domain_paths` 段。
- 往返真验依赖本地 checkout 的固定布局（`<VISITMODEL_PATH>/src` + 同级 `VisitIR/src`、`OptiCore/src`）；布局变化只影响守卫测试，不影响适配器与信封契约。
