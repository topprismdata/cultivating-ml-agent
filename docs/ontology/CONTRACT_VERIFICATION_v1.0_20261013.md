# TopPrism ML Agent 本体升级 · 跨仓库契约核验报告 v1.0

日期：2026-10-13 ｜ 输入：`TopPrism_ML_Agent_Ontology_Upgrade_v0.1.md`（2026-10-09 基线，状态"待跨仓库契约核验"）｜ 方法：四仓实跑核验 + 跨仓投影互验

## 0. 总裁决

**计划文档已被仓库 HEAD 超越。** 第一批十项工程任务在 `cultivating-ml-agent@0f87aed` 中**九项已实现**（含 schema+代码+测试），一项（仓储 ADAPTER-003）为显式参考桩待引擎仓。测试基线 **220 passed / 7 skipped**（Python 3.12 实跑，60s）。§8 P0 四项修复**全部落地成形**。跨仓契约**实测互验通过**（DecisionTrace→ANF experience-record 投影 3/3 valid）。

结论：本线不需要"重新实施"，需要的是**核验确认 + 三处真问题处置**。核验状态可从"待核"升级为"已核·契约成立"。

## 1. 第一批任务对照（计划 §12）

| 任务 | 状态 | 证据 |
|---|---|---|
| ADR-001 跨仓职责 | ✅ 已实现 | `docs/adr/ADR-001-cross-repo-responsibilities.md`（authority 平面"Not promotion"） |
| ADR-002 DecisionTrace 边界 | ✅ 已实现 | `docs/adr/ADR-002-decision-trace-boundary.md`；decision_trace.py 头部引用 |
| SPEC-001 ExperimentIR Schema | ✅ 已实现 | `schemas/experiment-ir/0.1.0/` + `framework/src/ir/experiment_ir.py`（560 行，6 硬门禁 G1-G6，canonical sha256 + supersedes 修订） |
| SPEC-002 ML Profile 与映射 | ✅ 提案成形 | `docs/ontology/ml-profile/0.1.0/`（concepts.yaml 全标 pending_proposal + SHACL + JSON-LD）；**概念未上推 prism-ontology 注册表**（见 §3） |
| TEST-001 P0 配置/OOF/泄漏回归 | ✅ 已实现 | `tests/test_pipeline_guards.py`（模板键/OOF 真值/嵌套解析/时序·分组·分层不变量） |
| TEST-002 两案例重放 | ✅ 已实现 | `replays/`（store-sales-style + s6e5-style + golden manifest.canonical.json）；`runner.py` 同 IR+数据+代码+种子⇒逐字节一致 |
| ADAPTER-001 MLflow | ✅ 已实现 | `pipeline/mlflow_adapter.py`（tag source_system=cultivating；mlflow 可选依赖，CI 刻意不装） |
| ADAPTER-002 PJP | ✅ 已实现 | `adapters/pjp.py`（EstimateEnvelope→sp_solve_ip kwargs，从不 import ortools，业务 KPI 与 ML 指标分列） |
| ADAPTER-003 仓储 | 🟡 参考桩 | `adapters/warehouse.py`（stub_decision_engine 显式 blocked-on-engine-repo，SPEC-006 §8） |
| GOV-001 ANF 晋级接口 | ✅ 已实现 | `governance/promotion.py`（633 行，阈值全来自 hash 锁定 policy；**G4 只产 draft，activate 恒抛异常**——授权边界正确） |

## 2. §8 P0 修复核验（逐条）

| P0 线索 | 现状 | 证据 |
|---|---|---|
| config cv_strategy/strategy 不一致 | ✅ 已修 | `config_template.yaml` training 节用 `strategy`；`config.py` ValidationConfig.strategy 枚举 stratified/kfold/time_based/group；未知键警告丢弃 |
| OOF 保存列名而非真实标签 | ✅ 已修 | `pipeline/oof.py` build_oof_frame 列序 [id,]target,oof_pred；targets 缺失**整个省略 target 列，绝不伪造占位** |
| 泄漏测试缺失 | ✅ 已补 | test_pipeline_guards 覆盖时序/分组/分层切分不变量（G1-G6 门禁在 IR 层强制） |
| "AUC≈0.5 即对齐"过强结论 | ✅ 已分级 | 16-principles 原则16"Adversarial Validation **Limitation**"；failure 文档给分级：<0.55 停止 / 0.55-0.70 谨慎 / >0.70 净化必要 |

## 3. §13 待核事项核验

### 3.1 各仓 HEAD（实测）
- cultivating-ml-agent `0f87aed` · agent-nurture-framework `33f444b` · skill-tester `c5022b8` · topprismwiki `014c66c` · prism-ontology `5009080`

### 3.2 真实已发布 Profile（prism-ontology）
- dist/ **仅 outlet-insight 一个 profile**（rc1–rc4）；rc4 concepts.yaml **34 概念**（Core 8/Outlet 11/Quality 2/Sales 5/Insight 8），47 关系，563 行度量定义
- 发布流程：GOVERNANCE.md 六态提案状态机 + 两段式确定性发行 + 发行不可变铁律 + SemVer；SHACL 在 TTL 源与 tests/ 双环节

### 3.3 共享核心概念逐项对照（计划 §3 的 14 词）
| 概念 | 注册表 | 说明 |
|---|---|---|
| Observation | ✅ 存在 | Core 类 |
| Estimate | 🟡 近似 | 发布为 `DerivedEstimate`（SHACL 强制 ≠Observation） |
| Evidence | ✅ 存在 | Core 类 |
| Claim | 🟡 近似 | 发布为 `InsightClaim` |
| Decision | ✅ 存在 | Core 类 |
| Plan | ✅ 存在 | Core 类 |
| Constraint | ✅ 存在 | Core 类 |
| Candidate | 🟡 近似 | 发布为 `ElevationCandidate`（worldStateEffect=none） |
| WorldStateRef | ❌ 不存在 | 计划侧提案 |
| Task | ❌ **显式禁止** | profile prohibited_execution_concepts |
| Goal | ❌ 不存在 | |
| Execution | ❌ **显式禁止** | 同上（本体只读语义，禁执行概念=设计意图非疏漏） |
| Outcome | ❌ 不存在 | |
| Capability | ❌ 不存在 | ANF 侧有 capability-record schema，但**未上推 prism-ontology 注册表** |

**关键判定**：计划 §3 自己写明"均须逐项核对现有注册表，未发布者视为提案"——核对结果 **8 存在/近似、6 不存在**，其中 Task/Execution 的"不存在"是 ontology 的**主动设计**（禁止执行概念，ML 侧 ExperimentIR 用 `task_ref`/`world_state_ref` 作**引用**而非注册概念，与禁止令不冲突）。ML Profile 全部 concepts 标 pending_proposal，符合"先局部 Profile 再申请晋升"纪律。

### 3.4 ANF 生命周期命名（发现真问题 #1）
- schema `capability-record.lifecycle_status` 枚举：`draft | active | refining | stable | disputed | deprecated`
- Charter §3.2 六契约生命周期：`Candidate → Observed → Validated Local → Validated Transfer → Stable → Disputed/Deprecated`
- **两套词表不一致**（refining/active 无对应，Candidate/Observed/Validated Local/Transfer 无对应）——ANF 内部文档与 schema 分裂，需 ADR 裁决统一或显式声明映射关系

### 3.5 授权模型 + CI 状态
- 授权：promotion.py `activation_authority=A2`（default-policy.json，hash 锁定，locked_at 2026-10-09）；G4 只产 draft、activate 恒抛——**授权边界实现正确**
- CI：cultivating-ml-agent `.github/workflows/ci.yml` 3.10/3.12 矩阵，**刻意不装** mlflow/lightgbm/xgboost/catboost/kaggle（可选依赖边界）；实跑 220 passed/7 skipped
- ANF：无测试无 CI（04_ARCHITECTURE_REVIEW 自承为风险）——**真问题 #2**

## 4. 跨仓契约互验（实测，非声明）

`project_to_anf()` 把 DecisionTrace 投影为 ANF experience-record，喂进 ANF schema 校验：
```
[full-lifecycle] → 通过  record_id=exp-er-dt-fa5c8e01e0d7-evaluated
[rejected]       → 通过  record_id=exp-er-dt-fa5c8e01e0d7-rejected
[correction]     → 通过  record_id=exp-er-dt-fa5c8e01e0d7-evaluated
```
record_id 模式 `exp-er-<decision_id>-<status>` 满足 ANF `^exp-[A-Za-z0-9._-]+$`；投影有损且按 status append-only，全保真留在 DecisionTrace（ADR-002 OQ1-OQ3 决议）。**契约成立。**

## 5. 遗留真问题（需处置，非阻塞核验）

|#|问题|建议|
|---|---|---|
|Q1|ANF 生命周期词表分裂（schema vs Charter）|ANF 出 ADR 统一或声明映射；ML 侧 project_to_anf 的 trust_level 映射需对齐最终词表|
|Q2|ANF 无测试无 CI|至少给 4 schema 加 jsonschema 正/负例测试 + frontmatter 解析器单测（复用 ML 仓 schemas 正反例范式）|
|Q3|Capability 概念未上推注册表|ML Profile 晋升前，Capability 只在 ANF schema 层——跨域消费需 prism-ontology 提案走六态流程；当前 draft-only 无 activate 是正确保守|
|Q4|仓储 ADAPTER-003 是桩|blocked-on-engine-repo 已显式声明，非缺陷；等真实仓储引擎|


## 6. 处置记录（同日）

| 问题 | 处置 | 证据 |
|---|---|---|
| Q1 词表分裂 | **ANF ADR-002**（proposed，待维护者签署）：Charter §3.2 为规范词表；capability-record 0.1.0→**0.2.0**（枚举 candidate/observed/validated-local/validated-transfer/stable/disputed/deprecated，新增 validated-transfer=ML 策略 min_task_families=2+min_negative_transfer=1 的门禁机器化）；structural_lint/knowledge-architecture 同步；ML 侧协同改 G4 字面量 draft→candidate（promotion.py/policy schema/SPEC-007/3 断言），ML 内部标识符（approved_for_draft 等 ledger 持久格式）**不改名**避免迁移 | 双仓实跑：ANF **25 passed**（新套件）；ML **220 passed/7 skipped**，其中 TestAnfSchema 2 项对新枚举实校验 PASSED |
| Q2 ANF 无测试 | **tests/×3 + examples/schemas 正反例 + ci.yml（3.10/3.12）+ requirements-dev**；含 ADR-002 回归钉：旧值 draft 必须校验失败、枚举漂移即 CI 红 | 25/25 绿，CI success |
| Q3 概念未上推 | **prism-ontology PR #24**（batch-003）：ml-profile 10 项局部概念（MS-PROP-038..047，一概念一文件+批次 README，预期全 profile_local）；共享核心 8 项已由 PR #21 先行提交不重复；发现编号空间冲突（datagovernance 023 已并入 main 与 PR#21 的 023 撞号，main 台账仍标 022）已在 PR 与 README 中提请委员会处置 | 本体仓 9 测试绿；PR: github.com/topprismdata/prism-ontology/pull/24 |
| 附加发现（已修） | ① ML 测试 ANF schema 拉取路径陈旧克隆永不刷新 + 原 clone 目标路径错误（嵌套 schemas/schemas，自动拉取实际从未生效）→ 已改 clone-to-root + fetch/reset origin/main，清空缓存端到端验证 220/7 绿 ② PJP 5 项 parity 本地用 VisitModel venv 实跑 **26/26**（真实求解器对照），CI 无领域路径合理 skip | 本仓后续提交 + 核验输出 |

签署 ADR-002 后，capability-record 具备 joint sign-off 条件（词表统一 + 双仓互验绿）。

## 7. 独立判断

计划文档的**架构纪律与仓库实现高度自洽**（事实/推断分离、内容哈希 ID、无 uuid4、时钟注入不入哈希、四眼分离、fail-closed、只产 draft 不自动 activate）——这套纪律与我同期在 SRP 四模块落地的十一仓**同源同型**（DecisionEpisode 重放锚、ActionLedger append-only、plan-release 四眼链、specification-frozen 诚实定级）。两条线可共享的**已验证契约**：Evidence/Claim/Decision/Outcome 语义 + 不可变事件链 + hash 锁定策略。

**不建议**为对齐"本体"而重构已通过的 220 测试实现——核验通过即应把计划文档状态从"待核"改为"已核·契约成立"，剩余是 Q1-Q4 治理动作，不是工程缺口。
