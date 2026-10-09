# ML Profile v0.1.0 草案（cultivating-ml-agent）

## 定位

本目录是 cultivating-ml-agent 机器学习实验域的受管概念层 **ML Profile 0.1.0 草案**。
命名空间 `prism-ml: <prism://ontology/ml/>` 状态为 **local / pending_proposal**：
概念 URI 为提案性质，尚未进入 prism-ontology 共享核心，治理通过前禁止跨 Profile 复用。

配套文档：

| 文档 | 路径 |
| --- | --- |
| 概念规格说明 | `docs/specs/SPEC-002-ml-profile.md` |
| DecisionTrace 边界决议 | `docs/adr/ADR-002-decision-trace-boundary.md` |
| 兼容性 BOM（G0） | `docs/ontology/compatibility.bom.yaml` |
| 共享核心净新增提案批次 | `docs/adr/ADR-001`（ExperimentIR 线维护，本文档仅引用） |

## 与 outlet-insight 样板的结构对齐

本草案在文件骨架上对齐只读样板 `prism-ontology/profiles/outlet-insight/`（@9d93a99）：

| outlet-insight | 本草案 | 对齐说明 |
| --- | --- | --- |
| `concepts.yaml` | `concepts.yaml` | 同为「version / profile / concepts 列表（uri、name、category）」骨架；本草案每个概念追加 `status`、`governance_status`、`parent`、`definition` 字段，以显式表达局部命名空间与晋升状态。 |
| `field-mapping.yaml` | `mappings.yaml` | 样板做物理列降级映射（mapping_only）；本草案改为「概念 → 共享核心父类」映射并登记 8 项共享核心净新增提案（proposed_local），`governance_status: mapping_only` 语义保留。 |
| `context.jsonld` | `context.jsonld` | 同为 `@context` 前缀映射；本草案在命名空间之外为每个概念与属性补充最小 term 定义，供实验文档直接使用短名。 |
| `constraints.shacl.ttl` | `ml.shacl.ttl` | 同为 NodeShape + sh:targetClass + 中文 sh:message 风格；本草案每个受管概念至少 1 条约束（kind 判别），并仿照样板的执行禁令写法为 EvaluationResult 设置 Prohibit 形状。 |
| `profile.yaml` | （本版不含） | 运行契约（allowed/prohibited concept scopes）待治理晋升阶段随 dist 发布补齐；草案阶段以 SPEC-002 与本 README 为准。 |

样板中的执行禁令（prohibited_execution_concepts）思想被继承为：本草案显式不设 `EvaluationResult` 实体（ADR-002），SHACL 侧以 `prism-ml:ProhibitEvaluationResultShape` 禁止实例化。

## 校验现状

- 全部 YAML / JSON 已通过 `python3`（`yaml.safe_load` / `json.load`）解析自检。
- `ml.shacl.ttl` 仅做语法级目检（@prefix 配对、分号句法、括号配对）。
- **pyshacl 未接入**：本版不做语义级校验；语义校验待 prism-ontology 治理 CI 接入 pyshacl 后执行。

## 晋升路径

1. **MS-PROP 提案批次**：将 `concepts.yaml` + `mappings.yaml`（含 8 项共享核心净新增，归属 ADR-001 提案批次）打包为 MS-PROP 提案，附 SPEC-002 的映射理由与 ADR-002 的边界决议。
2. **prism-ontology 治理评审**：命名空间授予（`prism://ontology/ml/`）、概念与 SHACL 评审、pyshacl 语义校验接入；评审意见回灌本草案并升版本号。
3. **dist 发布**：治理通过后经 prism-ontology dist 版本化发布；本仓库 `docs/ontology/compatibility.bom.yaml` 中的 `prism-ontology` 锁定引用随之更新，本 Profile 状态由 `pending_proposal` 翻转为受管。
