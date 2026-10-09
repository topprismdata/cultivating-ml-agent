# ml-core-batch-001 — 上游提交路径与检查单

本目录是 cultivating-ml-agent 向 prism-ontology 提交 **8 项共享核心净新增概念**
（WorldStateRef、Task、Goal、Candidate、Execution、Claim、Outcome、Capability）
的提案批次草案。判定依据：`docs/adr/ADR-001-cross-repo-responsibilities.md` §决策.3。

**当前状态：`draft`（未提交上游）。提交时机待 prism-ontology 维护方裁定。**

## 文件清单

| 文件 | 内容 |
| --- | --- |
| `proposal.yaml` | 批次元数据：batch_id、status、上游流程、依赖的已发布类、范围与处置规则 |
| `concept-proposals.yaml` | 逐项提案：定义 / 父类候选（含否决备选及理由）/ 与 ML 局部命名空间的差分 / arena 边界 |
| `competency_questions.yaml` | 每概念 ≥1 条可回答 CQ（含 semantic_expressibility / data_answerability 评估） |

## 上游提交路径

1. **等待维护方裁定提交时机**（本草案 status: draft 的原因）。裁定后，由上游分配
   MS-PROP 编号（GOVERNANCE.md §2：编号单调递增、当前已分配至 MS-PROP-022；
   分配前本草案使用 ML-CORE-NNN 临时编号）。
2. **提交**：将本目录四个文件按上游目录约定复制到 prism-ontology `proposals/`
   下（先例：`proposals/mousheng/`），以 PR 形式进入 prism-ontology 治理评审。
3. **评审**：GOVERNANCE.md §2 六态状态机逐项裁定（accepted / aligned_to_existing /
   profile_local / mapping_only / rejected / deferred），五问法评估。
4. **发布**：GOVERNANCE.md §3 两段式发布（干净工作树源码提交 → 基于 commit 哈希
   构建 dist 并签署 manifest → Annotated Git Tag）；SemVer 按 §3.3（净新增类为
   非 breaking 变更）。
5. **回灌**：批次通过后，本仓库 `docs/ontology/compatibility.bom.yaml` 以**新 BOM
   版本**更新 `prism-ontology` 锁定引用（SHA 一经锁定不可变，修改必须走新 BOM
   版本）；`docs/ontology/ml-profile/0.1.0/` 的 `pending_proposal` 状态翻转，
   `mappings.yaml` 以新版本（supersedes）更新受影响映射（如 MLTask → Task → Activity）。

## 提交前检查单

- [ ] **仓库内一致性**：concept-proposals.yaml 的 8 项概念与 ADR-001 映射表一字不差
      （WorldStateRef、Task、Goal、Candidate、Execution、Claim、Outcome、Capability）；
      中文名称与 `docs/ontology/ml-profile/0.1.0/mappings.yaml` core_proposals 一致。
- [ ] **依赖已发布类核对**：父类候选（prism:core/InformationObject、Activity、
      Observation）均为 core.ttl 已发布类（BOM 锁定 ref `9d93a99`）。
- [ ] **YAML 可解析**：三个 YAML 通过 `python3 -c "import yaml,sys;[yaml.safe_load(open(f)) for f in sys.argv[1:]]"` 自检。
- [ ] **上游注册表完整性**：进入 upstream 后 `tests/test_profile_registry_integrity.py`
      必须通过（概念登记进 profile 注册表后的一致性把守）。
- [ ] **上游本体完整性**：`tests/test_ontology_integrity.py` 与
      `tests/test_cq_expressibility.py`（CQ 表达力回归）通过。
- [ ] **SHACL**：上游为净新增类补 NodeShape 约束（仿 constraints.shacl.ttl 中文体
      sh:message 风格），pyshacl 语义校验随上游治理 CI 执行。
- [ ] **版本兼容**：SemVer 评估（§3.3）；`docs/ontology/compatibility.bom.yaml`
      更新走新 BOM 版本；历史 dist 版本目录不可变（§3.2 发行包不可变性铁律）。
- [ ] **术语风格**：与 `dist/outlet-insight/0.1.0-rc4/concepts.yaml` 语感一致
      （uri / name / category 骨架 + 中文概念名）。

## 边界备忘（评审时主动声明）

- 批次通过前，8 项概念仅存在于 ML 局部命名空间
  （`prism://ontology/ml/`，status: local / pending_proposal），禁止跨 Profile 复用。
- 6 项可映射概念（Observation、Estimate、Evidence、Constraint、Decision、Plan）
  不在批次内，直接引用 `prism-core:` 已发布类。
- ML 局部扩展概念（MLTask、DatasetSnapshot、FeatureSet、ValidationProtocol、
  MetricDefinition、Hypothesis、ExperimentPlan、ExperimentRun、EvaluationResult、
  ModelArtifact、TransferEvidence）默认不申请进入 core.ttl（ADR-001）。
