"""GOV-001 promotion gate tests (SPEC-007).

Covers: pre-registered threshold evaluation (逐条 reason, fixed vocabulary),
draft-only output (G4: lifecycle_status 恒为 draft, activate 恒抛), policy
content-hash pinning (tamper 拒载), deterministic decision identity (no
uuid4/wall clock in hashes), mechanical P3 projection reuse, ledger
roundtrip + tamper interception, and validation of the draft capability
record plus evidence fixtures against the real upstream ANF schemas
(agent-nurture-framework), skipped with a stated reason if unavailable.
"""
import copy
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.src.governance import (
    DEFAULT_LEDGER_DIR,
    PROMOTION_REASONS,
    GovernanceError,
    GovernanceLedgerConflictError,
    GovernanceLedgerTamperError,
    PolicyError,
    activate_capability,
    append_decision,
    compute_policy_content_hash,
    draft_experience_projection,
    evaluate_promotion,
    load_decision,
    load_policy,
    verify_policy,
)
from framework.src.ir import decision_trace as dt

jsonschema = pytest.importorskip("jsonschema")

POLICY_SCHEMA_PATH = ROOT / "schemas" / "governance" / "promotion-policy.schema.json"
DEFAULT_POLICY_PATH = (
    ROOT / "framework" / "src" / "governance" / "policies" / "default-policy.json"
)

ANF_SCHEMA_DIR = Path(
    "/tmp/anf-skilltester-check/agent-nurture-framework/schemas"
)
ANF_REPO_URL = "https://github.com/topprismdata/agent-nurture-framework"

NOW = datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc)
FRESH_TS = "2026-09-01T00:00:00+00:00"
CAPABILITY_ID = "cap-demo-tabular-skill"


def anf_schema_or_none(name):
    """Locate (or fetch) a real upstream ANF schema; None if unavailable."""
    path = ANF_SCHEMA_DIR / name
    if path.exists():
        return path
    subprocess.run(
        ["git", "clone", "--depth", "1", ANF_REPO_URL, str(ANF_SCHEMA_DIR)],
        capture_output=True,
    )
    if path.exists():
        return path
    return None


def make_evidence(index, *, evidence_type="task_success", task_family="tabular", timestamp=FRESH_TS):
    """ANF evidence-envelope-shaped fixture (validated against the real
    schema in TestAnfSchema)."""
    return {
        "evidence_id": f"ev-fixture-{index:04d}",
        "capability_id": CAPABILITY_ID,
        "skill_id": "skill-demo",
        "evidence_type": evidence_type,
        "source_system": "cultivating",
        "measurement_protocol": "replay/oof-v1",
        "protocol_version": "1.0.0",
        "metric_name": "auc",
        "metric_value": 0.861,
        "metric_version": "1.0.0",
        "task_id": f"task-{index:04d}",
        "task_family": task_family,
        "executor": {"name": "agent-ml", "version": "0.8.4"},
        "artifact_refs": [],
        "provenance_chain_id": "prov-fixture0000000000",
        "timestamp": timestamp,
    }


def passing_evidence():
    """Minimum evidence set satisfying the default pre-registered thresholds."""
    return [
        make_evidence(1, task_family="tabular"),
        make_evidence(2, task_family="timeseries"),
        make_evidence(3, task_family="tabular"),
        make_evidence(4, evidence_type="negative_transfer"),
    ]


@pytest.fixture(scope="module")
def policy():
    return load_policy(DEFAULT_POLICY_PATH)


# ---------------------------------------------------------------------------
# Policy pinning (pre-registration)
# ---------------------------------------------------------------------------

class TestPolicyPinning:
    def test_default_policy_verifies_and_matches_schema(self, policy):
        assert verify_policy(policy) == policy
        schema = json.loads(POLICY_SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.validate(policy, schema)

    def test_pre_registered_threshold_values_locked(self, policy):
        assert policy["thresholds"] == {
            "min_task_success": 3,
            "min_task_families": 2,
            "require_negative_case": True,
            "min_negative_transfer": 1,
            "max_evidence_age_days": 180,
            "require_no_dispute": True,
        }
        assert policy["activation_authority"] == "A2"
        assert policy["supersedes"] is None

    def test_locked_at_never_enters_content_hash(self, policy):
        shifted = copy.deepcopy(policy)
        shifted["locked_at"] = "2099-01-01T00:00:00+00:00"
        assert compute_policy_content_hash(shifted) == policy["policy_content_hash"]

    def test_threshold_change_changes_hash(self, policy):
        drifted = copy.deepcopy(policy)
        drifted["thresholds"]["min_task_success"] = 2  # 事后改口径
        assert compute_policy_content_hash(drifted) != policy["policy_content_hash"]
        with pytest.raises(PolicyError, match="policy_content_hash 不匹配"):
            verify_policy(drifted)

    def test_tampered_policy_rejected_on_load(self, tmp_path, policy):
        forged = copy.deepcopy(policy)
        forged["thresholds"]["max_evidence_age_days"] = 100000  # 哈希未重算
        path = tmp_path / "forged-policy.json"
        path.write_text(json.dumps(forged, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(PolicyError):
            load_policy(path)

    def test_missing_field_rejected(self, policy):
        broken = copy.deepcopy(policy)
        del broken["thresholds"]["require_negative_case"]
        with pytest.raises(PolicyError, match="缺少"):
            verify_policy(broken)


# ---------------------------------------------------------------------------
# Threshold evaluation (逐条 reason, fixed vocabulary)
# ---------------------------------------------------------------------------

class TestThresholdTriggers:
    def evaluate(self, policy, evidence, **kw):
        kw.setdefault("capability_id", CAPABILITY_ID)
        kw.setdefault("risk_level", "medium")
        kw.setdefault("scope", "project")
        kw.setdefault("now", NOW)
        return evaluate_promotion(kw.pop("capability_id"), evidence, policy, **kw)

    def test_single_success_rejected(self, policy):
        decision = self.evaluate(policy, [make_evidence(1)])
        assert decision.status == "rejected"
        assert [r["reason"] for r in decision.reasons] == [
            "insufficient_task_success",
            "insufficient_task_families",
            "missing_negative_case",
        ]
        assert decision.draft_capability_record is None

    def test_few_families_rejected(self, policy):
        evidence = [
            make_evidence(1, task_family="tabular"),
            make_evidence(2, task_family="tabular"),
            make_evidence(3, task_family="tabular"),
            make_evidence(4, evidence_type="negative_transfer"),
        ]
        decision = self.evaluate(policy, evidence)
        assert [r["reason"] for r in decision.reasons] == ["insufficient_task_families"]

    def test_missing_negative_case_rejected_g4(self, policy):
        evidence = [
            make_evidence(1, task_family="tabular"),
            make_evidence(2, task_family="timeseries"),
            make_evidence(3, task_family="tabular"),
        ]
        decision = self.evaluate(policy, evidence)
        assert decision.status == "rejected"
        assert [r["reason"] for r in decision.reasons] == ["missing_negative_case"]

    def test_stale_evidence_rejected(self, policy):
        stale_ts = (NOW - timedelta(days=181)).isoformat()
        evidence = passing_evidence()
        evidence.append(make_evidence(5, timestamp=stale_ts))
        decision = self.evaluate(policy, evidence)
        assert [r["reason"] for r in decision.reasons] == ["stale_evidence"]
        # 时钟不入哈希：stale detail 只带证据 id 与阈值，不带计算出的天数
        assert str(181) not in decision.reasons[0]["detail"]

    def test_freshness_boundary_exactly_180_days_passes(self, policy):
        edge_ts = (NOW - timedelta(days=180)).isoformat()
        evidence = [
            make_evidence(1, task_family="tabular", timestamp=edge_ts),
            make_evidence(2, task_family="timeseries", timestamp=edge_ts),
            make_evidence(3, task_family="tabular", timestamp=edge_ts),
            make_evidence(4, evidence_type="negative_transfer", timestamp=edge_ts),
        ]
        decision = self.evaluate(policy, evidence)
        assert decision.status == "approved_for_draft"

    def test_unresolved_dispute_and_contradiction_rejected(self, policy):
        evidence = passing_evidence() + [
            make_evidence(6, evidence_type="dispute"),
            make_evidence(7, evidence_type="contradiction"),
        ]
        decision = self.evaluate(policy, evidence)
        assert [r["reason"] for r in decision.reasons] == [
            "unresolved_dispute",
            "unresolved_contradiction",
        ]

    def test_resolved_dispute_ignored(self, policy):
        resolved = make_evidence(6, evidence_type="dispute")
        resolved["resolved"] = True
        decision = self.evaluate(policy, passing_evidence() + [resolved])
        assert decision.status == "approved_for_draft"

    def test_all_reasons_reported_at_once(self, policy):
        evidence = [make_evidence(1, timestamp=(NOW - timedelta(days=200)).isoformat())]
        decision = self.evaluate(policy, evidence)
        assert [r["reason"] for r in decision.reasons] == [
            "insufficient_task_success",
            "insufficient_task_families",
            "missing_negative_case",
            "stale_evidence",
        ]

    def test_reason_vocabulary_closed(self, policy):
        decision = self.evaluate(policy, [make_evidence(1)])
        for entry in decision.reasons:
            assert entry["reason"] in PROMOTION_REASONS
            assert entry["detail"]

    def test_malformed_evidence_rejected_as_contract_error(self, policy):
        bad = make_evidence(1)
        bad["evidence_type"] = "vibes"
        with pytest.raises(GovernanceError, match="evidence_type"):
            self.evaluate(policy, [bad])
        with pytest.raises(GovernanceError, match="timestamp"):
            self.evaluate(policy, [{**make_evidence(1), "timestamp": "not-a-date"}])

    def test_input_validation_of_capability_risk_scope(self, policy):
        with pytest.raises(GovernanceError, match="capability_id"):
            self.evaluate(policy, passing_evidence(), capability_id="not-a-cap-id")
        with pytest.raises(GovernanceError, match="risk_level"):
            self.evaluate(policy, passing_evidence(), risk_level="existential")
        with pytest.raises(GovernanceError, match="scope"):
            self.evaluate(policy, passing_evidence(), scope="galaxy")


# ---------------------------------------------------------------------------
# Approval = draft production only (G4)
# ---------------------------------------------------------------------------

class TestDraftOnlyApproval:
    def test_full_pass_produces_draft_not_active(self, policy):
        decision = evaluate_promotion(
            CAPABILITY_ID,
            passing_evidence(),
            policy,
            risk_level="medium",
            scope="project",
            now=NOW,
        )
        assert decision.status == "approved_for_draft"
        record = decision.draft_capability_record
        assert record is not None
        assert record["lifecycle_status"] == "draft"
        assert record["lifecycle_status"] != "active"
        assert record["risk_level"] == "medium"
        assert record["scope"] == "project"
        assert record["capability_id"] == CAPABILITY_ID

    def test_provenance_chain_derived_from_evidence_ids(self, policy):
        decision = evaluate_promotion(
            CAPABILITY_ID,
            passing_evidence(),
            policy,
            risk_level="low",
            scope="team",
            now=NOW,
        )
        record = decision.draft_capability_record
        evidence_ids = sorted({e["evidence_id"] for e in passing_evidence()})
        assert record["evidence_ids"] == evidence_ids
        assert record["provenance_chain_id"].startswith("prov-")
        # 同一证据链 -> 同一 provenance；证据变 -> provenance 变
        other = evaluate_promotion(
            CAPABILITY_ID,
            passing_evidence()[:3] + [make_evidence(9, evidence_type="negative_transfer")],
            policy,
            risk_level="low",
            scope="team",
            now=NOW,
        )
        assert (
            other.draft_capability_record["provenance_chain_id"]
            != record["provenance_chain_id"]
        )
        twin = evaluate_promotion(
            CAPABILITY_ID,
            passing_evidence(),
            policy,
            risk_level="low",
            scope="team",
            now=NOW,
        )
        assert twin.draft_capability_record["provenance_chain_id"] == record["provenance_chain_id"]

    def test_authority_constraints_carry_activation_authority(self, policy):
        decision = evaluate_promotion(
            CAPABILITY_ID,
            passing_evidence(),
            policy,
            risk_level="medium",
            scope="project",
            now=NOW,
        )
        constraints = decision.draft_capability_record["authority_constraints"]
        assert policy["policy_id"] in constraints
        assert any("a2" in c for c in constraints)  # activation_floor_a2
        assert decision.draft_capability_record["activation_authority"] == "A2"

    def test_decision_id_excludes_wall_clock(self, policy):
        early = evaluate_promotion(
            CAPABILITY_ID, passing_evidence(), policy,
            risk_level="medium", scope="project", now=NOW,
        )
        # 平移参考时钟 1 天：评估仍在 freshness 窗口内，仅 evaluated_at 变化
        late = evaluate_promotion(
            CAPABILITY_ID, passing_evidence(), policy,
            risk_level="medium", scope="project",
            now=NOW + timedelta(days=1),
        )
        assert early.decision_id == late.decision_id
        assert early.evaluated_at != late.evaluated_at  # 易失审计字段
        assert early.to_dict()["decision_id"] == early.decision_id
        assert PromotionDecision_roundtrip(early)

    def test_activate_capability_always_raises_g4(self, policy):
        decision = evaluate_promotion(
            CAPABILITY_ID, passing_evidence(), policy,
            risk_level="medium", scope="project", now=NOW,
        )
        with pytest.raises(GovernanceError, match="激活需 A2\\+ 人工授权，接口不自动晋升"):
            activate_capability(decision.draft_capability_record)
        with pytest.raises(GovernanceError):
            activate_capability(decision.draft_capability_record, actor="root", authorization_ref="auth://all")
        with pytest.raises(GovernanceError):
            activate_capability({})  # 无条件拒绝：不存在可激活路径


def PromotionDecision_roundtrip(decision):
    """PromotionDecision <-> dict 互转保持身份与内容。"""
    from framework.src.governance import PromotionDecision

    restored = PromotionDecision.from_dict(decision.to_dict())
    return restored == decision and restored.decision_id == decision.decision_id


# ---------------------------------------------------------------------------
# P3 projection reuse (mechanical)
# ---------------------------------------------------------------------------

class TestDraftExperienceProjection:
    def _p3_projection(self):
        outcome = {
            "intent_ref": "intent://task/demo",
            "state_snapshot_ref": {"ir_content_hash": "a" * 64},
            "alternatives": [
                {"candidate_id": "cand-lgbm", "family": "lightgbm", "params": {}, "metric_value": 0.86}
            ],
            "constraint_check": [{"gate_id": "G3_METRIC_DEFINITION", "status": "pass"}],
            "baseline_ref": {"baseline_id": "baseline-logreg", "protocol_ref": {}, "score": 0.85},
            "selected_option": {
                "candidate_id": "cand-lgbm", "family": "lightgbm", "params": {}, "metric_value": 0.86
            },
            "rejected_options_and_reasons": [],
            "uncertainty": {"metric_std": 0.01, "margin": 0.02, "note": ""},
            "decision_actor": "agent-ml",
            "authorization_ref": "auth://g6/001",
            "evidence_refs": ["evidence://run-001"],
            "metric_name": "auc",
            "metric_direction": "maximize",
        }
        decision = dt.new_decision(outcome)
        dt.append_event(
            decision, "approved", actor="human-reviewer"
        )
        dt.append_event(
            decision, "executed", actor="agent-ml", execution_ref="manifest://run-001"
        )
        dt.append_event(
            decision,
            "evaluated",
            actor="agent-ml",
            evidence_refs=["evidence://run-001/envelope"],
            outcome_refs=["outcome://run-001"],
        )
        return dt.project_to_anf(decision)

    def test_valid_projection_becomes_evidence_source(self):
        projection = self._p3_projection()
        source = draft_experience_projection(projection)
        assert source["evidence_id"].startswith("ev-")
        assert source["evidence_type"] == "task_success"  # outcome.result=success
        assert source["metric_name"] == "auc"
        assert source["metric_value"] == 0.86
        assert source["timestamp"] == projection["timestamp"]
        assert source["projection_ref"] == projection["record_id"]
        assert source["provenance_chain_id"].startswith("prov-")
        # 机械转录：同输入同输出（无 uuid4/墙钟）
        assert draft_experience_projection(projection) == source

    def test_projection_feeds_promotion_evaluation(self, policy):
        projection = self._p3_projection()
        source = draft_experience_projection(projection)
        source["task_family"] = "tabular"  # 家族信息在调用点补齐（P3 投影不携带）
        evidence = [
            source,
            make_evidence(2, task_family="timeseries"),
            make_evidence(3, task_family="tabular"),
            make_evidence(4, evidence_type="negative_transfer"),
        ]
        decision = evaluate_promotion(
            CAPABILITY_ID, evidence, policy,
            risk_level="medium", scope="project", now=NOW,
        )
        assert decision.status == "approved_for_draft"

    def test_failure_projection_maps_to_task_failure(self):
        projection = self._p3_projection()
        bare = dict(projection)
        bare["outcome"] = {"result": "failure", "notes": "regressed"}
        source = draft_experience_projection(bare)
        assert source["evidence_type"] == "task_failure"

    def test_incomplete_projection_rejected(self):
        projection = self._p3_projection()
        broken = dict(projection)
        del broken["decision_summary"]
        with pytest.raises(GovernanceError, match="decision_summary"):
            draft_experience_projection(broken)
        bad_result = dict(projection)
        bad_result["outcome"] = {"result": "vibes"}
        with pytest.raises(GovernanceError, match="result"):
            draft_experience_projection(bad_result)


# ---------------------------------------------------------------------------
# Architecture guard: governance stays pure stdlib
# ---------------------------------------------------------------------------

def test_governance_never_imports_heavy_or_forbidden_modules():
    """镜像 P4 家法（test_adapters_never_import_solvers）：governance 运行时
    纯 stdlib（内部只依赖同为纯 stdlib 的 ir.decision_trace），任何
    mlflow/sklearn/ortools/numpy/pandas/torch import 即红。"""
    import re as _re

    pattern = _re.compile(
        r"^\s*(?:import|from)\s+(mlflow|sklearn|ortools|numpy|pandas|torch)\b",
        _re.MULTILINE,
    )
    gov_root = ROOT / "framework" / "src" / "governance"
    offenders = []
    for path in sorted(gov_root.rglob("*.py")):
        hit = pattern.search(path.read_text(encoding="utf-8"))
        if hit:
            offenders.append(f"{path.relative_to(ROOT)}: {hit.group(0).strip()}")
    assert offenders == [], f"governance 包出现禁用 import: {offenders}"


# ---------------------------------------------------------------------------
# Real ANF schema validation
# ---------------------------------------------------------------------------

class TestAnfSchema:
    def test_evidence_fixtures_match_real_anf_envelope(self):
        schema_path = anf_schema_or_none("evidence-envelope.schema.json")
        if schema_path is None:
            pytest.skip(
                f"真实 ANF evidence-envelope schema 不可得：{ANF_SCHEMA_DIR} 不存在且 git clone 失败"
            )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        for evidence in passing_evidence():
            jsonschema.validate(evidence, schema)

    def test_draft_capability_record_matches_real_anf_schema(self, policy):
        schema_path = anf_schema_or_none("capability-record.schema.json")
        if schema_path is None:
            pytest.skip(
                f"真实 ANF capability-record schema 不可得：{ANF_SCHEMA_DIR} 不存在且 git clone 失败"
            )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        decision = evaluate_promotion(
            CAPABILITY_ID, passing_evidence(), policy,
            risk_level="high", scope="organization", now=NOW,
        )
        jsonschema.validate(decision.draft_capability_record, schema)
        assert decision.draft_capability_record["lifecycle_status"] == "draft"


# ---------------------------------------------------------------------------
# Ledger roundtrip + tamper interception
# ---------------------------------------------------------------------------

class TestLedger:
    def test_default_dir_is_outputs_governance(self):
        assert str(DEFAULT_LEDGER_DIR) == str(Path("outputs") / "governance")

    def test_roundtrip_and_idempotent_append(self, policy, tmp_path):
        decision = evaluate_promotion(
            CAPABILITY_ID, passing_evidence(), policy,
            risk_level="medium", scope="project", now=NOW,
        )
        path = append_decision(decision.to_dict(), tmp_path)
        assert path.parent == tmp_path
        assert path.name == f"{decision.decision_id}.jsonl"
        again = append_decision(decision.to_dict(), tmp_path)
        assert again == path
        assert len(path.read_bytes().splitlines()) == 1  # append-only, 幂等
        doc = load_decision(path)
        assert doc["decision_id"] == decision.decision_id
        assert doc["kind"] == "promotion"
        assert doc["decision"]["status"] == "approved_for_draft"
        assert doc["events"][0]["prev_event_id"] == "genesis"
        assert doc["events"][0]["event_id"].startswith("gev-")
        assert doc["events"][0]["recorded_at"]  # 易失审计字段，不入 event_id

    def test_tampered_line_rejected_on_load(self, policy, tmp_path):
        decision = evaluate_promotion(
            CAPABILITY_ID, passing_evidence(), policy,
            risk_level="medium", scope="project", now=NOW,
        )
        path = append_decision(decision.to_dict(), tmp_path)
        lines = path.read_bytes().splitlines()
        event = json.loads(lines[0])
        event["decision"]["capability_id"] = "cap-evil"  # 篡改内容
        path.write_bytes(
            json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        )
        with pytest.raises(GovernanceLedgerTamperError):
            load_decision(path)

    def test_tampered_chain_hash_rejected_on_load(self, policy, tmp_path):
        decision = evaluate_promotion(
            CAPABILITY_ID, passing_evidence(), policy,
            risk_level="medium", scope="project", now=NOW,
        )
        path = append_decision(decision.to_dict(), tmp_path)
        lines = path.read_bytes().splitlines()
        event = json.loads(lines[0])
        # 篡改哈希覆盖字段（status）而不重算 event_id -> 链哈希失配
        event["status"] = "rejected"
        path.write_bytes(
            json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        )
        with pytest.raises(GovernanceLedgerTamperError, match="event_id"):
            load_decision(path)

    def test_volatile_recorded_at_change_is_not_tamper(self, policy, tmp_path):
        """recorded_at 不入 event_id（镜像 P3 created_at 纪律）：只改它不断链。"""
        decision = evaluate_promotion(
            CAPABILITY_ID, passing_evidence(), policy,
            risk_level="medium", scope="project", now=NOW,
        )
        path = append_decision(decision.to_dict(), tmp_path)
        lines = path.read_bytes().splitlines()
        event = json.loads(lines[0])
        event["recorded_at"] = "2099-01-01T00:00:00+00:00"
        path.write_bytes(
            json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        )
        doc = load_decision(path)  # 不抛：易失字段不在链哈希内
        assert doc["decision_id"] == decision.decision_id

    def test_append_conflict_on_diverged_storage(self, policy, tmp_path):
        decision = evaluate_promotion(
            CAPABILITY_ID, passing_evidence(), policy,
            risk_level="medium", scope="project", now=NOW,
        )
        path = append_decision(decision.to_dict(), tmp_path)
        stored = json.loads(path.read_bytes().splitlines()[0])
        stored["decision"]["capability_id"] = "cap-evil"
        path.write_bytes(
            json.dumps(stored, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        )
        with pytest.raises(GovernanceLedgerConflictError):
            append_decision(decision.to_dict(), tmp_path)

    def test_recorded_at_is_volatile_not_in_event_id(self, policy, tmp_path):
        decision = evaluate_promotion(
            CAPABILITY_ID, passing_evidence(), policy,
            risk_level="medium", scope="project", now=NOW,
        )
        path_a = append_decision(decision.to_dict(), tmp_path / "a", recorded_at="2026-01-01T00:00:00+00:00")
        path_b = append_decision(decision.to_dict(), tmp_path / "b", recorded_at="2099-12-31T00:00:00+00:00")
        event_a = json.loads(path_a.read_bytes().splitlines()[0])
        event_b = json.loads(path_b.read_bytes().splitlines()[0])
        assert event_a["event_id"] == event_b["event_id"]
        assert event_a["recorded_at"] != event_b["recorded_at"]
