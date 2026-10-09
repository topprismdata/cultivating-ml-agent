"""DecisionTrace contract tests (SPEC-004, ADR-002).

Covers: lifecycle transitions, separation of duties, mandatory event fields,
immutable correction semantics, hash-chain tamper evidence across the JSONL
ledger, deterministic sha256 identity (no uuid4, created_at excluded), and the
lossy ANF experience-record projection validated against the real upstream
schema (agent-nurture-framework), skipped with a stated reason if the schema
is unavailable.
"""
import copy
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.src.ir import decision_trace as dt

jsonschema = pytest.importorskip("jsonschema")

SCHEMA_PATH = ROOT / "schemas" / "decision-trace" / "0.1.0" / "decision-trace.schema.json"
EXAMPLES_DIR = SCHEMA_PATH.parent / "examples"

ANF_SCHEMA_PATH = Path(
    "/tmp/anf-skilltester-check/agent-nurture-framework/schemas/experience-record.schema.json"
)
ANF_REPO_URL = "https://github.com/topprismdata/agent-nurture-framework"

#: The nine ANF-required fields must all be present and non-empty.
ANF_REQUIRED_FIELDS = (
    "record_id",
    "task",
    "decision",
    "decision_summary",
    "action",
    "outcome",
    "classification",
    "trust_level",
    "timestamp",
)

T0 = datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc)


def fixed_clock():
    """Injected clock: 10-minute steps from T0 (created_at never hits hashes)."""
    state = {"n": 0}

    def clock():
        state["n"] += 1
        return T0 + timedelta(minutes=10 * state["n"])

    return clock


def make_outcome(**overrides):
    """Shared-contract DecisionOutcome (B-line shape) for testing."""
    outcome = {
        "intent_ref": "intent://task/titanic-2026",
        "state_snapshot_ref": {
            "ir_content_hash": "a" * 64,
            "data_sha256": "b" * 64,
            "code_version": "e756797",
        },
        "alternatives": [
            {
                "candidate_id": "cand-lgbm",
                "family": "lightgbm",
                "params": {"n_estimators": 300, "learning_rate": 0.05},
                "metric_value": 0.861,
            },
            {
                "candidate_id": "cand-xgb",
                "family": "xgboost",
                "params": {"n_estimators": 300},
                "metric_value": 0.858,
            },
        ],
        "constraint_check": [
            {"gate_id": "G3_METRIC_DEFINITION", "status": "pass"},
            {"gate_id": "G5_BASELINE_SAME_PROTOCOL", "status": "pass"},
        ],
        "baseline_ref": {
            "baseline_id": "baseline-logreg",
            "protocol_ref": {"strategy": "stratified", "folds": 5},
            "score": 0.85,
        },
        "selected_option": {
            "candidate_id": "cand-lgbm",
            "family": "lightgbm",
            "params": {"n_estimators": 300, "learning_rate": 0.05},
            "metric_value": 0.861,
        },
        "rejected_options_and_reasons": [
            {
                "candidate_id": "cand-xgb",
                "reason": "lost_to_selected",
                "detail": "0.858 < 0.861 同协议 5 折均值",
            }
        ],
        "uncertainty": {"metric_std": 0.004, "margin": 0.011, "note": "5 折标准差"},
        "decision_actor": "agent-ml",
        "authorization_ref": "auth://g6/2026-10-09/001",
        "evidence_refs": ["evidence://replay/run-001"],
        "metric_name": "auc",
        "metric_direction": "maximize",
    }
    outcome.update(overrides)
    return outcome


def decision_at(status, *, outcome=None, clock=None):
    """Build a decision advanced to the requested status."""
    clock = clock or fixed_clock()
    doc = dt.new_decision(outcome or make_outcome(), clock=clock)
    if status == "proposed":
        return doc
    dt.append_event(doc, "approved", actor="human-reviewer", clock=clock)
    if status == "approved":
        return doc
    if status == "rejected":
        dt.append_event(doc, "rejected", actor="human-reviewer", reason="预算不足", clock=clock)
        return doc
    dt.append_event(
        doc, "executed", actor="agent-ml", execution_ref="manifest://run-001", clock=clock
    )
    if status == "executed":
        return doc
    dt.append_event(
        doc,
        "evaluated",
        actor="agent-ml",
        evidence_refs=["evidence://replay/run-001/envelope"],
        outcome_refs=["outcome://run-001"],
        clock=clock,
    )
    return doc


def _anf_schema_or_none():
    """Locate (or fetch) the real upstream ANF experience-record schema."""
    if ANF_SCHEMA_PATH.exists():
        return ANF_SCHEMA_PATH
    subprocess.run(
        ["git", "clone", "--depth", "1", ANF_REPO_URL, str(ANF_SCHEMA_PATH.parents[1])],
        check=False,
        capture_output=True,
    )
    if ANF_SCHEMA_PATH.exists():
        return ANF_SCHEMA_PATH
    return None


# ---------------------------------------------------------------------------
# Schema and committed examples
# ---------------------------------------------------------------------------

def test_committed_examples_are_schema_valid_and_chain_valid():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    names = {p.name for p in EXAMPLES_DIR.glob("*.json")}
    assert {
        "full-lifecycle.json",
        "rejected.json",
        "correction.json",
    } <= names, f"缺少示例文件: {sorted(names)}"
    for path in sorted(EXAMPLES_DIR.glob("*.json")):
        example = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.validate(example, schema)
        dt.verify_decision(example)


def test_decision_document_matches_schema_at_every_stage():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    clock = fixed_clock()
    doc = dt.new_decision(make_outcome(), clock=clock)
    jsonschema.validate(doc, schema)
    dt.append_event(doc, "approved", actor="human-reviewer", clock=clock)
    jsonschema.validate(doc, schema)
    dt.append_event(doc, "executed", actor="agent-ml", execution_ref="manifest://run-001", clock=clock)
    jsonschema.validate(doc, schema)
    dt.append_event(
        doc, "evaluated", actor="agent-ml", evidence_refs=["evidence://x"], clock=clock
    )
    jsonschema.validate(doc, schema)
    dt.correct(doc, doc["events"][0]["event_id"], "更正备注", actor="auditor", clock=clock)
    jsonschema.validate(doc, schema)


# ---------------------------------------------------------------------------
# Lifecycle and state machine
# ---------------------------------------------------------------------------

def test_full_lifecycle_happy_path():
    clock = fixed_clock()
    doc = dt.new_decision(make_outcome(), clock=clock)
    assert doc["status"] == "proposed"
    assert doc["events"][0]["event_type"] == "proposed"
    assert doc["events"][0]["prev_event_id"] == dt.GENESIS_PREV_EVENT_ID
    assert doc["decision_actor"] == doc["events"][0]["actor"]
    dt.append_event(doc, "approved", actor="human-reviewer", clock=clock)
    assert doc["status"] == "approved"
    dt.append_event(doc, "executed", actor="agent-ml", execution_ref="manifest://run-001", clock=clock)
    assert doc["status"] == "executed"
    assert doc["execution_ref"] == "manifest://run-001"
    dt.append_event(
        doc,
        "evaluated",
        actor="agent-ml",
        evidence_refs=["evidence://replay/run-001/envelope"],
        outcome_refs=["outcome://run-001"],
        clock=clock,
    )
    assert doc["status"] == "evaluated"
    assert doc["outcome_refs"] == ["outcome://run-001"]
    # forward linkage + statuses mirror event_type for lifecycle events
    events = doc["events"]
    assert [e["event_type"] for e in events] == ["proposed", "approved", "executed", "evaluated"]
    assert [e["status"] for e in events] == ["proposed", "approved", "executed", "evaluated"]
    for prev, nxt in zip(events, events[1:]):
        assert nxt["prev_event_id"] == prev["event_id"]
    dt.verify_decision(doc)


@pytest.mark.parametrize(
    "current, illegal_event",
    [
        ("proposed", "executed"),
        ("proposed", "evaluated"),
        ("proposed", "correction"),
        ("approved", "approved"),
        ("approved", "evaluated"),
        ("approved", "correction"),
        ("executed", "executed"),
        ("executed", "rejected"),
        ("executed", "correction"),
        ("evaluated", "approved"),
        ("evaluated", "rejected"),
        ("evaluated", "correction"),
        ("rejected", "approved"),
        ("rejected", "executed"),
        ("rejected", "correction"),
    ],
)
def test_illegal_transitions_rejected(current, illegal_event):
    doc = decision_at(current)
    kwargs = {
        "actor": "someone-else",
        "execution_ref": "manifest://x",
        "evidence_refs": ["evidence://x"],
        "reason": "占位",
    }
    with pytest.raises(dt.IllegalTransition):
        dt.append_event(doc, illegal_event, clock=fixed_clock(), **kwargs)
    assert doc["status"] == current  # nothing moved
    assert len(doc["events"]) == {
        "proposed": 1,
        "approved": 2,
        "executed": 3,
        "evaluated": 4,
        "rejected": 3,
    }[current]


def test_alternate_rejection_path_approved_then_rejected():
    doc = decision_at("approved")
    dt.append_event(doc, "rejected", actor="human-reviewer", reason="G4 预算越界", clock=fixed_clock())
    assert doc["status"] == "rejected"
    assert doc["events"][-1]["reason"] == "G4 预算越界"
    dt.verify_decision(doc)


# ---------------------------------------------------------------------------
# Separation of duties and mandatory fields
# ---------------------------------------------------------------------------

def test_self_approval_rejected_four_eyes():
    doc = dt.new_decision(make_outcome(), clock=fixed_clock())
    with pytest.raises(dt.SeparationOfDutiesError):
        dt.append_event(doc, "approved", actor=doc["decision_actor"], clock=fixed_clock())
    assert doc["status"] == "proposed"
    # a different actor may approve
    dt.append_event(doc, "approved", actor="human-reviewer", clock=fixed_clock())
    assert doc["status"] == "approved"


def test_executed_requires_execution_ref():
    doc = decision_at("approved")
    with pytest.raises(dt.MissingEventFieldError, match="execution_ref"):
        dt.append_event(doc, "executed", actor="agent-ml", clock=fixed_clock())
    with pytest.raises(dt.MissingEventFieldError):
        dt.append_event(doc, "executed", actor="agent-ml", execution_ref="  ", clock=fixed_clock())


def test_evaluated_requires_nonempty_evidence_refs():
    doc = decision_at("executed")
    with pytest.raises(dt.MissingEventFieldError, match="evidence_refs"):
        dt.append_event(doc, "evaluated", actor="agent-ml", clock=fixed_clock())
    with pytest.raises(dt.MissingEventFieldError):
        dt.append_event(doc, "evaluated", actor="agent-ml", evidence_refs=[], clock=fixed_clock())
    with pytest.raises(dt.MissingEventFieldError):
        dt.append_event(
            doc, "evaluated", actor="agent-ml", evidence_refs=["", "  "], clock=fixed_clock()
        )


def test_rejected_requires_reason():
    doc = decision_at("proposed")
    with pytest.raises(dt.MissingEventFieldError, match="reason"):
        dt.append_event(doc, "rejected", actor="human-reviewer", clock=fixed_clock())
    with pytest.raises(dt.MissingEventFieldError):
        dt.append_event(doc, "rejected", actor="human-reviewer", reason="", clock=fixed_clock())


def test_unknown_event_fields_rejected():
    doc = decision_at("approved")
    with pytest.raises(TypeError, match="bogus"):
        dt.append_event(
            doc, "executed", actor="agent-ml", execution_ref="m://1", bogus="x", clock=fixed_clock()
        )


# ---------------------------------------------------------------------------
# Correction semantics
# ---------------------------------------------------------------------------

def test_correction_appends_without_rewriting_history():
    doc = decision_at("evaluated")
    snapshot = copy.deepcopy(doc["events"])
    status_before = doc["status"]
    target_id = doc["events"][1]["event_id"]  # correct the approval event
    doc = dt.correct(
        doc,
        target_id,
        reason="批准引用的 authorization_ref 有误",
        actor="auditor",
        payload={"authorization_ref": "auth://g6/2026-10-09/002"},
        clock=fixed_clock(),
    )
    assert doc["status"] == status_before  # correction never changes status
    assert doc["events"][:-1] == snapshot  # history byte-identical
    correction = doc["events"][-1]
    assert correction["event_type"] == "correction"
    assert correction["correction_of"] == target_id
    assert correction["reason"] == "批准引用的 authorization_ref 有误"
    assert correction["status"] == status_before
    dt.verify_decision(doc)


def test_correction_of_unknown_event_rejected():
    doc = decision_at("evaluated")
    with pytest.raises(dt.UnknownEventError):
        dt.correct(doc, "ev-" + "0" * 64, "理由", actor="auditor", clock=fixed_clock())
    with pytest.raises(dt.MissingEventFieldError):
        dt.correct(doc, doc["events"][0]["event_id"], "", actor="auditor", clock=fixed_clock())


def test_correction_survives_save_load_roundtrip(tmp_path):
    doc = decision_at("evaluated")
    doc = dt.correct(
        doc,
        doc["events"][0]["event_id"],
        reason="更正 uncertainty 备注",
        actor="auditor",
        payload={"uncertainty": {"metric_std": 0.004, "margin": 0.011, "note": "更正后"}},
        clock=fixed_clock(),
    )
    path = dt.save_decision(doc, tmp_path)
    loaded = dt.load_decision(path)
    assert loaded == doc
    assert loaded["events"][-1]["event_type"] == "correction"
    assert loaded["status"] == "evaluated"


# ---------------------------------------------------------------------------
# JSONL ledger: append-only + tamper evidence
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_and_incremental_append(tmp_path):
    doc = dt.new_decision(make_outcome(), clock=fixed_clock())
    path = dt.save_decision(doc, tmp_path)
    assert path.read_text(encoding="utf-8").count("\n") == 1
    dt.append_event(doc, "approved", actor="human-reviewer", clock=fixed_clock())
    dt.save_decision(doc, tmp_path)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    dt.append_event(doc, "executed", actor="agent-ml", execution_ref="manifest://run-001", clock=fixed_clock())
    dt.save_decision(doc, tmp_path)
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 3
    # the first two stored lines were never rewritten
    first_two = json.loads(raw_lines[0]), json.loads(raw_lines[1])
    loaded = dt.load_decision(path)
    assert loaded == doc
    assert loaded["events"][0] == first_two[0]
    assert loaded["events"][1] == first_two[1]
    # one event per line
    for line in raw_lines:
        assert "event_id" in json.loads(line)


def test_tampered_stored_line_rejected_on_load(tmp_path):
    doc = decision_at("evaluated")
    path = dt.save_decision(doc, tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[2])  # executed event
    event["execution_ref"] = "manifest://evil-run"
    lines[2] = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(dt.ChainHashMismatch):
        dt.load_decision(path)


def test_tampered_first_line_payload_rejected_on_load(tmp_path):
    doc = decision_at("proposed")
    path = dt.save_decision(doc, tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[0])
    event["payload"]["selected_option"]["metric_value"] = 0.999
    lines[0] = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(dt.ChainHashMismatch):
        dt.load_decision(path)


def test_load_detects_broken_forward_link(tmp_path):
    doc = decision_at("approved")
    path = dt.save_decision(doc, tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    second = json.loads(lines[1])
    second["prev_event_id"] = "ev-" + "f" * 64  # relink to a non-existent event
    lines[1] = json.dumps(second, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(dt.ChainHashMismatch, match="prev_event_id"):
        dt.load_decision(path)


def test_save_refuses_overwriting_modified_existing_lines(tmp_path):
    doc = decision_at("approved")
    path = dt.save_decision(doc, tmp_path)
    before = path.read_bytes()
    # a rival ledger with the same decision_id but genuinely different history
    clock = fixed_clock()
    rival = dt.new_decision(make_outcome(), clock=clock)
    dt.append_event(rival, "approved", actor="different-reviewer", clock=clock)
    assert rival["decision_id"] == doc["decision_id"]
    with pytest.raises(dt.LedgerWriteError, match="已被改动"):
        dt.save_decision(rival, tmp_path)
    assert path.read_bytes() == before  # file untouched


def test_save_refuses_when_file_is_ahead_of_memory(tmp_path):
    doc = decision_at("executed")
    path = dt.save_decision(doc, tmp_path)
    stale = decision_at("proposed", outcome=make_outcome())  # same decision_id, 1 event
    with pytest.raises(dt.LedgerWriteError):
        dt.save_decision(stale, tmp_path)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


# ---------------------------------------------------------------------------
# Deterministic sha256 identity
# ---------------------------------------------------------------------------

def test_decision_id_deterministic_and_content_sensitive():
    clock_a = fixed_clock()
    clock_b = lambda: datetime(1999, 1, 1, tzinfo=timezone.utc)  # noqa: E731
    d1 = dt.new_decision(make_outcome(), clock=clock_a)
    d2 = dt.new_decision(make_outcome(), clock=clock_b)
    assert d1["decision_id"] == d2["decision_id"]
    assert re.fullmatch(r"dt-[0-9a-f]{12}", d1["decision_id"])
    # created_at differs, event ids do not: created_at is out of the hash
    assert d1["created_at"] != d2["created_at"]
    assert d1["events"][0]["event_id"] == d2["events"][0]["event_id"]
    assert re.fullmatch(r"ev-[0-9a-f]{64}", d1["events"][0]["event_id"])
    # different selection -> different id
    changed = make_outcome()
    changed["selected_option"] = dict(changed["selected_option"], metric_value=0.77)
    d3 = dt.new_decision(changed, clock=clock_a)
    assert d3["decision_id"] != d1["decision_id"]


def test_created_at_excluded_from_chain_hash():
    doc = decision_at("evaluated")
    mutated = copy.deepcopy(doc)
    for event in mutated["events"]:
        event["created_at"] = "2000-01-01T00:00:00+00:00"
    dt.verify_decision(mutated)  # must not raise: created_at never hashed


def test_module_purity_no_uuid_no_mlflow_no_sklearn():
    source = (ROOT / "framework" / "src" / "ir" / "decision_trace.py").read_text(encoding="utf-8")
    for forbidden in (
        "import uuid",
        "from uuid",
        "uuid4(",
        "import mlflow",
        "from mlflow",
        "import sklearn",
        "from sklearn",
        "import jsonschema",
        "from jsonschema",
    ):
        assert forbidden not in source, f"decision_trace.py 不得包含 {forbidden!r}"


# ---------------------------------------------------------------------------
# Shared-contract entry point
# ---------------------------------------------------------------------------

def test_record_from_decision_outcome_maps_contract_fields():
    clock = fixed_clock()
    outcome = make_outcome()
    doc = dt.record_from_decision_outcome(outcome, ir_content_hash="a" * 64, clock=clock)
    assert doc["decision_id"] == dt.new_decision(outcome, clock=clock)["decision_id"]
    assert doc["state_snapshot_ref"]["ir_content_hash"] == "a" * 64
    assert doc["intent_ref"] == outcome["intent_ref"]
    assert doc["selected_option"] == outcome["selected_option"]
    assert "metric_direction" not in doc  # payload-only, not top level
    assert "metric_name" not in doc
    payload = doc["events"][0]["payload"]
    assert payload["metric_name"] == "auc"
    assert payload["metric_direction"] == "maximize"


def test_record_from_decision_outcome_missing_field_raises_keyerror():
    for missing in dt.__dict__["_OUTCOME_FIELDS"]:
        outcome = make_outcome()
        del outcome[missing]
        with pytest.raises(KeyError) as excinfo:
            dt.record_from_decision_outcome(outcome, ir_content_hash="c" * 64)
        assert missing in str(excinfo.value)


def test_record_from_decision_outcome_rejects_hash_conflict():
    outcome = make_outcome()
    outcome["state_snapshot_ref"]["ir_content_hash"] = "d" * 64
    with pytest.raises(ValueError, match="ir_content_hash"):
        dt.record_from_decision_outcome(outcome, ir_content_hash="c" * 64)


# ---------------------------------------------------------------------------
# ANF projection (ADR-002)
# ---------------------------------------------------------------------------

def test_projection_record_id_matches_anf_pattern_at_every_status():
    for status in ("proposed", "approved", "executed", "evaluated", "rejected"):
        record = dt.project_to_anf(decision_at(status))
        assert re.fullmatch(dt.ANF_RECORD_ID_PATTERN, record["record_id"]), record["record_id"]


def test_projection_canonical_id_and_mechanical_record_id():
    doc = decision_at("evaluated")
    record = dt.project_to_anf(doc)
    did = doc["decision_id"]
    assert record["context"] == f"decision_trace_ref=er-{did}@evaluated"
    assert record["record_id"] == f"exp-er-{did}-evaluated"
    assert record["action"] == "execute_experiment(cand-lgbm)"
    assert record["execution_trace_refs"] == ["manifest://run-001"]
    assert record["classification"] == "internal"
    assert record["trust_level"] == "verified_execution"
    assert record["timestamp"] == doc["created_at"]
    # append-only per status: same decision at another status yields a sibling record
    proposed_record = dt.project_to_anf(decision_at("proposed"))
    assert proposed_record["record_id"] == f"exp-er-{did}-proposed"
    assert proposed_record["trust_level"] == "agent_inference"
    assert proposed_record["context"].startswith(f"decision_trace_ref=er-{did}@")


def test_projection_nine_required_fields_nonempty():
    for status in ("proposed", "evaluated", "rejected"):
        record = dt.project_to_anf(decision_at(status))
        for field in ANF_REQUIRED_FIELDS:
            assert field in record, f"{status}: 缺少 {field}"
            value = record[field]
            if isinstance(value, str):
                assert value.strip(), f"{status}: {field} 为空"
            else:
                assert value, f"{status}: {field} 为空"
        assert record["outcome"]["result"] in ("success", "failure", "partial")
        assert record["decision_summary"], "decision_summary 为空"


def test_projection_outcome_result_mapping():
    # outcome_refs present -> success
    record = dt.project_to_anf(decision_at("evaluated"))
    assert record["outcome"]["result"] == "success"
    assert record["outcome"]["metric_name"] == "auc"
    assert record["outcome"]["metric_value"] == 0.861
    # no outcome_refs but evidence_refs -> partial
    proposed = dt.new_decision(make_outcome(), clock=fixed_clock())
    record2 = dt.project_to_anf(proposed)
    assert record2["outcome"]["result"] == "partial"
    # neither -> failure
    bare = make_outcome()
    bare["evidence_refs"] = []
    record3 = dt.project_to_anf(dt.new_decision(bare, clock=fixed_clock()))
    assert record3["outcome"]["result"] == "failure"


def test_projection_rejected_decision_maps_rejection_reasons():
    doc = decision_at("rejected")
    record = dt.project_to_anf(doc)
    assert record["trust_level"] == "agent_inference"
    assert "rejected" in record["decision_summary"]


def test_projection_deterministic():
    doc = decision_at("evaluated")
    assert dt.project_to_anf(doc) == dt.project_to_anf(copy.deepcopy(doc))


def test_projection_valid_against_real_anf_schema():
    schema_path = _anf_schema_or_none()
    if schema_path is None:
        pytest.skip(
            "真实 ANF experience-record schema 不可得："
            f"{ANF_SCHEMA_PATH} 不存在且 git clone {ANF_REPO_URL} 失败"
        )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    for status in ("proposed", "executed", "evaluated", "rejected"):
        record = dt.project_to_anf(decision_at(status))
        jsonschema.validate(record, schema)


def test_projection_decision_summary_truncated_to_anf_limit():
    outcome = make_outcome(
        rejected_options_and_reasons=[
            {
                "candidate_id": f"cand-{i:03d}",
                "reason": "worse_than_baseline",
                "detail": "细节" * 120,
            }
            for i in range(40)
        ],
    )
    doc = dt.new_decision(outcome, clock=fixed_clock())
    record = dt.project_to_anf(doc)
    assert len(record["decision_summary"]) <= 2000
