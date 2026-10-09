"""ExperimentIR contract tests (SPEC-001).

Gate-triggering scenarios are driven from the committed example documents
in schemas/experiment-ir/0.1.0/examples/ so examples and schema can never
drift: every invalid sample is schema-legal (gate-level illegality, not
schema-level) and fails exactly one hard gate.

The validator under test is pure stdlib; jsonschema is only used by the
tests and by load_ir when available (CI installs it).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.src.ir import experiment_ir as eir
from framework.src.ir.experiment_ir import (
    ContentHashMismatch,
    GateResult,
    IRSchemaError,
    authorize_execution,
    canonical_bytes,
    compute_content_hash,
    load_ir,
    run_gates,
)

jsonschema = pytest.importorskip("jsonschema")

SCHEMA_PATH = ROOT / "schemas" / "experiment-ir" / "0.1.0" / "experiment-ir.schema.json"
EXAMPLES_DIR = SCHEMA_PATH.parent / "examples"
VALID_PATH = EXAMPLES_DIR / "valid-minimal.json"

EXPECTED_GATE_ORDER = [
    "G1_TEMPORAL_AVAILABILITY",
    "G2_SPLIT_SEPARATION",
    "G3_METRIC_DEFINITION",
    "G4_BUDGET_BOUNDS",
    "G5_BASELINE_SAME_PROTOCOL",
    "G6_RUN_AUTHORIZATION",
]

# (example file, the only hard gate it must fail)
INVALID_GATE_CASES = [
    ("invalid-g1-time-col.json", "G1_TEMPORAL_AVAILABILITY"),
    ("invalid-g3-direction.json", "G3_METRIC_DEFINITION"),
    ("invalid-g4-budget.json", "G4_BUDGET_BOUNDS"),
    ("invalid-g5-baseline-protocol.json", "G5_BASELINE_SAME_PROTOCOL"),
    ("invalid-g6-authorization.json", "G6_RUN_AUTHORIZATION"),
]


def _schema_validator():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


def _iter_schema_errors(ir):
    return list(_schema_validator().iter_errors(ir))


def _write_ir(tmp_path, ir, name="mutated.json"):
    ir["content_hash"] = compute_content_hash(ir)
    path = tmp_path / name
    path.write_text(json.dumps(ir, ensure_ascii=False), encoding="utf-8")
    return path


# ---------- valid minimal document ----------

def test_valid_minimal_passes_schema():
    ir = json.loads(VALID_PATH.read_text(encoding="utf-8"))
    assert _iter_schema_errors(ir) == []


def test_valid_minimal_loads_and_passes_all_gates():
    ir = load_ir(VALID_PATH)
    gates = run_gates(ir)
    assert [g.gate_id for g in gates] == EXPECTED_GATE_ORDER
    # valid-minimal uses time_based: G1/G2 must actively pass, not skip
    assert gates[0].status == "pass"
    assert gates[1].status == "pass"
    assert all(g.status == "pass" for g in gates)
    verdict = authorize_execution(ir)
    assert verdict.ok and not verdict.blocked
    assert verdict.failed_gates == () and verdict.reasons == ()


def test_g1_not_applicable_for_non_temporal_strategy():
    """stratified has no temporal requirements: G1 reports not_applicable
    and never blocks."""
    ir = json.loads(VALID_PATH.read_text(encoding="utf-8"))
    protocol = {"strategy": "stratified", "n_folds": 5}
    ir["validation_protocol_ref"] = dict(protocol)
    ir["baseline_ref"]["protocol_ref"] = dict(protocol)
    # even with label_cutoff entirely missing, a non-temporal strategy
    # must not trip G1
    ir["dataset_snapshot_ref"].pop("label_cutoff", None)
    gates = run_gates(ir)
    by_id = {g.gate_id: g for g in gates}
    assert by_id["G1_TEMPORAL_AVAILABILITY"].status == "not_applicable"
    assert authorize_execution(ir).ok


def test_valid_minimal_content_hash_consistent():
    ir = json.loads(VALID_PATH.read_text(encoding="utf-8"))
    assert compute_content_hash(ir) == ir["content_hash"]


@pytest.mark.parametrize("filename,gate_id", [
    ("invalid-g1-time-col.json", "G1_TEMPORAL_AVAILABILITY"),
    ("invalid-g3-direction.json", "G3_METRIC_DEFINITION"),
    ("invalid-g4-budget.json", "G4_BUDGET_BOUNDS"),
    ("invalid-g5-baseline-protocol.json", "G5_BASELINE_SAME_PROTOCOL"),
    ("invalid-g6-authorization.json", "G6_RUN_AUTHORIZATION"),
])
def test_invalid_sample_is_schema_legal_but_gate_blocked(filename, gate_id):
    """Invalid examples must pass the JSON schema (gate-level illegality),
    load cleanly (hash intact) and fail exactly their target gate."""
    path = EXAMPLES_DIR / filename
    ir = json.loads(path.read_text(encoding="utf-8"))
    assert _iter_schema_errors(ir) == [], f"{filename} must be schema-legal"
    loaded = load_ir(path)  # no ContentHashMismatch
    verdict = authorize_execution(loaded)
    assert verdict.blocked
    assert list(verdict.failed_gates) == [gate_id]
    assert verdict.reasons[0].startswith(gate_id)
    blocking = [g for g in run_gates(loaded) if g.status != "pass"]
    assert [g.gate_id for g in blocking] == [gate_id]


# ---------- content-hash integrity ----------

def test_tampered_body_rejected(tmp_path):
    ir = json.loads(VALID_PATH.read_text(encoding="utf-8"))
    ir["objective"] += "（被篡改）"
    path = tmp_path / "tampered-body.json"
    path.write_text(json.dumps(ir, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ContentHashMismatch):
        load_ir(path)


def test_tampered_hash_field_rejected(tmp_path):
    ir = json.loads(VALID_PATH.read_text(encoding="utf-8"))
    ir["content_hash"] = "0" * 64
    path = tmp_path / "tampered-hash.json"
    path.write_text(json.dumps(ir, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ContentHashMismatch):
        load_ir(path)


def test_content_hash_excludes_hash_field_itself():
    ir = json.loads(VALID_PATH.read_text(encoding="utf-8"))
    with_hash = dict(ir, content_hash="deadbeef" * 8)
    without_hash = {k: v for k, v in ir.items() if k != "content_hash"}
    assert compute_content_hash(with_hash) == compute_content_hash(without_hash)


# ---------- revision discipline ----------

def test_supersedes_chain(tmp_path):
    rev1 = json.loads(VALID_PATH.read_text(encoding="utf-8"))
    assert "supersedes" not in rev1  # optional field absent on first revision

    rev2 = json.loads(json.dumps(rev1))  # deepcopy
    rev2["objective"] += "；v2：加入门店类别交叉特征"
    rev2["revision"] = 2
    rev2["supersedes"] = rev1["content_hash"]
    path2 = _write_ir(tmp_path, rev2, "rev2.json")

    loaded2 = load_ir(path2)
    assert loaded2["revision"] == 2
    assert loaded2["supersedes"] == rev1["content_hash"]
    assert loaded2["content_hash"] != rev1["content_hash"]
    assert authorize_execution(loaded2).ok

    rev3 = json.loads(json.dumps(loaded2))
    rev3["revision"] = 3
    rev3["supersedes"] = loaded2["content_hash"]
    path3 = _write_ir(tmp_path, rev3, "rev3.json")
    loaded3 = load_ir(path3)
    assert loaded3["supersedes"] == loaded2["content_hash"]
    assert authorize_execution(loaded3).ok


# ---------- canonicalization ----------

def test_canonical_bytes_deterministic():
    ir = json.loads(VALID_PATH.read_text(encoding="utf-8"))
    reordered = {k: ir[k] for k in reversed(list(ir))}
    reordered["budget"] = {k: ir["budget"][k] for k in reversed(list(ir["budget"]))}
    assert canonical_bytes(reordered) == canonical_bytes(ir)
    assert canonical_bytes(ir) == canonical_bytes(
        json.loads(canonical_bytes(ir).decode("utf-8")))
    assert b"content_hash" in canonical_bytes(ir)


# ---------- schema-level rejection ----------

def test_unknown_top_level_field_rejected(tmp_path):
    ir = json.loads(VALID_PATH.read_text(encoding="utf-8"))
    ir["extra_field"] = 1
    path = _write_ir(tmp_path, ir, "unknown-field.json")
    with pytest.raises(IRSchemaError):
        load_ir(path)


def test_unparseable_datetime_rejected(tmp_path):
    """format is only an annotation in draft 2020-12, so datetime strictness
    must not depend on jsonschema extras."""
    ir = json.loads(VALID_PATH.read_text(encoding="utf-8"))
    ir["created_at"] = "2026-13-40T99:00:00Z"
    path = _write_ir(tmp_path, ir, "bad-datetime.json")
    with pytest.raises(IRSchemaError):
        load_ir(path)


def test_fallback_validator_used_without_jsonschema(tmp_path, monkeypatch):
    """Without jsonschema, the stdlib minimal check still accepts the valid
    document and still rejects missing/unknown fields."""
    monkeypatch.setattr(eir, "jsonschema", None)
    loaded = load_ir(VALID_PATH)
    assert loaded["experiment_id"] == json.loads(
        VALID_PATH.read_text(encoding="utf-8"))["experiment_id"]

    missing = json.loads(VALID_PATH.read_text(encoding="utf-8"))
    del missing["budget"]
    with pytest.raises(IRSchemaError, match="budget"):
        load_ir(_write_ir(tmp_path, missing, "missing-budget.json"))

    unknown = json.loads(VALID_PATH.read_text(encoding="utf-8"))
    unknown["surprise"] = True
    with pytest.raises(IRSchemaError, match="surprise"):
        load_ir(_write_ir(tmp_path, unknown, "unknown-fallback.json"))


def test_gate_result_rejects_unknown_status():
    with pytest.raises(ValueError):
        GateResult("G1_TEMPORAL_AVAILABILITY", "skipped", "bogus status")


# ---------- runtime dependency hygiene ----------

def test_runtime_module_imports_no_mlflow_or_sklearn():
    code = (
        "import sys;"
        "import framework.src.ir.experiment_ir;"
        "bad = [n for n in ('mlflow', 'sklearn') if n in sys.modules];"
        "print(''.join(bad))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""
