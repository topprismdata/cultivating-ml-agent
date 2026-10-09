"""SPEC-003 replay & evidence reproducibility tests.

Proves the G2 replay gate on the two committed replay cases: same
ExperimentIR + same data bytes + same code version + same seed must give a
byte-identical canonical manifest hash and bitwise-equal metrics.

Committed golden files (replays/<case>/golden/manifest.canonical.json)
regression-pin every canonical byte except ``code_version`` (kept as an
``@CODE_VERSION@`` placeholder — pinning a commit hash would invalidate the
golden on every commit). Fixture regeneration, tamper rejection (data and
IR), gate-blocked execution, ANF evidence required fields and the
compare_replays tolerance path are all covered here.
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.src.ir.experiment_ir import (
    ContentHashMismatch,
    canonical_bytes,
    load_ir,
)
from framework.src.ir.runner import (
    DataHashMismatch,
    ExecutionBlocked,
    RunResult,
    canonical_sha256,
    compare_replays,
    evidence_content_hash,
    execute_experiment,
    manifest_canonical_hash,
)

REPLAY_CASES = [ROOT / "replays" / "s6e5-style", ROOT / "replays" / "store-sales-style"]
CASE_IDS = [case.name for case in REPLAY_CASES]
EXAMPLES_DIR = ROOT / "schemas" / "experiment-ir" / "0.1.0" / "examples"

#: ANF evidence-envelope required fields (plan baseline v0.1 §9 / ADR-002:
#: evaluation results land as evidence-envelope, ML Profile sets no
#: EvaluationResult entity).
ANF_REQUIRED_FIELDS = {
    "evidence_id",
    "capability_id",
    "skill_id",
    "evidence_type",
    "source_system",
    "measurement_protocol",
    "protocol_version",
    "metric_name",
    "metric_value",
    "metric_version",
    "executor",
    "provenance_chain_id",
    "task_id",
    "artifact_refs",
    "timestamp",
}


def _run_case(case_dir: Path, project_root: Path, seed: int = 42) -> RunResult:
    """Execute a replay case with data resolved against the repo (the
    committed fixture) and outputs written under ``project_root``."""
    ir = load_ir(case_dir / "experiment.ir.json")
    return execute_experiment(ir, project_root, seed=seed, data_root=ROOT)


def _golden(case_dir: Path) -> dict:
    return json.loads(
        (case_dir / "golden" / "manifest.canonical.json").read_text(encoding="utf-8")
    )


def _masked(canonical: dict) -> dict:
    """Canonical with code_version masked for golden comparison."""
    body = dict(canonical)
    body["code_version"] = "@CODE_VERSION@"
    return body


# ---------- G2 replay gate: double run is byte-identical ----------

@pytest.mark.parametrize("case_dir", REPLAY_CASES, ids=CASE_IDS)
def test_double_run_identical(case_dir: Path, tmp_path: Path):
    run_a = _run_case(case_dir, tmp_path / "a")
    run_b = _run_case(case_dir, tmp_path / "b")

    assert run_a.canonical_sha256 == run_b.canonical_sha256
    assert run_a.manifest["canonical"] == run_b.manifest["canonical"]
    assert run_a.canonical["metrics"] == run_b.canonical["metrics"]
    assert run_a.oof_sha256 == run_b.oof_sha256
    assert evidence_content_hash(run_a.evidence) == evidence_content_hash(run_b.evidence)
    # Volatile zones may differ (wall clock); they never enter any hash.
    assert run_a.manifest["volatile"] != run_b.manifest["volatile"]

    verdict = compare_replays(run_a, run_b)
    assert verdict.canonical_hash_equal
    assert verdict.metrics_within_tolerance
    assert verdict.data_sha256_equal
    assert verdict.ir_content_hash_equal
    assert verdict.seed_equal
    assert verdict.code_version_equal
    assert verdict.identical


@pytest.mark.parametrize("case_dir", REPLAY_CASES, ids=CASE_IDS)
def test_golden_regression(case_dir: Path, tmp_path: Path):
    run = _run_case(case_dir, tmp_path)
    golden = _golden(case_dir)

    # code_version is git-bound; every other canonical byte is frozen.
    assert re.fullmatch(r"(?:[0-9a-f]{40}|unknown)", run.canonical["code_version"])
    assert _masked(run.canonical) == golden["canonical"]
    assert canonical_sha256(golden["canonical"]) == golden["canonical_sha256"]

    assert run.canonical["metrics"] == golden["metrics"]

    stable = {k: v for k, v in run.evidence.items() if k != "timestamp"}
    assert stable == golden["evidence_stable"]
    assert evidence_content_hash(run.evidence) == golden["evidence_content_sha256"]


@pytest.mark.parametrize("case_dir", REPLAY_CASES, ids=CASE_IDS)
def test_oof_artifact_layout(case_dir: Path, tmp_path: Path):
    run = _run_case(case_dir, tmp_path)
    assert run.oof_path.exists()
    assert run.oof_path == tmp_path / "outputs" / "oof" / f"{run.ir['experiment_id']}.csv"
    assert run.evidence["artifact_refs"] == [
        {"path": run.oof_path.relative_to(tmp_path).as_posix(), "sha256": run.oof_sha256}
    ]
    assert (
        hashlib.sha256(run.oof_path.read_bytes()).hexdigest() == run.oof_sha256
    )

    frame = pd.read_csv(run.oof_path)
    assert list(frame.columns) == ["id", "target", "oof_pred"]
    assert frame["target"].notna().all()
    scored = frame["oof_pred"].notna()
    val_size = run.canonical["fold_sizes"][0]["val"]
    train_size = run.canonical["fold_sizes"][0]["train"]
    assert int(scored.sum()) == val_size
    assert int((~scored).sum()) == train_size
    # time_based forward split: no temporal leakage. Multi-series fixtures
    # interleave per-store blocks, so positions are not a contiguous tail;
    # every scored row must be strictly after every unscored row in time.
    fixture = pd.read_csv(case_dir / "fixtures" / "train.csv")
    dates = dict(zip(fixture["id"].to_numpy(), fixture["date"].to_numpy()))
    max_train_date = max(dates[i] for i in frame.loc[~scored, "id"])
    min_val_date = min(dates[i] for i in frame.loc[scored, "id"])
    assert max_train_date < min_val_date


# ---------- tamper rejection ----------

def test_fixture_tamper_raises_data_hash_mismatch(tmp_path: Path):
    case_dir = REPLAY_CASES[0]
    data_root = tmp_path / "data"
    dest = data_root / "replays" / "s6e5-style" / "fixtures"
    dest.mkdir(parents=True)
    raw = (case_dir / "fixtures" / "train.csv").read_bytes()

    marker = raw.index(b"\n") + 1  # flip one byte on the first data line
    tampered = raw[: marker + 2] + bytes([raw[marker + 2] ^ 0x01]) + raw[marker + 3 :]
    assert tampered != raw
    (dest / "train.csv").write_bytes(tampered)

    ir = load_ir(case_dir / "experiment.ir.json")
    with pytest.raises(DataHashMismatch) as excinfo:
        execute_experiment(ir, tmp_path, seed=42, data_root=data_root)
    assert ir["dataset_snapshot_ref"]["sha256"] in str(excinfo.value)


def test_ir_content_hash_tamper_rejected(tmp_path: Path):
    case_dir = REPLAY_CASES[0]
    ir = json.loads((case_dir / "experiment.ir.json").read_text(encoding="utf-8"))
    ir["objective"] = "被篡改的目标"
    path = tmp_path / "mutated.ir.json"
    path.write_text(json.dumps(ir, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ContentHashMismatch):
        load_ir(path)


@pytest.mark.parametrize(
    "example_name",
    [
        "invalid-g1-time-col.json",
        "invalid-g3-direction.json",
        "invalid-g4-budget.json",
        "invalid-g5-baseline-protocol.json",
        "invalid-g6-authorization.json",
    ],
)
def test_blocked_ir_refused_execution(example_name: str, tmp_path: Path):
    ir = load_ir(EXAMPLES_DIR / example_name)  # loads fine: gate-level illegality
    with pytest.raises(ExecutionBlocked) as excinfo:
        execute_experiment(ir, tmp_path, seed=42, data_root=ROOT)
    assert "blocked by hard gates" in str(excinfo.value)


# ---------- ANF evidence envelope ----------

@pytest.mark.parametrize("case_dir", REPLAY_CASES, ids=CASE_IDS)
def test_evidence_anf_required_fields(case_dir: Path, tmp_path: Path):
    ir = load_ir(case_dir / "experiment.ir.json")
    run = _run_case(case_dir, tmp_path)
    evidence = run.evidence

    assert ANF_REQUIRED_FIELDS <= set(evidence)
    assert evidence["source_system"] == "cultivating"
    assert evidence["capability_id"] == "ml-experiment"
    assert evidence["skill_id"] == "experiment-run"
    assert evidence["evidence_type"] == "task_success"
    assert evidence["task_id"] == ir["experiment_id"]
    assert evidence["evidence_id"] == f"ev-{ir['content_hash'][:12]}"
    assert evidence["provenance_chain_id"] == f"pch-{ir['content_hash'][:12]}"
    assert evidence["measurement_protocol"] == canonical_bytes(
        ir["validation_protocol_ref"]
    ).decode("utf-8")
    assert evidence["executor"] == "framework/src/ir/runner.py"
    assert evidence["protocol_version"] == "0.1.0"
    assert evidence["metric_name"] == ir["metric_definition_ref"]["name"]
    assert evidence["metric_value"] == run.canonical["metrics"][evidence["metric_name"]]

    # Deterministic identifiers, never uuid4; timestamp is the only volatile.
    assert evidence["timestamp"]
    mutated = dict(evidence)
    mutated["timestamp"] = "1999-12-31T23:59:59+00:00"
    assert evidence_content_hash(mutated) == evidence_content_hash(evidence)


# ---------- compare_replays tolerance path ----------

def test_compare_replays_tolerance_path(tmp_path: Path):
    run_a = _run_case(REPLAY_CASES[0], tmp_path / "a")

    drifted_manifest = json.loads(json.dumps(run_a.manifest))
    name = next(iter(drifted_manifest["canonical"]["metrics"]))
    drifted_manifest["canonical"]["metrics"][name] += 1e-6
    drifted = RunResult(
        ir=run_a.ir,
        manifest=drifted_manifest,
        evidence=dict(run_a.evidence),
        oof_path=run_a.oof_path,
        oof_sha256=run_a.oof_sha256,
    )

    strict = compare_replays(run_a, drifted)  # default tolerance 1e-9
    assert not strict.canonical_hash_equal
    assert not strict.metrics_within_tolerance
    assert not strict.identical
    assert strict.differences

    loose = compare_replays(run_a, drifted, tolerance=1e-2)
    assert loose.metrics_within_tolerance  # tolerance governs the metrics flag
    assert not loose.canonical_hash_equal  # canonical bytes still differ
    assert not loose.identical             # overall verdict stays False

    self_cmp = compare_replays(run_a, run_a)
    assert self_cmp.identical and not self_cmp.differences


# ---------- fixture integrity & idempotency ----------

@pytest.mark.parametrize("case_dir", REPLAY_CASES, ids=CASE_IDS)
def test_fixture_generation_idempotent(case_dir: Path):
    fixture = case_dir / "fixtures" / "train.csv"
    before = fixture.read_bytes()

    proc = subprocess.run(
        [sys.executable, str(case_dir / "make_fixtures.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr

    after = fixture.read_bytes()
    assert after == before  # rerun reproduces the committed file byte for byte
    assert proc.stdout.strip().splitlines()[-1] == hashlib.sha256(after).hexdigest()


@pytest.mark.parametrize("case_dir", REPLAY_CASES, ids=CASE_IDS)
def test_ir_records_fixture_sha256_and_rows(case_dir: Path):
    ir = json.loads((case_dir / "experiment.ir.json").read_text(encoding="utf-8"))
    fixture = case_dir / "fixtures" / "train.csv"
    actual_sha = hashlib.sha256(fixture.read_bytes()).hexdigest()
    assert ir["dataset_snapshot_ref"]["sha256"] == actual_sha

    frame = pd.read_csv(fixture)
    assert ir["dataset_snapshot_ref"]["rows"] == len(frame)
    assert "target" in frame.columns
    assert "date" in frame.columns
    # Fixed serialization: "%.6f" floats, LF endings, UTF-8.
    raw = fixture.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")


# ---------- runtime dependency hygiene ----------

def test_runner_imports_no_mlflow():
    """runner must import cleanly with mlflow import-blocked: zero mlflow on
    any import path, even on machines where mlflow is installed."""
    code = (
        "import sys\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'mlflow' or name.startswith('mlflow.'):\n"
        "            raise ImportError('mlflow forbidden')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        "import framework.src.ir.runner as r\n"
        "bad = [n for n in sys.modules if n == 'mlflow' or n.startswith('mlflow.')]\n"
        "assert not bad, bad\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_experiment_ir_import_stays_sklearn_free():
    """The lazy runner exports in ir/__init__ must keep the pure-stdlib
    experiment_ir import path free of numpy/sklearn (SPEC-001 guard)."""
    code = (
        "import sys;"
        "import framework.src.ir.experiment_ir;"
        "import framework.src.ir as pkg;"
        "heavy = [n for n in ('numpy', 'pandas', 'sklearn') if n in sys.modules];"
        "print(','.join(heavy))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""


def test_lazy_runner_exports_resolvable():
    import framework.src.ir as ir_pkg

    for name in ("execute_experiment", "compare_replays", "RunResult", "ReplayComparison"):
        assert getattr(ir_pkg, name) is not None
    assert "execute_experiment" in dir(ir_pkg)
