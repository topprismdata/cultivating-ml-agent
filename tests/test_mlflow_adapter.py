"""ADAPTER-001 tests: MLflow tracking adapter (SPEC-005).

Real-tracking-store tests against a tmp_path sqlite tracking URI (mlflow is
installed locally; CI without mlflow skips the whole module via
``importorskip`` — see SPEC-005 §7). Covers the manifest→params flattening
(deterministic, dot-joined, capped), the params/metrics/tags/artifact
readback, the RuntimeError import guard, and cross-run reproducibility: two
executions of the same IR must land bitwise-equal metrics in the store.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

# Must be set before mlflow is imported anywhere (mlflow 3.17 prints an
# agent-hint notice on import; tests keep their output clean).
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import pytest

pytest.importorskip("mlflow")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlflow
from mlflow.tracking import MlflowClient

from framework.src.ir.experiment_ir import load_ir
from framework.src.ir.runner import RunResult, execute_experiment, evidence_content_hash
from framework.src.pipeline import mlflow_adapter
from framework.src.pipeline.mlflow_adapter import (
    MAX_PARAMS,
    flatten_manifest,
    log_experiment_run,
)

CASE_DIR = ROOT / "replays" / "s6e5-style"
TRACKING_DB = "sqlite:///mlflow-adapter-test.db"  # resolved against tmp cwd


@pytest.fixture(scope="module")
def replay_runs(tmp_path_factory):
    """Two executions of the same committed s6e5-style IR (G2 replay pair)."""
    tmp = tmp_path_factory.mktemp("mlflow-adapter")
    ir = load_ir(CASE_DIR / "experiment.ir.json")
    run_a = execute_experiment(ir, tmp / "a", seed=42, data_root=ROOT)
    run_b = execute_experiment(ir, tmp / "b", seed=42, data_root=ROOT)
    return ir, run_a, run_b


def _search_row(ir: dict, run_id: str):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(ir["experiment_id"])
    assert experiment is not None
    frame = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    rows = frame[frame["run_id"] == run_id]
    assert len(rows) == 1
    return rows.iloc[0]


# ---------- flattening ----------

def test_flatten_manifest_deterministic_shape():
    canonical = {
        "seed": 42,
        "metrics": {"rmsle": 0.1},
        "fold_sizes": [{"train": 1948, "val": 28}],
        "flag": True,
        "nothing": None,
        "note": "hello",
    }
    flat = flatten_manifest(canonical)
    assert flat == {
        "flag": "true",
        "fold_sizes.0.train": "1948",
        "fold_sizes.0.val": "28",
        "metrics.rmsle": "0.1",
        "nothing": "null",
        "note": "hello",
        "seed": "42",
    }
    assert flatten_manifest(canonical) == flat  # deterministic


def test_flatten_manifest_key_count_is_uncapped():
    big = {f"key_{i:04d}": i for i in range(MAX_PARAMS + 50)}
    assert len(flatten_manifest(big)) == MAX_PARAMS + 50


# ---------- cap enforcement ----------

def test_log_rejects_manifest_over_param_cap():
    canonical = {f"key_{i:04d}": i for i in range(MAX_PARAMS + 1)}
    run = RunResult(
        ir={},
        manifest={"canonical": canonical, "volatile": {}},
        evidence={},
        oof_path=Path("oof.csv"),
        oof_sha256="c" * 64,
    )
    with pytest.raises(ValueError, match="cap"):
        log_experiment_run(run, {}, {"experiment_id": "cap-check"})


# ---------- import guard ----------

def test_missing_mlflow_raises_runtime_error(monkeypatch, replay_runs):
    ir, run_a, _ = replay_runs
    monkeypatch.setattr(mlflow_adapter, "mlflow", None)
    with pytest.raises(RuntimeError, match="pip install mlflow"):
        log_experiment_run(run_a, run_a.evidence, ir)


# ---------- real round trip ----------

def test_log_and_readback(replay_runs, tmp_path, monkeypatch):
    ir, run_a, _ = replay_runs
    monkeypatch.chdir(tmp_path)  # sqlite db + artifact root land in tmp_path

    run_id = log_experiment_run(run_a, run_a.evidence, ir, tracking_uri=TRACKING_DB)
    assert isinstance(run_id, str) and run_id

    mlflow.set_tracking_uri(TRACKING_DB)
    row = _search_row(ir, run_id)

    # params: flattened canonical manifest, stringified
    canonical = run_a.canonical
    assert row["params.metrics.rmsle"] == str(canonical["metrics"]["rmsle"])
    assert row["params.model_family"] == "ridge"
    assert row["params.model_params.alpha"] == "1.0"
    assert row["params.seed"] == "42"
    assert row["params.fold_sizes.0.train"] == "1948"
    assert row["params.ir_content_hash"] == canonical["ir_content_hash"]
    assert row["params.data_sha256"] == canonical["data_sha256"]

    # metrics: {metric_name: value}, exact readback
    assert float(row["metrics.rmsle"]) == canonical["metrics"]["rmsle"]

    # tags: source_system + schema_version (decision tags await P3-A)
    assert row["tags.source_system"] == "cultivating"
    assert row["tags.schema_version"] == "0.1.0"

    # artifacts: evidence.json + OOF csv under evidence/
    client = MlflowClient()
    artifact_paths = {
        info.path for info in client.list_artifacts(run_id, "evidence")
    }
    assert artifact_paths == {
        "evidence/evidence.json",
        f"evidence/{ir['experiment_id']}.csv",
    }

    downloaded = Path(
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="evidence/evidence.json",
            dst_path=str(tmp_path / "dl-evidence"),
            tracking_uri=TRACKING_DB,
        )
    )
    parsed = json.loads(downloaded.read_text(encoding="utf-8"))
    assert evidence_content_hash(parsed) == evidence_content_hash(run_a.evidence)

    downloaded_oof = Path(
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=f"evidence/{ir['experiment_id']}.csv",
            dst_path=str(tmp_path / "dl-oof"),
            tracking_uri=TRACKING_DB,
        )
    )
    assert (
        hashlib.sha256(downloaded_oof.read_bytes()).hexdigest() == run_a.oof_sha256
    )


def test_double_run_metrics_bitwise_equal_in_store(replay_runs, tmp_path, monkeypatch):
    ir, run_a, run_b = replay_runs
    # G2 replay pair: same IR + data + code + seed ⇒ identical manifests.
    assert run_a.canonical == run_b.canonical

    monkeypatch.chdir(tmp_path)
    run_id_a = log_experiment_run(run_a, run_a.evidence, ir, tracking_uri=TRACKING_DB)
    run_id_b = log_experiment_run(run_b, run_b.evidence, ir, tracking_uri=TRACKING_DB)
    assert run_id_a != run_id_b

    mlflow.set_tracking_uri(TRACKING_DB)
    row_a = _search_row(ir, run_id_a)
    row_b = _search_row(ir, run_id_b)

    # Reproducibility through the tracking layer: bitwise-equal metrics and
    # identical params across the two runs.
    assert float(row_a["metrics.rmsle"]) == float(row_b["metrics.rmsle"])
    assert float(row_a["metrics.rmsle"]) == run_a.canonical["metrics"]["rmsle"]

    param_cols = [col for col in row_a.index if col.startswith("params.")]
    assert param_cols
    assert row_a[param_cols].to_dict() == row_b[param_cols].to_dict()
