"""MLflow tracking adapter (ADAPTER-001, SPEC-005).

Bridges one executed ExperimentIR run (``ir.runner.RunResult`` plus its ANF
evidence envelope) into an MLflow tracking store. Import-guarded: mlflow is
an optional dependency — importing this module never fails, calling
:func:`log_experiment_run` without mlflow installed raises ``RuntimeError``
pointing at the install remedy.

Mapping (SPEC-005 §6):

- experiment: ``ir["experiment_id"]`` (``mlflow.set_experiment`` by name)
- params: ``manifest["canonical"]`` flattened — nested mappings dot-join,
  list items index-join, leaves stringified deterministically; capped at
  :data:`MAX_PARAMS` keys (surplus ⇒ ``ValueError``, never silent loss)
- metrics: ``manifest["canonical"]["metrics"]`` verbatim
  (``{metric_name: value}``)
- artifacts: ``evidence.json`` + the OOF CSV under ``artifact_path="evidence"``
- tags: ``source_system=cultivating`` + ``schema_version``; decision tags
  land once the A-line DecisionTrace (P3-A) is wired — deliberately absent.

Reproducibility contract: two replays of the same IR produce the same
canonical manifest, hence bitwise-equal metrics and identical params in the
tracking store — the G2 replay discipline extends through tracking
(proved by ``tests/test_mlflow_adapter.py``).

Dependency policy: stdlib at module level; ``mlflow`` imported behind a
guard. No sklearn/numpy/pandas.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

try:  # optional dependency: the adapter stays importable without mlflow
    import mlflow
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    mlflow = None

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..ir.runner import RunResult

__all__ = ["ARTIFACT_PATH", "MAX_PARAMS", "SOURCE_SYSTEM", "flatten_manifest", "log_experiment_run"]

#: Upper bound on flattened params per run (adapter-level cap; the canonical
#: manifest of a SPEC-003 run stays far below it).
MAX_PARAMS = 500

#: Fixed ``source_system`` tag value (ANF evidence vocabulary).
SOURCE_SYSTEM = "cultivating"

#: Artifact directory holding evidence.json and the OOF CSV.
ARTIFACT_PATH = "evidence"


def _require_mlflow() -> None:
    if mlflow is None:
        raise RuntimeError(
            "mlflow is not installed; experiment tracking is optional — "
            "install it with `pip install mlflow` to use "
            "framework.src.pipeline.mlflow_adapter.log_experiment_run"
        )


def _stringify(value: Any) -> str:
    """Deterministic leaf stringification for MLflow params (params are
    strings server-side; booleans/None keep their JSON spelling)."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool) or value is None:
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def flatten_manifest(value: Any, _prefix: str = "") -> dict[str, str]:
    """Flatten a canonical manifest into MLflow params.

    Nested mappings dot-join their keys (``metrics.rmsle``), list items
    index-join (``fold_sizes.0.train``); mapping keys are visited in sorted
    order so the output is deterministic. Leaves stringify via
    :func:`_stringify`.
    """
    flat: dict[str, str] = {}
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            child = f"{_prefix}.{key}" if _prefix else str(key)
            flat.update(flatten_manifest(value[key], child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            flat.update(flatten_manifest(item, f"{_prefix}.{index}"))
    else:
        flat[_prefix] = _stringify(value)
    return flat


def log_experiment_run(
    run_result: "RunResult",
    evidence: Mapping[str, Any],
    ir: Mapping[str, Any],
    *,
    tracking_uri: str | None = None,
) -> str:
    """Log one executed run (manifest + evidence + OOF artifact) to MLflow.

    Args:
        run_result: executor output (``ir.runner.execute_experiment``).
        evidence: ANF evidence-envelope projection to store as
            ``evidence/evidence.json`` (usually ``run_result.evidence``).
        ir: the ExperimentIR dict that was executed (naming the MLflow
            experiment and the run).
        tracking_uri: optional MLflow tracking URI (e.g.
            ``sqlite:///<path>``); when omitted the caller's active
            tracking configuration is used.

    Returns:
        The MLflow run_id.

    Raises:
        RuntimeError: mlflow is not installed.
        ValueError: the flattened canonical manifest exceeds
            :data:`MAX_PARAMS` keys.
    """
    _require_mlflow()
    canonical = run_result.manifest["canonical"]

    params = flatten_manifest(canonical)
    if len(params) > MAX_PARAMS:
        raise ValueError(
            f"flattened canonical manifest has {len(params)} params, "
            f"exceeding the adapter cap of {MAX_PARAMS}"
        )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(ir["experiment_id"])

    metrics = {
        name: float(value) for name, value in canonical["metrics"].items()
    }
    tags = {
        "source_system": str(evidence.get("source_system") or SOURCE_SYSTEM),
        "schema_version": str(canonical["schema_version"]),
    }

    with mlflow.start_run(
        run_name=ir["experiment_id"], experiment_id=experiment.experiment_id
    ) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            (stage / "evidence.json").write_text(
                json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2)
                + "\n",
                encoding="utf-8",
            )
            oof_copy = stage / f"{ir['experiment_id']}.csv"
            shutil.copyfile(run_result.oof_path, oof_copy)
            mlflow.log_artifacts(str(stage), artifact_path=ARTIFACT_PATH)
        return run.info.run_id
