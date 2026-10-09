"""ExperimentIR executor and evidence layer (SPEC-003).

Turns an authorized ExperimentIR document into a replayable training run
plus an ANF evidence-envelope projection. Replay contract (plan G2 gate):

    same ExperimentIR + same data bytes + same code version + same seed
        => byte-identical canonical manifest hash, bitwise-equal metrics.

Everything derived from the wall clock, the host or the interpreter lives
ONLY in volatile zones (``manifest["volatile"]``, ``evidence["timestamp"]``)
and never enters any content hash. Identifiers derived from content
(``evidence_id`` = ``ev-<ir_content_hash[:12]>``) replace uuid4, which is
banned from canonical material (past-lesson: uuid4/timestamps break
content-hash reproducibility).

Dependency policy: stdlib + numpy/pandas/sklearn only, zero mlflow, zero
network. Guarded by ``tests/test_replay_reproducibility.py``.

Replay conventions (P2 vertical slice, documented in SPEC-003):

- The label column is named ``target`` (matches ``pipeline.oof`` default).
- Feature columns are every CSV column except the reserved set
  ``{"id", "target", validation_protocol_ref.time_col}``, taken in CSV
  column order.
- Rows whose label is missing are dropped before splitting; the OOF frame
  covers the remaining rows.
- Non-numeric feature columns are integer-coded against their sorted
  unique values (``np.unique`` order — never set iteration order).
- Fold seeds derive deterministically from ``(seed, fold_idx)``.

Usage:
    from framework.src.ir.runner import execute_experiment, compare_replays

    ir = load_ir("replays/s6e5-style/experiment.ir.json")
    run = execute_experiment(ir, project_root=Path("."), data_root=Path("."))
    verdict = compare_replays(run_a, run_b)
    assert verdict.identical
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .experiment_ir import (
    ContentHashMismatch,
    authorize_execution,
    canonical_bytes,
    compute_content_hash,
)
from framework.src.pipeline.oof import build_oof_frame
from framework.src.pipeline.splits import make_folds

__all__ = [
    "EVIDENCE_VOLATILE_FIELDS",
    "EXECUTOR",
    "LABEL_COL",
    "MANIFEST_SCHEMA_VERSION",
    "METRIC_VERSION",
    "MODEL_FAMILIES",
    "PROTOCOL_VERSION",
    "DataHashMismatch",
    "ExecutionBlocked",
    "ReplayComparison",
    "RunResult",
    "RunnerError",
    "canonical_json",
    "canonical_sha256",
    "compare_replays",
    "evidence_content_hash",
    "execute_experiment",
    "manifest_canonical_hash",
]

# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------

_METRICS_PATH = Path(__file__).resolve().parents[1] / "utils" / "metrics.py"


def _load_metrics_module():
    """Load the existing ``framework/src/utils/metrics.py`` implementation.

    Loaded by file location because the ``framework.src.utils`` package
    ``__init__`` re-exports submission helpers whose legacy top-level
    imports (``from pipeline.validate import ...``) only resolve when
    ``framework/src`` itself sits on sys.path — a mode the runtime does not
    use (see SPEC-003 §依赖纪律). This reuses the metric implementations
    (rmsle/rmse/mae live there) without forking them.
    """
    cached = sys.modules.get("framework_utils_metrics")
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location("framework_utils_metrics", _METRICS_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RunnerError(f"cannot load metrics module from {_METRICS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["framework_utils_metrics"] = module
    spec.loader.exec_module(module)
    return module


get_metric = _load_metrics_module().get_metric

MANIFEST_SCHEMA_VERSION = "0.1.0"
PROTOCOL_VERSION = "0.1.0"
#: Metric implementation pin: framework/src/utils/metrics.py (rmsle/rmse/mae
#: implemented there; this constant is the ``metric_version`` reported in
#: evidence so a future metric change must bump it and regenerate goldens).
METRIC_VERSION = "framework-utils-metrics/1.0.0"
EXECUTOR = "framework/src/ir/runner.py"
LABEL_COL = "target"
#: Evidence fields that are volatile and excluded from evidence_content_hash.
EVIDENCE_VOLATILE_FIELDS = ("timestamp",)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INT32_SEED_MODULUS = 2**31 - 1


class RunnerError(Exception):
    """Base class for executor failures."""


class ExecutionBlocked(RunnerError):
    """A hard gate refused execution (authorize_execution not ok)."""


class DataHashMismatch(RunnerError):
    """Data bytes disagree with dataset_snapshot_ref.sha256."""


class UnknownModelFamily(RunnerError):
    """candidate.family is not registered in MODEL_FAMILIES."""


class UnknownMetric(RunnerError):
    """metric_definition_ref.name is not in utils.metrics registry."""


# ---------------------------------------------------------------------------
# Canonical hashing (same canonicalization contract as ExperimentIR)
# ---------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    """Deterministic JSON text: sorted keys, compact separators, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(obj: Any) -> str:
    """sha256 hexdigest over canonical_json encoded as UTF-8."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def manifest_canonical_hash(manifest: Mapping[str, Any]) -> str:
    """Content hash of a run manifest's canonical zone only."""
    return canonical_sha256(manifest["canonical"])


def evidence_content_hash(evidence: Mapping[str, Any]) -> str:
    """Content hash of an evidence envelope, excluding volatile fields."""
    body = {k: v for k, v in evidence.items() if k not in EVIDENCE_VOLATILE_FIELDS}
    return canonical_sha256(body)


# ---------------------------------------------------------------------------
# Model families
# ---------------------------------------------------------------------------

def _make_ridge(params: Mapping[str, Any], seed: int) -> Ridge:
    """sklearn Ridge; params passed through (e.g. alpha). Closed-form and
    deterministic, so no seed is consumed."""
    return Ridge(**dict(params))


def _make_hgb(params: Mapping[str, Any], seed: int) -> HistGradientBoostingRegressor:
    """HistGradientBoostingRegressor; params passed through, with
    random_state defaulting to the derived fold seed when absent."""
    resolved = dict(params)
    resolved.setdefault("random_state", seed)
    return HistGradientBoostingRegressor(**resolved)


#: Registry mapping candidate.family to a (params, seed) -> estimator factory.
#: candidate[0] whose family is registered here is the default execution
#: candidate of execute_experiment.
MODEL_FAMILIES: Mapping[str, Callable[[Mapping[str, Any], int], Any]] = MappingProxyType(
    {
        "ridge": _make_ridge,
        "hgb": _make_hgb,
    }
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunResult:
    """Outcome of one experiment execution.

    manifest splits into a canonical zone (every replay-relevant fact) and a
    volatile zone (wall clock / host / interpreter). evidence is the ANF
    evidence-envelope projection; its content hash excludes volatile fields.
    """

    ir: dict
    manifest: dict
    evidence: dict
    oof_path: Path
    oof_sha256: str

    @property
    def canonical(self) -> dict:
        return self.manifest["canonical"]

    @property
    def canonical_sha256(self) -> str:
        return manifest_canonical_hash(self.manifest)


@dataclass(frozen=True)
class ReplayComparison:
    """Per-item booleans plus the overall verdict for two runs.

    ``identical`` is the conjunction of all flags; the tolerance governs only
    the metrics flag, so any canonical difference (including metric bytes)
    still fails the overall verdict — replay equality is bitwise by
    construction and the tolerance exists to diagnose near-miss drift.
    """

    canonical_hash_equal: bool
    metrics_within_tolerance: bool
    data_sha256_equal: bool
    ir_content_hash_equal: bool
    seed_equal: bool
    code_version_equal: bool
    differences: tuple[str, ...] = ()

    @property
    def identical(self) -> bool:
        return not self.differences


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------

def _fold_seed(seed: int, fold_idx: int) -> int:
    """Per-fold seed derived deterministically from (seed, fold_idx)."""
    return (seed + fold_idx) % _INT32_SEED_MODULUS


def _utc_now_iso() -> str:
    """Wall-clock timestamp — volatile material only, never hashed."""
    return datetime.now(timezone.utc).isoformat()


def _code_version() -> str:
    """``git rev-parse HEAD`` of the framework repo; "unknown" when git or
    the repository is unavailable. Deliberately resolved from this module's
    repo root, not from project_root (which is only an output location)."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _resolve_data_path(ir: Mapping[str, Any], project_root: Path,
                       data_root: Optional[Path]) -> Path:
    """dataset_snapshot_ref.uri resolution: ``data_root`` overrides the base
    when given, otherwise paths resolve relative to ``project_root``."""
    uri = ir["dataset_snapshot_ref"]["uri"]
    if uri.startswith("file://"):
        uri = uri[len("file://"):]
    path = Path(uri)
    if path.is_absolute():
        return path
    base = Path(data_root) if data_root is not None else Path(project_root)
    return base / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _time_values(frame: pd.DataFrame, time_col: str) -> np.ndarray:
    """Materialize the splits-contract time array (float weeks).

    Numeric columns are taken as-is (assumed weeks per pipeline.splits);
    date columns must be ISO ``YYYY-MM-DD`` and map to ordinal/7.
    """
    series = frame[time_col]
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(dtype=np.float64)
    parsed = pd.to_datetime(series, format="%Y-%m-%d")
    return parsed.map(lambda ts: ts.toordinal() / 7.0).to_numpy(dtype=np.float64)


def _feature_matrix(frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Design matrix in fixed IR/CSV column order. Non-numeric columns are
    integer-coded against sorted unique values (never set iteration order)."""
    columns = []
    for name in feature_cols:
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series):
            columns.append(series.to_numpy(dtype=np.float64))
            continue
        values = series.astype(str)
        uniques = list(np.unique(values.to_numpy()))  # sorted, deterministic
        coding = {value: float(index) for index, value in enumerate(uniques)}
        columns.append(values.map(coding).to_numpy(dtype=np.float64))
    return np.column_stack(columns)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def execute_experiment(
    ir: dict,
    project_root: Path,
    *,
    seed: int = 42,
    data_root: Optional[Path] = None,
) -> RunResult:
    """Execute an authorized ExperimentIR and produce RunResult.

    Pipeline: authorization gates -> IR integrity re-check -> data hash
    verification -> fold split -> per-fold training -> OOF metric ->
    OOF artifact -> manifest + evidence.

    Raises:
        ExecutionBlocked: any hard gate failed.
        ContentHashMismatch: the dict was mutated after compilation.
        DataHashMismatch: data bytes disagree with the IR snapshot digest.
        UnknownModelFamily / UnknownMetric: unregistered family or metric.
    """
    project_root = Path(project_root)

    verdict = authorize_execution(ir)
    if not verdict.ok:
        raise ExecutionBlocked(
            "ExperimentIR blocked by hard gates: " + "; ".join(verdict.reasons)
        )
    if compute_content_hash(ir) != ir.get("content_hash"):
        raise ContentHashMismatch(
            "ExperimentIR dict mutated after compilation "
            f"(content_hash={ir.get('content_hash')!r} does not match recomputed hash)"
        )

    candidates = ir["candidates"]
    if not candidates:
        raise RunnerError("ExperimentIR carries no candidates to execute")
    candidate = candidates[0]  # candidate[0] is the default execution candidate
    family = candidate["family"]
    if family not in MODEL_FAMILIES:
        raise UnknownModelFamily(
            f"model family {family!r} is not registered; "
            f"available: {sorted(MODEL_FAMILIES)}"
        )

    data_path = _resolve_data_path(ir, project_root, data_root)
    data_sha256 = _sha256_file(data_path)
    expected_sha256 = ir["dataset_snapshot_ref"]["sha256"]
    if data_sha256 != expected_sha256:
        raise DataHashMismatch(
            f"data sha256 mismatch: IR records {expected_sha256}, "
            f"file {data_path} hashes to {data_sha256}"
        )

    frame = pd.read_csv(data_path)
    if LABEL_COL not in frame.columns:
        raise RunnerError(
            f"dataset {data_path} lacks the label column {LABEL_COL!r} "
            "(P2 replay convention)"
        )
    kept = frame[frame[LABEL_COL].notna()].reset_index(drop=True)
    y = kept[LABEL_COL].to_numpy(dtype=np.float64)

    proto = ir["validation_protocol_ref"]
    strategy = proto["strategy"]
    validation = SimpleNamespace(
        strategy=strategy,
        n_folds=int(proto.get("n_folds", 5)),
        val_size_weeks=proto.get("val_size_weeks", 20),
        group_col=proto.get("group_col", ""),
    )
    time_values = _time_values(kept, proto["time_col"]) if strategy == "time_based" else None
    group_values = (
        kept[proto["group_col"]].astype(str).to_numpy() if strategy == "group" else None
    )
    folds = make_folds(
        y,
        validation,
        time=time_values,
        groups=group_values,
        random_state=seed,
    )

    reserved = {LABEL_COL, "id"}
    if proto.get("time_col"):
        reserved.add(proto["time_col"])
    feature_cols = [col for col in kept.columns if col not in reserved]

    factory = MODEL_FAMILIES[family]
    X = _feature_matrix(kept, feature_cols)
    oof_pred = np.full(kept.shape[0], np.nan, dtype=np.float64)
    fold_sizes: list[dict[str, int]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        model = factory(candidate["params"], _fold_seed(seed, fold_idx))
        model.fit(X[train_idx], y[train_idx])
        oof_pred[val_idx] = model.predict(X[val_idx])
        fold_sizes.append({"train": int(train_idx.size), "val": int(val_idx.size)})

    metric_name = ir["metric_definition_ref"]["name"]
    try:
        metric_fn = get_metric(metric_name)
    except KeyError as exc:
        raise UnknownMetric(
            f"metric {metric_name!r} is not registered in framework/src/utils/metrics.py"
        ) from exc
    scored = np.isfinite(oof_pred)
    if not scored.any():
        raise RunnerError("no validation rows were scored; split is degenerate")
    metric_value = float(metric_fn(y[scored], oof_pred[scored]))

    ids = kept["id"].to_numpy() if "id" in kept.columns else np.arange(kept.shape[0])
    oof_frame = build_oof_frame(oof_pred, ids=ids, targets=y)
    payload = oof_frame.to_csv(
        index=False, float_format="%.6f", lineterminator="\n"
    ).encode("utf-8")
    oof_dir = project_root / "outputs" / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    oof_path = oof_dir / f"{ir['experiment_id']}.csv"
    oof_path.write_bytes(payload)
    oof_sha256 = hashlib.sha256(payload).hexdigest()
    oof_relpath = (oof_path.relative_to(project_root)).as_posix()

    ir_content_hash = ir["content_hash"]
    manifest = {
        "canonical": {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "ir_content_hash": ir_content_hash,
            "code_version": _code_version(),
            "data_sha256": data_sha256,
            "seed": int(seed),
            "strategy": strategy,
            "n_folds": len(folds),
            "fold_sizes": fold_sizes,
            "metrics": {metric_name: metric_value},
            "oof_sha256": oof_sha256,
            "model_family": family,
            "model_params": candidate["params"],
        },
        "volatile": {
            "created_at": _utc_now_iso(),
            "host": platform.node(),
            "python_version": platform.python_version(),
        },
    }

    evidence = {
        "evidence_id": f"ev-{ir_content_hash[:12]}",
        "capability_id": "ml-experiment",
        "skill_id": "experiment-run",
        "evidence_type": "task_success",
        "source_system": "cultivating",
        "measurement_protocol": canonical_bytes(proto).decode("utf-8"),
        "protocol_version": PROTOCOL_VERSION,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metric_version": METRIC_VERSION,
        "executor": EXECUTOR,
        "provenance_chain_id": f"pch-{ir_content_hash[:12]}",
        "task_id": ir["experiment_id"],
        "artifact_refs": [{"path": oof_relpath, "sha256": oof_sha256}],
        "timestamp": _utc_now_iso(),  # volatile: excluded from content hash
    }

    return RunResult(
        ir=ir,
        manifest=manifest,
        evidence=evidence,
        oof_path=oof_path,
        oof_sha256=oof_sha256,
    )


# ---------------------------------------------------------------------------
# Replay comparison
# ---------------------------------------------------------------------------

def compare_replays(
    a: RunResult,
    b: RunResult,
    *,
    tolerance: float = 1e-9,
) -> ReplayComparison:
    """Compare two runs item by item; ``identical`` is the overall verdict."""
    ca, cb = a.manifest["canonical"], b.manifest["canonical"]

    canonical_hash_equal = manifest_canonical_hash(a.manifest) == manifest_canonical_hash(b.manifest)

    metrics_a, metrics_b = ca["metrics"], cb["metrics"]
    if metrics_a.keys() == metrics_b.keys():
        max_delta = max(
            (abs(float(metrics_a[name]) - float(metrics_b[name])) for name in metrics_a),
            default=0.0,
        )
        metrics_ok = max_delta <= tolerance
    else:
        metrics_ok = False

    differences: list[str] = []
    if not canonical_hash_equal:
        differences.append("canonical hash mismatch")
    if not metrics_ok:
        differences.append(f"metric delta exceeds tolerance {tolerance!r}")
    if ca["data_sha256"] != cb["data_sha256"]:
        differences.append("data sha256 mismatch")
    if ca["ir_content_hash"] != cb["ir_content_hash"]:
        differences.append("ir content hash mismatch")
    if ca["seed"] != cb["seed"]:
        differences.append("seed mismatch")
    if ca["code_version"] != cb["code_version"]:
        differences.append("code version mismatch")

    return ReplayComparison(
        canonical_hash_equal=canonical_hash_equal,
        metrics_within_tolerance=metrics_ok,
        data_sha256_equal=ca["data_sha256"] == cb["data_sha256"],
        ir_content_hash_equal=ca["ir_content_hash"] == cb["ir_content_hash"],
        seed_equal=ca["seed"] == cb["seed"],
        code_version_equal=ca["code_version"] == cb["code_version"],
        differences=tuple(differences),
    )
