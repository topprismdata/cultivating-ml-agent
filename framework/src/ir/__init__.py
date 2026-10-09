"""ExperimentIR contract: schema, canonical hashing, hard gates, executor.

Public API lives in ``ir.experiment_ir`` (pure stdlib) and ``ir.runner``
(executor: stdlib + numpy/pandas/sklearn). Both are re-exported here so
callers can write ``from ir import load_ir, execute_experiment``.

Runner exports are lazy (PEP 562): importing ``ir.experiment_ir`` must stay
free of numpy/sklearn — the runtime module purity guard in
tests/test_experiment_ir.py depends on it.
"""
from .experiment_ir import (
    SCHEMA_VERSION,
    AuthorizationVerdict,
    ContentHashMismatch,
    GateResult,
    IRError,
    IRSchemaError,
    authorize_execution,
    canonical_bytes,
    compute_content_hash,
    load_ir,
    run_gates,
    schema_validate,
)

#: Names resolved lazily from ``ir.runner`` on first attribute access.
_RUNNER_EXPORTS = frozenset(
    {
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
    }
)

__all__ = [
    "SCHEMA_VERSION",
    "AuthorizationVerdict",
    "ContentHashMismatch",
    "GateResult",
    "IRError",
    "IRSchemaError",
    "authorize_execution",
    "canonical_bytes",
    "compute_content_hash",
    "load_ir",
    "run_gates",
    "schema_validate",
    *_RUNNER_EXPORTS,
]


def __getattr__(name: str):
    if name in _RUNNER_EXPORTS:
        from . import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _RUNNER_EXPORTS)
