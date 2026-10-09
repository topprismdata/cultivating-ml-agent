"""IR contracts: ExperimentIR (SPEC-001) and DecisionTrace (SPEC-004).

Public API lives in ``ir.experiment_ir`` (pure stdlib), ``ir.decision_trace``
(pure stdlib) and ``ir.runner`` (executor: stdlib + numpy/pandas/sklearn).
All are re-exported here so callers can write ``from ir import load_ir,
new_decision, execute_experiment``.

Runner exports are lazy (PEP 562): importing ``ir.experiment_ir`` or
``ir.decision_trace`` must stay free of numpy/sklearn — the runtime module
purity guard in tests/test_experiment_ir.py depends on it.
"""
from .decision_trace import (
    ANF_RECORD_ID_PATTERN,
    ChainHashMismatch,
    GENESIS_PREV_EVENT_ID,
    IllegalTransition,
    LedgerWriteError,
    MissingEventFieldError,
    SeparationOfDutiesError,
    TraceError,
    TraceSchemaError,
    UnknownEventError,
    append_event,
    canonical_bytes as canonical_decision_bytes,
    compute_decision_id,
    compute_event_id,
    correct,
    load_decision,
    new_decision,
    project_to_anf,
    record_from_decision_outcome,
    save_decision,
    verify_decision,
)
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
    "ANF_RECORD_ID_PATTERN",
    "SCHEMA_VERSION",
    "AuthorizationVerdict",
    "ChainHashMismatch",
    "ContentHashMismatch",
    "GENESIS_PREV_EVENT_ID",
    "GateResult",
    "IRError",
    "IllegalTransition",
    "IRSchemaError",
    "LedgerWriteError",
    "MissingEventFieldError",
    "SeparationOfDutiesError",
    "TraceError",
    "TraceSchemaError",
    "UnknownEventError",
    "append_event",
    "authorize_execution",
    "canonical_bytes",
    "canonical_decision_bytes",
    "compute_content_hash",
    "compute_decision_id",
    "compute_event_id",
    "correct",
    "load_decision",
    "load_ir",
    "new_decision",
    "project_to_anf",
    "record_from_decision_outcome",
    "run_gates",
    "save_decision",
    "schema_validate",
    "verify_decision",
    *_RUNNER_EXPORTS,
]


def __getattr__(name: str):
    if name in _RUNNER_EXPORTS:
        from . import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _RUNNER_EXPORTS)
