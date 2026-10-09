"""ExperimentIR contract: schema, canonical hashing and hard gates.

Public API lives in ``ir.experiment_ir``; re-exported here so callers can
write ``from ir import load_ir, authorize_execution``.
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
]
