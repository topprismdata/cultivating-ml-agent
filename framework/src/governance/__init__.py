"""P5 governance package (SPEC-007): promotion gate, skill-tester gate,
append-only decision ledger.

Pure stdlib runtime (jsonschema only in tests). Iron rules: thresholds are
pre-registered and hash-pinned; the interface produces drafts only —
activation is a human/authority action (G4, ADR-001 "Not promotion").
"""
from .errors import EvidenceError, GovernanceError, PolicyError
from .ledger import (
    DEFAULT_LEDGER_DIR,
    GovernanceLedgerConflictError,
    GovernanceLedgerError,
    GovernanceLedgerTamperError,
    append_decision,
    decision_event,
    load_decision,
    verify_decision_integrity,
)
from .promotion import (
    PROMOTION_REASONS,
    PROMOTION_VOLATILE_FIELDS,
    PromotionDecision,
    activate_capability,
    compute_policy_content_hash,
    draft_experience_projection,
    evaluate_promotion,
    load_policy,
    verify_policy,
)
from .skill_gate import (
    GATE_VOLATILE_FIELDS,
    SKILL_GATE_REASONS,
    GateDecision,
    evaluate_skill_gate,
)

__all__ = [
    "DEFAULT_LEDGER_DIR",
    "EvidenceError",
    "GATE_VOLATILE_FIELDS",
    "GateDecision",
    "GovernanceError",
    "GovernanceLedgerConflictError",
    "GovernanceLedgerError",
    "GovernanceLedgerTamperError",
    "POLICY_SCHEMA_VERSION",
    "PROMOTION_REASONS",
    "PROMOTION_VOLATILE_FIELDS",
    "PolicyError",
    "PromotionDecision",
    "SKILL_GATE_REASONS",
    "activate_capability",
    "append_decision",
    "compute_policy_content_hash",
    "decision_event",
    "draft_experience_projection",
    "evaluate_promotion",
    "evaluate_skill_gate",
    "load_decision",
    "load_policy",
    "verify_decision_integrity",
    "verify_policy",
]
