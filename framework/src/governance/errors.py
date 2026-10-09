"""Exception hierarchy for the P5 governance package (SPEC-007).

Kept in a leaf module so promotion / skill_gate / ledger can import the base
classes without circular imports.
"""
from __future__ import annotations

__all__ = [
    "EvidenceError",
    "GovernanceError",
    "PolicyError",
]


class GovernanceError(Exception):
    """Base class for governance contract violations."""


class PolicyError(GovernanceError):
    """Policy document is malformed or its content hash does not verify."""


class EvidenceError(GovernanceError):
    """Evidence input violates the evidence-envelope-compatible contract."""
