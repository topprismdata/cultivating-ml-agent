"""Skill Tester gate (SPEC-007 §4, G4).

Consumes a skill-tester run report (dict with a ``tests`` array whose entries
carry ``case_type`` ∈ {positive, negative} and ``status`` ∈ {pass, fail}) and
produces a :class:`GateDecision`.

Iron rules (fixed, not configurable per run):

1. **负例 >= 1**（G4）：a skill with zero negative test cases never passes —
   the gate reason is ``missing_negative_case``. The requirement rides the
   pre-registered ``require_negative_case`` threshold of the supplied policy.
2. **无失败用例**：any failed case fails the gate (``failed_case_present``).
3. **skill_id 非空**：an empty skill id fails the gate
   (``empty_skill_id``).

Crystallization output is a *suggestion* only: the gate never writes to
``skills/``; crystallization stays a human-executed flow.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..ir.decision_trace import canonical_bytes
from .errors import GovernanceError
from .promotion import verify_policy

__all__ = [
    "CASE_STATUSES",
    "CASE_TYPES",
    "GATE_STATUS_FAIL",
    "GATE_STATUS_PASS",
    "GATE_VOLATILE_FIELDS",
    "SKILL_GATE_REASONS",
    "GateDecision",
    "compute_decision_id",
    "evaluate_skill_gate",
]

GATE_STATUS_PASS = "pass"
GATE_STATUS_FAIL = "fail"

CASE_TYPES = ("positive", "negative")
CASE_STATUSES = ("pass", "fail")

#: Closed vocabulary of skill-gate failure reasons.
SKILL_GATE_REASONS = (
    "empty_skill_id",
    "missing_negative_case",
    "failed_case_present",
)

#: Gate decisions carry no wall-clock content: they are fully deterministic.
GATE_VOLATILE_FIELDS: tuple = ()


def _gate_hash_body(decision: Mapping) -> dict:
    return {
        k: v
        for k, v in decision.items()
        if k not in ("decision_id",) + GATE_VOLATILE_FIELDS
    }


def compute_decision_id(decision: Mapping) -> str:
    """``gvg-`` + first 12 hex of sha256 over the hash body."""
    return "gvg-" + hashlib.sha256(canonical_bytes(_gate_hash_body(decision))).hexdigest()[:12]


@dataclass(frozen=True)
class GateDecision:
    """Outcome of :func:`evaluate_skill_gate` (SPEC-007 §4)."""

    gate: str
    skill_id: str
    status: str
    reasons: list
    counts: dict
    report_digest: str
    policy_id: str
    policy_content_hash: str
    crystallization: dict
    kind: str = "skill_gate"
    decision_id: str = field(default="")

    def __post_init__(self):
        if not self.decision_id:
            object.__setattr__(self, "decision_id", compute_decision_id(self.to_dict()))

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "decision_id": self.decision_id,
            "gate": self.gate,
            "skill_id": self.skill_id,
            "status": self.status,
            "reasons": [dict(r) for r in self.reasons],
            "counts": dict(self.counts),
            "report_digest": self.report_digest,
            "policy_id": self.policy_id,
            "policy_content_hash": self.policy_content_hash,
            "crystallization": dict(self.crystallization),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "GateDecision":
        return cls(
            gate=data["gate"],
            skill_id=data["skill_id"],
            status=data["status"],
            reasons=[dict(r) for r in data["reasons"]],
            counts=dict(data["counts"]),
            report_digest=data["report_digest"],
            policy_id=data["policy_id"],
            policy_content_hash=data["policy_content_hash"],
            crystallization=dict(data["crystallization"]),
            kind=data.get("kind", "skill_gate"),
            decision_id=data.get("decision_id", ""),
        )

    @property
    def passed(self) -> bool:
        return self.status == GATE_STATUS_PASS


def _validate_report(tester_report: Any) -> list:
    """Validate the skill-tester run report shape; return the tests list."""
    if not isinstance(tester_report, Mapping):
        raise GovernanceError("tester_report 必须是映射（skill-tester run 输出）")
    tests = tester_report.get("tests")
    if not isinstance(tests, list) or not tests:
        raise GovernanceError("tester_report.tests: 必须是非空数组")
    for idx, case in enumerate(tests):
        if not isinstance(case, Mapping):
            raise GovernanceError(f"tester_report.tests[{idx}]: 必须是映射")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise GovernanceError(f"tester_report.tests[{idx}].case_id: 期望非空字符串")
        case_type = case.get("case_type")
        if case_type not in CASE_TYPES:
            raise GovernanceError(
                f"tester_report.tests[{idx}].case_type: 必须为 {list(CASE_TYPES)}，得到 {case_type!r}"
            )
        status = case.get("status")
        if status not in CASE_STATUSES:
            raise GovernanceError(
                f"tester_report.tests[{idx}].status: 必须为 {list(CASE_STATUSES)}，得到 {status!r}"
            )
    return [dict(case) for case in tests]


def evaluate_skill_gate(skill_id: Any, tester_report: Mapping, policy: Mapping) -> GateDecision:
    """Evaluate the skill-tester gate; see module docstring for the rules.

    Malformed tester output (shape contract violation) raises
    :class:`GovernanceError`; rule violations produce a ``fail`` decision with
    fixed-vocabulary reasons. The crystallization field is a suggestion only.
    """
    policy = verify_policy(policy)
    tests = _validate_report(tester_report)
    reasons = []
    if not isinstance(skill_id, str) or not skill_id.strip():
        reasons.append(
            {
                "reason": "empty_skill_id",
                "detail": "skill_id 必须为非空字符串",
            }
        )
    negatives = [c for c in tests if c["case_type"] == "negative"]
    failed = sorted(c["case_id"] for c in tests if c["status"] == "fail")
    if policy["thresholds"]["require_negative_case"] and len(negatives) < 1:
        # G4: 负例缺失 -> fail，结晶建议不得给出。
        reasons.append(
            {
                "reason": "missing_negative_case",
                "detail": f"负例数 {len(negatives)} < 1（require_negative_case=true）",
            }
        )
    if failed:
        reasons.append(
            {
                "reason": "failed_case_present",
                "detail": f"失败用例: {failed}",
            }
        )
    status = GATE_STATUS_PASS if not reasons else GATE_STATUS_FAIL
    counts = {
        "tests": len(tests),
        "positive": sum(1 for c in tests if c["case_type"] == "positive"),
        "negative": len(negatives),
        "failed": len(failed),
    }
    crystallization = {
        "suggested": status == GATE_STATUS_PASS and bool(skill_id and skill_id.strip()),
        "skill_id": skill_id if isinstance(skill_id, str) else None,
        "note": "建议性输出（仅 suggest）：通过门禁后建议人工执行结晶流程写入 skills/；本接口不自动写文件。",
    }
    decision = GateDecision(
        gate="skill_tester",
        skill_id=skill_id if isinstance(skill_id, str) else "",
        status=status,
        reasons=reasons,
        counts=counts,
        report_digest=hashlib.sha256(canonical_bytes(tester_report)).hexdigest(),
        policy_id=policy["policy_id"],
        policy_content_hash=policy["policy_content_hash"],
        crystallization=crystallization,
    )
    if compute_decision_id(decision.to_dict()) != decision.decision_id:  # pragma: no cover
        raise GovernanceError("decision_id 自检失败")
    return decision
