"""GOV-001 promotion interface (SPEC-007, ADR-002, G4).

Pure stdlib: no mlflow/sklearn, no uuid4; wall clock only via the injectable
``now`` parameter and never inside any hash (same discipline as
``ir.decision_trace``).

Iron rules this module implements:

1. **Thresholds are pre-registered** (计划基线 §10): every threshold evaluated
   here comes from a hash-pinned promotion policy; there is no code-level
   default that could drift from the audited document.
2. **Draft-only output (G4)**: a passing evaluation produces an ANF
   capability-record with ``lifecycle_status="draft"`` and a decision marked
   ``approved_for_draft`` — never ``active``. :func:`activate_capability`
   exists as the seam and unconditionally raises: activation requires a
   human/authority action (ANF policy-record authority constraint
   "Not promotion").
3. **Fixed reason vocabulary**: rejection reasons come from
   :data:`PROMOTION_REASONS`; each failing pre-registered threshold yields one
   reason entry, and all failing thresholds are reported (逐条).
4. **Mechanical P3 reuse**: :func:`draft_experience_projection` validates a
   P3 ``project_to_anf`` output and transcribes it into an evidence-source
   envelope via a fixed declared mapping — no second semantic processing.

Evidence inputs use the ANF evidence-envelope field names (``evidence_type``,
``timestamp``, ``metric_name``, ``task_family``, ...); only the fields this
module consumes are required, so ANF-shaped envelopes pass through untouched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from ..ir.decision_trace import canonical_bytes
from .errors import EvidenceError, GovernanceError, PolicyError

__all__ = [
    "ANF_SCOPE_LEVELS",
    "ANF_RISK_LEVELS",
    "CAPABILITY_ID_RE",
    "DECISION_STATUS_APPROVED_FOR_DRAFT",
    "DECISION_STATUS_REJECTED",
    "EVIDENCE_TYPE_CONTRADICTION",
    "EVIDENCE_TYPE_DISPUTE",
    "EVIDENCE_TYPE_NEGATIVE_TRANSFER",
    "EVIDENCE_TYPE_TASK_SUCCESS",
    "EVIDENCE_TYPES",
    "POLICY_SCHEMA_VERSION",
    "PROMOTION_REASONS",
    "PROMOTION_VOLATILE_FIELDS",
    "EvidenceError",
    "GovernanceError",
    "PolicyError",
    "PromotionDecision",
    "activate_capability",
    "compute_decision_id",
    "compute_policy_content_hash",
    "draft_experience_projection",
    "evaluate_promotion",
    "load_policy",
    "verify_policy",
]

POLICY_SCHEMA_VERSION = "0.1.0"

CAPABILITY_ID_RE = re.compile(r"^cap-[A-Za-z0-9._-]+$")
EVIDENCE_ID_RE = re.compile(r"^ev-[A-Za-z0-9._-]+$")
POLICY_ID_RE = re.compile(r"^pol-[A-Za-z0-9._-]+$")
CONTENT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

ANF_RISK_LEVELS = ("low", "medium", "high")
ANF_SCOPE_LEVELS = ("personal", "project", "team", "organization")

#: ANF evidence-envelope ``evidence_type`` enum (verified against
#: agent-nurture-framework schemas/evidence-envelope.schema.json).
EVIDENCE_TYPE_TASK_SUCCESS = "task_success"
EVIDENCE_TYPE_TASK_FAILURE = "task_failure"
EVIDENCE_TYPE_NEGATIVE_TRANSFER = "negative_transfer"
EVIDENCE_TYPE_CORRECTION = "correction"
EVIDENCE_TYPE_CONTRADICTION = "contradiction"
EVIDENCE_TYPE_DISPUTE = "dispute"
EVIDENCE_TYPES = (
    EVIDENCE_TYPE_TASK_SUCCESS,
    EVIDENCE_TYPE_TASK_FAILURE,
    EVIDENCE_TYPE_NEGATIVE_TRANSFER,
    EVIDENCE_TYPE_CORRECTION,
    EVIDENCE_TYPE_CONTRADICTION,
    EVIDENCE_TYPE_DISPUTE,
)

DECISION_STATUS_APPROVED_FOR_DRAFT = "approved_for_draft"
DECISION_STATUS_REJECTED = "rejected"

#: Closed vocabulary of promotion rejection reasons (SPEC-007 §3). One entry
#: per failing pre-registered threshold; never extended ad hoc.
PROMOTION_REASONS = (
    "insufficient_task_success",
    "insufficient_task_families",
    "missing_negative_case",
    "insufficient_negative_transfer",
    "stale_evidence",
    "unresolved_dispute",
    "unresolved_contradiction",
)

#: Volatile promotion-decision fields excluded from the decision_id hash
#: (evaluated_at is audit metadata exactly like DecisionTrace created_at).
PROMOTION_VOLATILE_FIELDS = ("evaluated_at",)

_POLICY_REQUIRED_FIELDS = (
    "schema_version",
    "policy_id",
    "version",
    "supersedes",
    "locked_at",
    "policy_content_hash",
    "activation_authority",
    "thresholds",
)
_THRESHOLD_FIELDS = (
    "min_task_success",
    "min_task_families",
    "require_negative_case",
    "min_negative_transfer",
    "max_evidence_age_days",
    "require_no_dispute",
)
_THRESHOLD_MINIMUMS = {
    "min_task_success": 1,
    "min_task_families": 1,
    "min_negative_transfer": 0,
    "max_evidence_age_days": 1,
}
_AUTHORITY_LEVELS = ("A0", "A1", "A2", "A3")

#: P3 experience-record projection required fields (the ANF nine).
_PROJECTION_REQUIRED_FIELDS = (
    "record_id",
    "task",
    "decision",
    "decision_summary",
    "action",
    "outcome",
    "classification",
    "trust_level",
    "timestamp",
)
_PROJECTION_RESULTS = ("success", "failure", "partial")

#: Fixed declared mapping outcome.result -> evidence_type (mechanical
#: transcription of the P3 projection; no second semantic processing).
_EVIDENCE_TYPE_FROM_RESULT = {
    "success": EVIDENCE_TYPE_TASK_SUCCESS,
    "failure": EVIDENCE_TYPE_TASK_FAILURE,
    "partial": EVIDENCE_TYPE_CORRECTION,
}


# ---------------------------------------------------------------------------
# Canonical hashing (same family as ir.decision_trace / adapters.estimates)
# ---------------------------------------------------------------------------

def compute_policy_content_hash(policy: Mapping) -> str:
    """sha256 over canonical JSON of the policy minus the volatile pair.

    ``policy_content_hash`` and ``locked_at`` are excluded: the hash pins the
    pre-registered thresholds, not the moment they were locked.
    """
    body = {
        k: v
        for k, v in policy.items()
        if k not in ("policy_content_hash", "locked_at")
    }
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def verify_policy(policy: Mapping) -> dict:
    """Structurally validate a policy and recompute its content hash.

    Returns a deep copy on success; raises :class:`PolicyError` on any
    structural violation or hash mismatch (tampered policy is refused at
    load time — 阈值一经提交即锁定).
    """
    if not isinstance(policy, Mapping):
        raise PolicyError("policy 必须是映射")
    missing = [f for f in _POLICY_REQUIRED_FIELDS if f not in policy]
    if missing:
        raise PolicyError(f"policy 缺少必填字段: {missing}")
    if policy["schema_version"] != POLICY_SCHEMA_VERSION:
        raise PolicyError(
            f"schema_version 必须为 {POLICY_SCHEMA_VERSION!r}，得到 {policy['schema_version']!r}"
        )
    if not POLICY_ID_RE.match(policy["policy_id"] or ""):
        raise PolicyError(f"policy_id 必须匹配 ^pol-[A-Za-z0-9._-]+$，得到 {policy['policy_id']!r}")
    if not policy["version"] or not isinstance(policy["version"], str):
        raise PolicyError("version 必须为非空字符串")
    supersedes = policy["supersedes"]
    if supersedes is not None and not POLICY_ID_RE.match(supersedes):
        raise PolicyError(f"supersedes 必须为 policy_id 或 null，得到 {supersedes!r}")
    if policy["activation_authority"] not in _AUTHORITY_LEVELS:
        raise PolicyError(
            f"activation_authority 必须为 {_AUTHORITY_LEVELS} 之一，得到 {policy['activation_authority']!r}"
        )
    content_hash = policy["policy_content_hash"]
    if not isinstance(content_hash, str) or not CONTENT_HASH_RE.match(content_hash):
        raise PolicyError("policy_content_hash 必须为 64 位小写十六进制")
    thresholds = policy["thresholds"]
    if not isinstance(thresholds, Mapping):
        raise PolicyError("thresholds 必须是映射")
    unknown = sorted(set(thresholds) - set(_THRESHOLD_FIELDS))
    if unknown:
        raise PolicyError(f"thresholds 含未知字段: {unknown}")
    missing_t = [f for f in _THRESHOLD_FIELDS if f not in thresholds]
    if missing_t:
        raise PolicyError(f"thresholds 缺少字段: {missing_t}")
    for name in ("require_negative_case", "require_no_dispute"):
        if not isinstance(thresholds[name], bool):
            raise PolicyError(f"thresholds.{name} 必须为布尔值")
    for name, floor in _THRESHOLD_MINIMUMS.items():
        value = thresholds[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < floor:
            raise PolicyError(f"thresholds.{name} 必须为 >= {floor} 的整数，得到 {value!r}")
    recomputed = compute_policy_content_hash(policy)
    if recomputed != content_hash:
        raise PolicyError(
            "policy_content_hash 不匹配：策略被篡改或阈值口径事后漂移 "
            f"(期望 {content_hash}，重算 {recomputed})"
        )
    return copy.deepcopy(dict(policy))


def load_policy(path) -> dict:
    """Load and verify a promotion policy from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        try:
            policy = json.load(fh)
        except json.JSONDecodeError as exc:
            raise PolicyError(f"{path}: 不是合法 JSON: {exc}") from exc
    return verify_policy(policy)


# ---------------------------------------------------------------------------
# Evidence input contract (ANF evidence-envelope compatible)
# ---------------------------------------------------------------------------

def _parse_timestamp(value: Any, *, what: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{what}: 期望 ISO-8601 时间字符串，得到 {value!r}")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise EvidenceError(f"{what}: 无法解析的时间戳 {value!r}: {exc}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _validate_evidence(entry: Any, index: int) -> dict:
    """Validate one evidence dict against the fields this module consumes.

    ANF evidence-envelope-shaped dicts pass untouched; only consumed fields
    are enforced (envelope: evidence_id/evidence_type/timestamp 必填，其余
    按需)。unknown extra fields ride along verbatim.
    """
    if not isinstance(entry, Mapping):
        raise EvidenceError(f"evidence[{index}]: 必须是映射")
    evidence_id = entry.get("evidence_id")
    if not isinstance(evidence_id, str) or not EVIDENCE_ID_RE.match(evidence_id):
        raise EvidenceError(
            f"evidence[{index}].evidence_id: 必须匹配 ^ev-[A-Za-z0-9._-]+$，得到 {evidence_id!r}"
        )
    evidence_type = entry.get("evidence_type")
    if evidence_type not in EVIDENCE_TYPES:
        raise EvidenceError(
            f"evidence[{index}].evidence_type: 必须为封闭词表 {list(EVIDENCE_TYPES)}，得到 {evidence_type!r}"
        )
    task_family = entry.get("task_family")
    if task_family is not None and not isinstance(task_family, str):
        raise EvidenceError(f"evidence[{index}].task_family: 期望字符串，得到 {task_family!r}")
    metric_name = entry.get("metric_name")
    if metric_name is not None and not isinstance(metric_name, str):
        raise EvidenceError(f"evidence[{index}].metric_name: 期望字符串，得到 {metric_name!r}")
    _parse_timestamp(entry.get("timestamp"), what=f"evidence[{index}].timestamp")
    return dict(entry)


def _reason(reason: str, detail: str) -> dict:
    return {"reason": reason, "detail": detail}


# ---------------------------------------------------------------------------
# Decision identity
# ---------------------------------------------------------------------------

def _decision_hash_body(decision: Mapping) -> dict:
    """Content covered by decision_id: everything except decision_id and the
    volatile fields (evaluated_at)."""
    return {
        k: v
        for k, v in decision.items()
        if k not in ("decision_id",) + PROMOTION_VOLATILE_FIELDS
    }


def compute_decision_id(decision: Mapping) -> str:
    """``gvd-`` + first 12 hex of sha256 over the hash body (no wall clock)."""
    return "gvd-" + hashlib.sha256(canonical_bytes(_decision_hash_body(decision))).hexdigest()[:12]


@dataclass(frozen=True)
class PromotionDecision:
    """Outcome of :func:`evaluate_promotion` (SPEC-007 §3)."""

    capability_id: str
    status: str
    reasons: list
    thresholds: dict
    policy_id: str
    policy_content_hash: str
    activation_authority: str
    evaluated_at: str
    draft_capability_record: Optional[dict] = None
    kind: str = "promotion"
    decision_id: str = field(default="")

    def __post_init__(self):
        if not self.decision_id:
            object.__setattr__(self, "decision_id", compute_decision_id(self.to_dict()))

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "decision_id": self.decision_id,
            "capability_id": self.capability_id,
            "status": self.status,
            "reasons": [dict(r) for r in self.reasons],
            "thresholds": dict(self.thresholds),
            "policy_id": self.policy_id,
            "policy_content_hash": self.policy_content_hash,
            "activation_authority": self.activation_authority,
            "evaluated_at": self.evaluated_at,
            "draft_capability_record": (
                dict(self.draft_capability_record)
                if self.draft_capability_record is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "PromotionDecision":
        return cls(
            capability_id=data["capability_id"],
            status=data["status"],
            reasons=[dict(r) for r in data["reasons"]],
            thresholds=dict(data["thresholds"]),
            policy_id=data["policy_id"],
            policy_content_hash=data["policy_content_hash"],
            activation_authority=data["activation_authority"],
            evaluated_at=data["evaluated_at"],
            draft_capability_record=(
                dict(data["draft_capability_record"])
                if data.get("draft_capability_record") is not None
                else None
            ),
            kind=data.get("kind", "promotion"),
            decision_id=data.get("decision_id", ""),
        )

    @property
    def approved(self) -> bool:
        return self.status == DECISION_STATUS_APPROVED_FOR_DRAFT


# ---------------------------------------------------------------------------
# GOV-001: evaluate_promotion
# ---------------------------------------------------------------------------

def _check_thresholds(evidence: list, thresholds: Mapping) -> list:
    """Evaluate every pre-registered threshold; return one reason per failure."""
    reasons = []
    successes = [e for e in evidence if e["evidence_type"] == EVIDENCE_TYPE_TASK_SUCCESS]
    families = sorted({e["task_family"] for e in successes if e.get("task_family")})
    negatives = [e for e in evidence if e["evidence_type"] == EVIDENCE_TYPE_NEGATIVE_TRANSFER]
    require_negative = bool(thresholds["require_negative_case"])
    require_no_dispute = bool(thresholds["require_no_dispute"])

    if len(successes) < thresholds["min_task_success"]:
        reasons.append(
            _reason(
                "insufficient_task_success",
                f"task_success={len(successes)} < min_task_success={thresholds['min_task_success']}",
            )
        )
    if len(families) < thresholds["min_task_families"]:
        reasons.append(
            _reason(
                "insufficient_task_families",
                f"去重 task_family={len(families)} {families} < "
                f"min_task_families={thresholds['min_task_families']}",
            )
        )
    if require_negative and not negatives:
        # G4: 负例缺失一票否决，无负例的 capability 永远到不了 draft。
        reasons.append(
            _reason(
                "missing_negative_case",
                "require_negative_case=true 且 negative_transfer=0",
            )
        )
    elif len(negatives) < thresholds["min_negative_transfer"]:
        reasons.append(
            _reason(
                "insufficient_negative_transfer",
                f"negative_transfer={len(negatives)} < "
                f"min_negative_transfer={thresholds['min_negative_transfer']}",
            )
        )
    if require_no_dispute:
        unresolved_disputes = sorted(
            e["evidence_id"]
            for e in evidence
            if e["evidence_type"] == EVIDENCE_TYPE_DISPUTE and e.get("resolved") is not True
        )
        if unresolved_disputes:
            reasons.append(
                _reason("unresolved_dispute", f"未解决 dispute 证据: {unresolved_disputes}")
            )
        unresolved_contradictions = sorted(
            e["evidence_id"]
            for e in evidence
            if e["evidence_type"] == EVIDENCE_TYPE_CONTRADICTION
            and e.get("resolved") is not True
        )
        if unresolved_contradictions:
            reasons.append(
                _reason(
                    "unresolved_contradiction",
                    f"未解决 contradiction 证据: {unresolved_contradictions}",
                )
            )
    return reasons


def _check_freshness(evidence: list, thresholds: Mapping, now: datetime) -> list:
    """Stale evidence check: age measured against the injected reference time.

    Details never carry computed ages (they would smuggle the clock into the
    decision_id hash); only the offending evidence ids and the threshold.
    """
    limit = thresholds["max_evidence_age_days"]
    stale = sorted(
        e["evidence_id"]
        for e in evidence
        if (now - _parse_timestamp(e["timestamp"], what="timestamp")).days > limit
    )
    if stale:
        return [_reason("stale_evidence", f"证据超过 max_evidence_age_days={limit} 天: {stale}")]
    return []


def _draft_capability_record(
    capability_id: str,
    evidence: list,
    *,
    risk_level: str,
    scope: str,
    policy: Mapping,
) -> dict:
    """Build the ANF capability-record projection. lifecycle_status 恒为 draft."""
    activation = policy["activation_authority"]
    evidence_ids = sorted({e["evidence_id"] for e in evidence})
    provenance_chain_id = "prov-" + hashlib.sha256(
        canonical_bytes(sorted(evidence_ids))
    ).hexdigest()[:16]
    newest = max(
        evidence,
        key=lambda e: _parse_timestamp(e["timestamp"], what="timestamp"),
    )
    return {
        "capability_id": capability_id,
        "lifecycle_status": "draft",  # G4: 接口只产 draft，激活永远是人/权限的动作
        "risk_level": risk_level,
        "scope": scope,
        "evidence_ids": evidence_ids,
        "provenance_chain_id": provenance_chain_id,
        "freshness": {
            "last_verified": newest["timestamp"],
            "last_verified_evidence_id": newest["evidence_id"],
        },
        "authority_constraints": [
            policy["policy_id"],
            f"{policy['policy_id']}.activation_floor_{activation.lower()}",
        ],
        "activation_authority": activation,
    }


def evaluate_promotion(
    capability_id: str,
    evidence_list,
    policy: Mapping,
    *,
    risk_level: str,
    scope: str,
    now: Optional[Any] = None,
) -> PromotionDecision:
    """Evaluate GOV-001 against the pre-registered thresholds of ``policy``.

    ``evidence_list`` entries are ANF evidence-envelope-compatible dicts
    (consumed fields: ``evidence_id``, ``evidence_type``, ``timestamp``,
    ``task_family``, ``metric_name``; ``resolved`` optionally marks dispute/
    contradiction evidence as resolved). ``risk_level``/``scope`` are the
    caller-declared ANF risk/scope levels. ``now`` is the freshness reference
    time (datetime or ISO string); default wall clock, never hashed.

    Any failing threshold yields one fixed-vocabulary reason; all failures
    are reported. A fully passing evaluation returns
    ``status="approved_for_draft"`` with the draft capability record —
    approval authorizes draft production only, never activation.
    """
    policy = verify_policy(policy)
    if not isinstance(capability_id, str) or not CAPABILITY_ID_RE.match(capability_id):
        raise GovernanceError(
            f"capability_id 必须匹配 ^cap-[A-Za-z0-9._-]+$，得到 {capability_id!r}"
        )
    if risk_level not in ANF_RISK_LEVELS:
        raise GovernanceError(f"risk_level 必须为 {ANF_RISK_LEVELS} 之一，得到 {risk_level!r}")
    if scope not in ANF_SCOPE_LEVELS:
        raise GovernanceError(f"scope 必须为 {ANF_SCOPE_LEVELS} 之一，得到 {scope!r}")
    if not isinstance(evidence_list, (list, tuple)) or not evidence_list:
        raise EvidenceError("evidence_list 必须是非空列表")
    evidence = [_validate_evidence(e, i) for i, e in enumerate(evidence_list)]
    if now is None:
        reference = datetime.now(timezone.utc)
    elif isinstance(now, datetime):
        reference = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        reference = reference.astimezone(timezone.utc)
    else:
        reference = _parse_timestamp(now, what="now")

    reasons = _check_thresholds(evidence, policy["thresholds"])
    reasons.extend(_check_freshness(evidence, policy["thresholds"], reference))
    evaluated_at = reference.isoformat()

    record = None
    if reasons:
        status = DECISION_STATUS_REJECTED
    else:
        status = DECISION_STATUS_APPROVED_FOR_DRAFT
        record = _draft_capability_record(
            capability_id,
            evidence,
            risk_level=risk_level,
            scope=scope,
            policy=policy,
        )
    decision = PromotionDecision(
        capability_id=capability_id,
        status=status,
        reasons=reasons,
        thresholds=dict(policy["thresholds"]),
        policy_id=policy["policy_id"],
        policy_content_hash=policy["policy_content_hash"],
        activation_authority=policy["activation_authority"],
        evaluated_at=evaluated_at,
        draft_capability_record=record,
    )
    if compute_decision_id(decision.to_dict()) != decision.decision_id:  # pragma: no cover
        raise GovernanceError("decision_id 自检失败")
    return decision


def activate_capability(capability: Mapping, *, actor: str = "", authorization_ref: str = ""):
    """G4 seam: activation is NEVER performed by this interface.

    Draft -> active requires a human/authority action at A2 or above (ANF
    authority plane; policy-record authority constraint "Not promotion").
    This function exists so callers have an explicit, greppable refusal
    point instead of silently doing nothing.
    """
    raise GovernanceError("激活需 A2+ 人工授权，接口不自动晋升")


# ---------------------------------------------------------------------------
# P3 projection reuse (mechanical; no second semantic processing)
# ---------------------------------------------------------------------------

def draft_experience_projection(decision_trace_projection: Mapping) -> dict:
    """Validate a P3 ``ir.decision_trace.project_to_anf`` output and
    transcribe it into an evidence-source dict.

    Validation: all nine ANF experience-record required fields present and
    non-empty, ``outcome.result`` in the closed vocabulary. Transcription is
    the fixed declared mapping in ``_EVIDENCE_TYPE_FROM_RESULT`` plus verbatim
    copies — the returned dict can be appended to ``evidence_list`` directly
    (augment ``task_family`` at the call site if family counting needs it).
    No field is reinterpreted, no value recomputed beyond the sha256 ids.
    """
    if not isinstance(decision_trace_projection, Mapping):
        raise EvidenceError("decision_trace_projection 必须是映射")
    for name in _PROJECTION_REQUIRED_FIELDS:
        value = decision_trace_projection.get(name)
        if value is None or value == "" or value == [] or value == {}:
            raise EvidenceError(f"decision_trace_projection.{name}: 缺失或为空（P3 投影契约）")
    outcome = decision_trace_projection["outcome"]
    if not isinstance(outcome, Mapping) or outcome.get("result") not in _PROJECTION_RESULTS:
        raise EvidenceError(
            "decision_trace_projection.outcome.result: 必须为 "
            f"{list(_PROJECTION_RESULTS)} 之一，得到 {outcome if not isinstance(outcome, Mapping) else outcome.get('result')!r}"
        )
    projection = dict(decision_trace_projection)
    record_id = projection["record_id"]
    result = outcome["result"]
    execution_refs = projection.get("execution_trace_refs") or []
    identity = {"projection": projection, "purpose": "governance-evidence-source"}
    return {
        "evidence_id": "ev-" + hashlib.sha256(canonical_bytes(identity)).hexdigest(),
        "evidence_type": _EVIDENCE_TYPE_FROM_RESULT[result],
        "source_system": "cultivating",
        "metric_name": outcome.get("metric_name"),
        "metric_value": outcome.get("metric_value"),
        "timestamp": projection["timestamp"],
        "provenance_chain_id": "prov-"
        + hashlib.sha256(canonical_bytes({"projection_ref": record_id})).hexdigest()[:16],
        "projection_ref": record_id,
        "artifact_refs": list(execution_refs),
    }
