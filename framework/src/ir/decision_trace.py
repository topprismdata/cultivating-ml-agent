"""DecisionTrace ledger (SPEC-004, ADR-002): append-only decision state machine.

Pure stdlib: no mlflow/sklearn, no uuid4, no wall-clock reads except the
``clock=None`` default. Identity (``decision_id`` / ``event_id``) is always
sha256-derived from canonical JSON; ``created_at`` never enters any hash so
injected clocks and replay cannot change identity.

Immutability model: a decision is an ordered event chain. The first event is
``proposed`` with ``prev_event_id="genesis"``; every later event links to its
predecessor and its ``event_id`` covers the full event content (excluding
``event_id`` itself and ``created_at``). Corrections append a new event that
points at the corrected event via ``correction_of``; history is never edited.
``load_decision`` re-verifies every link, so any tampered stored line is
rejected.

The document (``new_decision`` result) is a fold over the event chain: the
proposed event's ``payload`` carries the full-fidelity document snapshot and
later ``executed``/``evaluated`` events promote ``execution_ref`` /
``outcome_refs`` to document level. Saving writes one JSON line per event
(append-only); loading rebuilds the document from the chain.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

__all__ = [
    "SCHEMA_VERSION",
    "SCHEMA_RELPATH",
    "STATUSES",
    "EVENT_TYPES",
    "GENESIS_PREV_EVENT_ID",
    "REJECTED_REASONS",
    "METRIC_DIRECTIONS",
    "ANF_RECORD_ID_PATTERN",
    "ChainHashMismatch",
    "IllegalTransition",
    "LedgerWriteError",
    "MissingEventFieldError",
    "SeparationOfDutiesError",
    "TraceError",
    "TraceSchemaError",
    "UnknownEventError",
    "append_event",
    "canonical_bytes",
    "compute_decision_id",
    "compute_event_id",
    "correct",
    "load_decision",
    "new_decision",
    "project_to_anf",
    "record_from_decision_outcome",
    "save_decision",
    "verify_decision",
]

SCHEMA_VERSION = "0.1.0"
SCHEMA_RELPATH = (
    Path("schemas") / "decision-trace" / SCHEMA_VERSION / "decision-trace.schema.json"
)

DECISION_ID_RE = re.compile(r"^dt-[0-9a-f]{12}$")
EVENT_ID_RE = re.compile(r"^ev-[0-9a-f]{64}$")
CHAIN_HEAD_RE = re.compile(r"^(genesis|ev-[0-9a-f]{64})$")

STATUS_PROPOSED = "proposed"
STATUS_APPROVED = "approved"
STATUS_EXECUTED = "executed"
STATUS_EVALUATED = "evaluated"
STATUS_REJECTED = "rejected"

STATUSES = (
    STATUS_PROPOSED,
    STATUS_APPROVED,
    STATUS_EXECUTED,
    STATUS_EVALUATED,
    STATUS_REJECTED,
)

EVENT_TYPES = (
    "proposed",
    "approved",
    "executed",
    "evaluated",
    "rejected",
    "correction",
)

GENESIS_PREV_EVENT_ID = "genesis"

#: Legal transitions; key = current decision status. Terminal states map to
#: the empty set. Corrections bypass the table (they never change status) and
#: must go through :func:`correct`.
_TRANSITIONS: dict[str, frozenset] = {
    STATUS_PROPOSED: frozenset({STATUS_APPROVED, STATUS_REJECTED}),
    STATUS_APPROVED: frozenset({STATUS_EXECUTED, STATUS_REJECTED}),
    STATUS_EXECUTED: frozenset({STATUS_EVALUATED}),
    STATUS_EVALUATED: frozenset(),
    STATUS_REJECTED: frozenset(),
}

#: Closed vocabulary of rejection reasons (shared DecisionOutcome contract).
REJECTED_REASONS = frozenset(
    {
        "worse_than_baseline",
        "below_min_improvement",
        "lost_to_selected",
        "constraint_gate_failed",
    }
)

METRIC_DIRECTIONS = ("maximize", "minimize")

#: Optional event extension fields; anything else passed via ``**fields`` is a
#: contract violation.
_EVENT_FIELDS = frozenset(
    {
        "authorization_ref",
        "execution_ref",
        "evidence_refs",
        "outcome_refs",
        "reason",
        "correction_of",
        "payload",
    }
)

#: Fields excluded from the event_id hash. ``created_at`` must never enter any
#: hash: identity survives clock injection and replay (SPEC-004 §6).
_HASH_EXCLUDED_FIELDS = ("event_id", "created_at")

#: Shared DecisionOutcome contract fields (B-line producer -> A-line consumer).
_OUTCOME_FIELDS = (
    "intent_ref",
    "state_snapshot_ref",
    "alternatives",
    "constraint_check",
    "baseline_ref",
    "selected_option",
    "rejected_options_and_reasons",
    "uncertainty",
    "decision_actor",
    "authorization_ref",
    "evidence_refs",
    "metric_name",
    "metric_direction",
)

#: Contract fields with no top-level slot in the document schema (which is
#: ``additionalProperties: false``); they ride in the proposed event payload.
_PAYLOAD_ONLY_FIELDS = ("metric_name", "metric_direction")

#: ANF experience-record ``record_id`` pattern (ADR-002 OQ3, verified against
#: agent-nurture-framework@33f444b).
ANF_RECORD_ID_PATTERN = r"^exp-[A-Za-z0-9._-]+$"

#: ANF trust levels used by the projection.
ANF_TRUST_VERIFIED = "verified_execution"
ANF_TRUST_INFERENCE = "agent_inference"

_ANF_SUMMARY_MAX = 2000  # experience-record decision_summary maxLength


class TraceError(Exception):
    """Base class for DecisionTrace contract violations."""


class TraceSchemaError(TraceError):
    """Document does not satisfy the DecisionTrace structure."""


class IllegalTransition(TraceError):
    """Status transition not allowed by the SPEC-004 state machine."""


class SeparationOfDutiesError(TraceError):
    """Four-eyes violation: approver equals proposer."""


class MissingEventFieldError(TraceError):
    """Event type requires a mandatory field that was not supplied."""


class UnknownEventError(TraceError):
    """Referenced event_id does not exist in the decision chain."""


class ChainHashMismatch(TraceError):
    """Recomputed hash chain does not match stored event ids (tamper)."""


class LedgerWriteError(TraceError):
    """Append-only ledger conflict: stored lines changed or diverged."""


# ---------------------------------------------------------------------------
# Canonical hashing
# ---------------------------------------------------------------------------

def canonical_bytes(obj: Any) -> bytes:
    """Deterministic UTF-8 JSON bytes: sorted keys, no insignificant space."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def compute_decision_id(
    intent_ref: str,
    selected_option: dict,
    alternatives: list,
    baseline_ref: dict,
) -> str:
    """``dt-`` + first 12 hex of sha256 over the identity seed.

    The seed covers exactly the selection-defining content: intent, selected
    option, alternatives and baseline. Timestamps, actors and evidence never
    enter, so the same decision content always yields the same id.
    """
    seed = {
        "alternatives": alternatives,
        "baseline_ref": baseline_ref,
        "intent_ref": intent_ref,
        "selected_option": selected_option,
    }
    return "dt-" + hashlib.sha256(canonical_bytes(seed)).hexdigest()[:12]


def event_content(event: dict) -> dict:
    """Event content covered by the chain hash (excludes event_id/created_at)."""
    return {k: v for k, v in event.items() if k not in _HASH_EXCLUDED_FIELDS}


def compute_event_id(event: dict) -> str:
    """``ev-`` + sha256 over canonical content including ``prev_event_id``."""
    return "ev-" + hashlib.sha256(canonical_bytes(event_content(event))).hexdigest()


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------

def _timestamp(clock: Optional[Callable[[], Any]]) -> str:
    """ISO-8601 UTC timestamp from the injected clock (default: wall clock)."""
    if clock is None:
        return datetime.now(timezone.utc).isoformat()
    value = clock()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, str):
        return value
    raise TypeError("clock must return datetime or ISO-8601 string")


# ---------------------------------------------------------------------------
# Contract-field validation
# ---------------------------------------------------------------------------

def _require_outcome_fields(outcome: dict) -> None:
    missing = [f for f in _OUTCOME_FIELDS if f not in outcome]
    if missing:
        # KeyError names every missing shared-contract field.
        raise KeyError(", ".join(missing))


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_outcome(outcome: dict) -> None:
    _require_outcome_fields(outcome)
    if not _nonempty_str(outcome["intent_ref"]):
        raise TraceSchemaError("intent_ref: 期望非空字符串")
    if not isinstance(outcome["selected_option"], dict):
        raise TraceSchemaError("selected_option: 期望对象")
    if not _nonempty_str(outcome["selected_option"].get("candidate_id", "")):
        raise TraceSchemaError("selected_option.candidate_id: 期望非空字符串")
    if not isinstance(outcome["alternatives"], list):
        raise TraceSchemaError("alternatives: 期望数组")
    for alt in outcome["alternatives"]:
        if not isinstance(alt, dict) or not _nonempty_str(alt.get("candidate_id", "")):
            raise TraceSchemaError("alternatives[*].candidate_id: 期望非空字符串")
    if not isinstance(outcome["baseline_ref"], dict):
        raise TraceSchemaError("baseline_ref: 期望对象")
    if outcome["metric_direction"] not in METRIC_DIRECTIONS:
        raise TraceSchemaError(
            f"metric_direction: 必须为 {METRIC_DIRECTIONS} 之一，得到 {outcome['metric_direction']!r}"
        )
    for item in outcome["rejected_options_and_reasons"]:
        if (
            not isinstance(item, dict)
            or item.get("reason") not in REJECTED_REASONS
        ):
            raise TraceSchemaError(
                f"rejected_options_and_reasons[*].reason: 必须为封闭词表 {sorted(REJECTED_REASONS)}"
            )


# ---------------------------------------------------------------------------
# Event construction and chain verification
# ---------------------------------------------------------------------------

def _build_event(
    prev_event_id: str,
    event_type: str,
    status: str,
    actor: str,
    created_at: str,
    fields: dict,
) -> dict:
    if not _nonempty_str(actor):
        raise TraceSchemaError("actor: 期望非空字符串")
    unknown = set(fields) - _EVENT_FIELDS
    if unknown:
        raise TypeError(f"未知事件字段: {sorted(unknown)}；允许: {sorted(_EVENT_FIELDS)}")
    event = {
        "prev_event_id": prev_event_id,
        "event_type": event_type,
        "status": status,
        "actor": actor,
        "created_at": created_at,
    }
    for key in (
        "authorization_ref",
        "execution_ref",
        "evidence_refs",
        "outcome_refs",
        "reason",
        "correction_of",
        "payload",
    ):
        if fields.get(key) is not None:
            event[key] = fields[key]
    event["event_id"] = compute_event_id(event)
    return event


def verify_decision(decision: dict) -> None:
    """Verify structure and recompute the full hash chain.

    Raises :class:`ChainHashMismatch` when any stored ``event_id`` or
    ``prev_event_id`` link fails recomputation (tamper detection);
    :class:`TraceSchemaError` for structural violations.
    """
    if not isinstance(decision, dict):
        raise TraceSchemaError("decision: 期望对象")
    did = decision.get("decision_id")
    if not isinstance(did, str) or not DECISION_ID_RE.match(did):
        raise TraceSchemaError(f"decision_id: 不匹配 {DECISION_ID_RE.pattern}: {did!r}")
    events = decision.get("events")
    if not isinstance(events, list) or not events:
        raise TraceSchemaError("events: 期望非空数组")
    first = events[0]
    if not isinstance(first, dict):
        raise TraceSchemaError("events[0]: 期望对象")
    if first.get("event_type") != "proposed":
        raise TraceSchemaError("events[0].event_type: 首事件必须为 proposed")
    if first.get("prev_event_id") != GENESIS_PREV_EVENT_ID:
        raise TraceSchemaError(
            f"events[0].prev_event_id: 首事件必须为 {GENESIS_PREV_EVENT_ID!r}"
        )
    prev = GENESIS_PREV_EVENT_ID
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            raise TraceSchemaError(f"events[{idx}]: 期望对象")
        if event.get("prev_event_id") != prev:
            raise ChainHashMismatch(
                f"events[{idx}].prev_event_id 前向链接断裂: 期望 {prev!r}，"
                f"得到 {event.get('prev_event_id')!r}"
            )
        expected = compute_event_id(event)
        if event.get("event_id") != expected:
            raise ChainHashMismatch(
                f"events[{idx}].event_id 链哈希不匹配: 期望 {expected}，"
                f"得到 {event.get('event_id')}（事件内容被篡改）"
            )
        if event.get("status") not in STATUSES:
            raise TraceSchemaError(
                f"events[{idx}].status: 非法状态 {event.get('status')!r}"
            )
        prev = event["event_id"]
    last = events[-1]
    if decision.get("status") != last["status"]:
        raise TraceSchemaError(
            f"status 与末事件不一致: doc={decision.get('status')!r} last={last['status']!r}"
        )
    for field in (
        "schema_version",
        "intent_ref",
        "state_snapshot_ref",
        "alternatives",
        "constraint_check",
        "baseline_ref",
        "selected_option",
        "rejected_options_and_reasons",
        "evidence_refs",
        "decision_actor",
        "authorization_ref",
        "uncertainty",
        "created_at",
    ):
        if field not in decision:
            raise TraceSchemaError(f"缺少必填字段: {field}")


def _promote(doc: dict, event: dict) -> None:
    """Promote event fields to document level (document = fold(events)).

    ``executed`` promotes ``execution_ref``; ``evaluated`` promotes
    ``outcome_refs``. Used both when appending live and when replaying a
    stored chain, so in-memory and loaded documents never diverge.
    """
    if event["event_type"] == "executed" and "execution_ref" in event:
        doc["execution_ref"] = event["execution_ref"]
    if event["event_type"] == "evaluated" and "outcome_refs" in event:
        doc["outcome_refs"] = event["outcome_refs"]


def _document_from_events(events: list) -> dict:
    """Fold the event chain back into the full-fidelity document.

    The proposed event payload is the document snapshot at proposal time;
    later events promote ``execution_ref`` (executed) and ``outcome_refs``
    (evaluated) to document level. Document ``status`` always equals the
    last event's status.
    """
    payload = events[0].get("payload")
    if not isinstance(payload, dict):
        raise TraceSchemaError("首事件缺少 payload 快照，无法重建文档")
    doc = {k: v for k, v in payload.items() if k not in _PAYLOAD_ONLY_FIELDS}
    doc["created_at"] = events[0]["created_at"]
    for event in events[1:]:
        _promote(doc, event)
    doc["status"] = events[-1]["status"]
    doc["events"] = list(events)
    return doc


def _payload_snapshot(outcome: dict, decision_id: str) -> dict:
    """Document snapshot stored in the proposed event payload.

    Excludes ``status``/``events`` (chain-derivable) and ``created_at``
    (must stay out of the hash).
    """
    snapshot = {"schema_version": SCHEMA_VERSION, "decision_id": decision_id}
    for field in _OUTCOME_FIELDS:
        if field in _PAYLOAD_ONLY_FIELDS:
            continue
        snapshot[field] = outcome[field]
    for field in _PAYLOAD_ONLY_FIELDS:
        snapshot[field] = outcome[field]
    return snapshot


# ---------------------------------------------------------------------------
# Public API: lifecycle
# ---------------------------------------------------------------------------

def new_decision(outcome: dict, *, clock: Optional[Callable[[], Any]] = None) -> dict:
    """Open a decision ledger from a shared-contract DecisionOutcome.

    Builds the genesis ``proposed`` event from ``outcome``; ``decision_id``
    and ``event_id`` are sha256-derived and clock-independent. Missing
    shared-contract fields raise ``KeyError`` naming the field.
    """
    _validate_outcome(outcome)
    decision_id = compute_decision_id(
        outcome["intent_ref"],
        outcome["selected_option"],
        outcome["alternatives"],
        outcome["baseline_ref"],
    )
    created_at = _timestamp(clock)
    event = _build_event(
        prev_event_id=GENESIS_PREV_EVENT_ID,
        event_type="proposed",
        status=STATUS_PROPOSED,
        actor=outcome["decision_actor"],
        created_at=created_at,
        fields={
            "authorization_ref": outcome["authorization_ref"],
            "evidence_refs": list(outcome["evidence_refs"]),
            "payload": _payload_snapshot(outcome, decision_id),
        },
    )
    doc = _document_from_events([event])
    verify_decision(doc)
    return doc


def append_event(
    decision: dict,
    event_type: str,
    *,
    actor: str,
    clock: Optional[Callable[[], Any]] = None,
    **fields: Any,
) -> dict:
    """Append one lifecycle event, enforcing the transition table and duties.

    - ``proposed`` -> approved | rejected; ``approved`` -> executed | rejected;
      ``executed`` -> evaluated; ``evaluated``/``rejected`` terminal.
    - Four-eyes: the approving actor must differ from ``decision_actor``.
    - ``executed`` requires ``execution_ref``; ``evaluated`` requires a
      non-empty ``evidence_refs``; ``rejected`` requires ``reason``.

    Returns the same decision object, mutated in place and re-verified.
    """
    verify_decision(decision)
    if event_type not in EVENT_TYPES:
        raise TraceSchemaError(f"event_type: 非法值 {event_type!r}")
    if event_type == "correction":
        raise IllegalTransition(
            "correction 事件必须经 correct() 追加（不改变 status，链接 correction_of）"
        )
    current = decision["status"]
    if event_type not in _TRANSITIONS[current]:
        legal = sorted(_TRANSITIONS[current]) or "（终态，无后继）"
        raise IllegalTransition(
            f"非法状态转移 {current} -> {event_type}；当前合法后继: {legal}"
        )
    if event_type == "approved" and actor == decision["decision_actor"]:
        raise SeparationOfDutiesError(
            f"四眼原则违规: 提案人 {decision['decision_actor']!r} 不得自行批准"
        )
    if event_type == "executed" and not _nonempty_str(fields.get("execution_ref")):
        raise MissingEventFieldError("executed 事件必须携带 execution_ref")
    if event_type == "evaluated":
        refs = fields.get("evidence_refs")
        if (
            not isinstance(refs, list)
            or not refs
            or not all(_nonempty_str(r) for r in refs)
        ):
            raise MissingEventFieldError(
                "evaluated 事件必须携带非空 evidence_refs 数组"
            )
    if event_type == "rejected" and not _nonempty_str(fields.get("reason")):
        raise MissingEventFieldError("rejected 事件必须携带 reason")
    event = _build_event(
        prev_event_id=decision["events"][-1]["event_id"],
        event_type=event_type,
        status=event_type,
        actor=actor,
        created_at=_timestamp(clock),
        fields=fields,
    )
    decision["events"].append(event)
    _promote(decision, event)
    decision["status"] = event["status"]
    verify_decision(decision)
    return decision


def correct(
    decision: dict,
    event_id: str,
    reason: str,
    *,
    actor: str,
    clock: Optional[Callable[[], Any]] = None,
    payload: Optional[dict] = None,
) -> dict:
    """Append a correction event pointing at ``event_id``; status unchanged.

    History is never rewritten: the correction is a new chain event whose
    ``correction_of`` links the corrected event and whose ``payload`` carries
    the amended content. Existing events are left byte-identical.
    """
    verify_decision(decision)
    if not _nonempty_str(reason):
        raise MissingEventFieldError("correction 事件必须携带 reason")
    target = next(
        (e for e in decision["events"] if e.get("event_id") == event_id), None
    )
    if target is None:
        known = ", ".join(e["event_id"] for e in decision["events"])
        raise UnknownEventError(
            f"correction_of 指向的事件不存在: {event_id!r}；已知事件: {known}"
        )
    fields = {"correction_of": event_id, "reason": reason}
    if payload is not None:
        fields["payload"] = payload
    event = _build_event(
        prev_event_id=decision["events"][-1]["event_id"],
        event_type="correction",
        status=decision["status"],
        actor=actor,
        created_at=_timestamp(clock),
        fields=fields,
    )
    decision["events"].append(event)
    verify_decision(decision)
    return decision


# ---------------------------------------------------------------------------
# Public API: JSONL ledger (append-only)
# ---------------------------------------------------------------------------

def _ledger_path(decisions_dir: str | Path, decision_id: str) -> Path:
    return Path(decisions_dir) / f"{decision_id}.jsonl"


def save_decision(decision: dict, decisions_dir: str | Path) -> Path:
    """Append the decision's events to ``<decision_id>.jsonl``, one per line.

    Append-only: when the file exists, every stored line must be byte-equal
    to the corresponding in-memory event; otherwise the write is refused.
    Only events missing from the file are appended. Returns the ledger path.
    """
    verify_decision(decision)
    path = _ledger_path(decisions_dir, decision["decision_id"])
    lines = [canonical_bytes(e) + b"\n" for e in decision["events"]]
    if path.exists():
        existing = path.read_bytes().splitlines()
        if len(existing) > len(decision["events"]):
            raise LedgerWriteError(
                f"{path}: 文件已有 {len(existing)} 行，超过内存文档的 "
                f"{len(decision['events'])} 个事件；请先 load_decision 再继续追加"
            )
        for idx, raw in enumerate(existing):
            stored = json.loads(raw)
            if stored != decision["events"][idx]:
                raise LedgerWriteError(
                    f"{path}: 第 {idx + 1} 行与内存事件不一致，既有记录已被改动，拒绝写入"
                )
        to_append = lines[len(existing):]
    else:
        to_append = lines
    if to_append:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "ab") as fh:
            for line in to_append:
                fh.write(line)
    return path


def load_decision(path: str | Path) -> dict:
    """Load a decision from its JSONL ledger, re-verifying the hash chain."""
    raw_lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not raw_lines:
        raise TraceSchemaError(f"{path}: 空账本文件")
    events = []
    for idx, line in enumerate(raw_lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TraceSchemaError(f"{path}: 第 {idx + 1} 行不是合法 JSON: {exc}") from exc
        if not isinstance(event, dict):
            raise TraceSchemaError(f"{path}: 第 {idx + 1} 行期望事件对象")
        events.append(event)
    doc = _document_from_events(events)
    verify_decision(doc)
    return doc


# ---------------------------------------------------------------------------
# Public API: shared-contract entry point
# ---------------------------------------------------------------------------

def record_from_decision_outcome(
    outcome: dict,
    *,
    ir_content_hash: str,
    clock: Optional[Callable[[], Any]] = None,
) -> dict:
    """Alias entry point: build a decision from a shared DecisionOutcome.

    Equivalent to :func:`new_decision`; ``ir_content_hash`` is pinned into
    ``state_snapshot_ref`` (a pre-existing different value is rejected).
    Missing contract fields raise ``KeyError`` naming the field.
    """
    if not _nonempty_str(ir_content_hash):
        raise TraceSchemaError("ir_content_hash: 期望非空字符串")
    _require_outcome_fields(outcome)
    normalized = dict(outcome)
    snapshot = dict(normalized.get("state_snapshot_ref") or {})
    recorded = snapshot.get("ir_content_hash")
    if recorded is not None and recorded != ir_content_hash:
        raise ValueError(
            f"state_snapshot_ref.ir_content_hash 与入参不一致: {recorded!r} != {ir_content_hash!r}"
        )
    snapshot["ir_content_hash"] = ir_content_hash
    normalized["state_snapshot_ref"] = snapshot
    return new_decision(normalized, clock=clock)


# ---------------------------------------------------------------------------
# Public API: ANF projection (ADR-002: lossy projection + pointers)
# ---------------------------------------------------------------------------

def _anf_outcome(decision: dict, payload: dict) -> dict:
    """Lossy outcome summary: outcome_refs -> success, evidence_refs -> partial."""
    outcome_refs = decision.get("outcome_refs") or []
    evidence_refs = decision.get("evidence_refs") or []
    if outcome_refs:
        result = "success"
        notes = "; ".join(outcome_refs)
    elif evidence_refs:
        result = "partial"
        notes = "; ".join(evidence_refs)
    else:
        result = "failure"
        notes = ""
    record: dict[str, Any] = {"result": result}
    if result == "success":
        record["metric_name"] = payload.get("metric_name", "")
        record["metric_value"] = decision["selected_option"]["metric_value"]
    if notes:
        record["notes"] = notes
    return record


def project_to_anf(decision: dict) -> dict:
    """Project a decision onto an ANF experience-record (ADR-002 OQ1-OQ3).

    Canonical id ``er-<decision_id>@<status>`` rides in
    ``context.decision_trace_ref``; the on-disk ``record_id`` is the
    mechanical projection ``exp-er-<decision_id>-<status>`` so it satisfies
    the ANF pattern ``^exp-[A-Za-z0-9._-]+$``. The projection is lossy and
    append-only per status: full fidelity stays in the DecisionTrace.
    """
    verify_decision(decision)
    did = decision["decision_id"]
    status = decision["status"]
    proposed = decision["events"][0]
    payload = proposed.get("payload") or {}
    selected = decision["selected_option"]
    metric_name = payload.get("metric_name") or "metric"
    direction = payload.get("metric_direction") or "n/a"

    rejected = decision["rejected_options_and_reasons"]
    uncertainty = decision["uncertainty"]
    if rejected:
        rejected_part = "rejected: " + ", ".join(
            f"{item['candidate_id']}({item['reason']})" for item in rejected
        )
    else:
        rejected_part = "rejected: none"
    summary = (
        f"{rejected_part}; uncertainty: metric_std={uncertainty.get('metric_std')}, "
        f"margin={uncertainty.get('margin')}, note={uncertainty.get('note') or '-'}"
    )
    if len(summary) > _ANF_SUMMARY_MAX:
        summary = summary[: _ANF_SUMMARY_MAX - 3] + "..."

    trust = (
        ANF_TRUST_VERIFIED
        if status in (STATUS_EXECUTED, STATUS_EVALUATED)
        else ANF_TRUST_INFERENCE
    )
    return {
        "record_id": f"exp-er-{did}-{status}",
        "task": decision["intent_ref"],
        "context": f"decision_trace_ref=er-{did}@{status}",
        "decision": (
            f"selected {selected['candidate_id']} [{selected['family']}] "
            f"{metric_name}={selected['metric_value']} ({direction})"
        ),
        "decision_summary": summary,
        "alternatives_considered": [
            f"{alt['candidate_id']} [{alt['family']}]" for alt in decision["alternatives"]
        ],
        "action": f"execute_experiment({selected['candidate_id']})",
        "execution_trace_refs": (
            [decision["execution_ref"]] if decision.get("execution_ref") else []
        ),
        "outcome": _anf_outcome(decision, payload),
        "classification": "internal",
        "trust_level": trust,
        "timestamp": decision["created_at"],
    }
