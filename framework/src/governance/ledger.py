"""Governance decision ledger (SPEC-007 §5): append-only JSONL chain.

Mirrors the P3 ``ir.decision_trace`` ledger style:

- decision identity is sha256-derived (``gvd-``/``gvg-`` prefixes; no uuid4,
  no wall clock — volatile fields never enter any hash);
- each stored decision is an event chain in ``<decision_id>.jsonl``, one
  canonical-JSON event per line; the first event links ``prev_event_id =
  "genesis"``, every later event links its predecessor;
- ``event_id`` = ``gev-`` + sha256 over the canonical event content
  (excluding ``event_id`` itself and the volatile ``recorded_at``);
- ``append_decision`` refuses to overwrite: every existing stored line must
  be byte-equal to the in-memory event, and a stored file longer than the
  in-memory chain is a conflict;
- ``load_decision`` recomputes every link and the embedded decision's own
  identity, so any tampered byte rejects the load.

Storage root: ``outputs/governance/`` (:data:`DEFAULT_LEDGER_DIR`).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from ..ir.decision_trace import GENESIS_PREV_EVENT_ID, canonical_bytes
from .errors import GovernanceError
from .promotion import PROMOTION_VOLATILE_FIELDS, compute_decision_id as compute_promotion_id
from .skill_gate import GATE_VOLATILE_FIELDS, compute_decision_id as compute_gate_id

__all__ = [
    "DEFAULT_LEDGER_DIR",
    "EVENT_ID_RE",
    "GENESIS_PREV_EVENT_ID",
    "GovernanceLedgerConflictError",
    "GovernanceLedgerError",
    "GovernanceLedgerTamperError",
    "KIND_DECISION_ID_PREFIX",
    "KIND_VOLATILE_FIELDS",
    "LEDGER_DECISION_STATUSES",
    "append_decision",
    "decision_event",
    "load_decision",
    "verify_decision_integrity",
]

DEFAULT_LEDGER_DIR = Path("outputs") / "governance"

DECISION_ID_RE_NOTE = "gvd-/gvg- + 12 hex"
EVENT_ID_RE_PREFIX = "gev-"

#: One hash-verified decision kind per governance module.
KIND_DECISION_ID_PREFIX = {
    "promotion": "gvd-",
    "skill_gate": "gvg-",
}

KIND_VOLATILE_FIELDS = {
    "promotion": PROMOTION_VOLATILE_FIELDS,
    "skill_gate": GATE_VOLATILE_FIELDS,
}

#: Stored statuses (informational mirror; verification is per-kind).
LEDGER_DECISION_STATUSES = ("approved_for_draft", "rejected", "pass", "fail")


class GovernanceLedgerError(GovernanceError):
    """Ledger contract violation."""


class GovernanceLedgerTamperError(GovernanceLedgerError):
    """Recomputed hash chain / decision identity does not match storage."""


class GovernanceLedgerConflictError(GovernanceLedgerError):
    """Append-only conflict: stored lines changed or diverged."""


def _kind_of(decision: dict) -> str:
    kind = decision.get("kind")
    if kind not in KIND_VOLATILE_FIELDS:
        raise GovernanceLedgerError(
            f"decision.kind 必须为 {sorted(KIND_VOLATILE_FIELDS)} 之一，得到 {kind!r}"
        )
    return kind


def verify_decision_integrity(decision: dict) -> None:
    """Recompute the embedded decision's sha256 identity in place-agnostically."""
    kind = _kind_of(decision)
    decision_id = decision.get("decision_id")
    expected_prefix = KIND_DECISION_ID_PREFIX[kind]
    if not isinstance(decision_id, str) or not decision_id.startswith(expected_prefix):
        raise GovernanceLedgerError(
            f"decision_id 必须为 {expected_prefix}+12hex（{DECISION_ID_RE_NOTE}），得到 {decision_id!r}"
        )
    if kind == "promotion":
        recomputed = compute_promotion_id(decision)
    else:
        recomputed = compute_gate_id(decision)
    if recomputed != decision_id:
        raise GovernanceLedgerTamperError(
            f"decision_id 与内容不符（期望 {recomputed}，存储 {decision_id}）：决策被篡改"
        )


def _event_content(event: dict) -> dict:
    """Event content covered by the chain hash (excludes the volatile pair)."""
    return {k: v for k, v in event.items() if k not in ("event_id", "recorded_at")}


def compute_event_id(event: dict) -> str:
    """``gev-`` + sha256 over canonical content including ``prev_event_id``."""
    return EVENT_ID_RE_PREFIX + hashlib.sha256(canonical_bytes(_event_content(event))).hexdigest()


def decision_event(
    decision: dict,
    *,
    prev_event_id: str = GENESIS_PREV_EVENT_ID,
    recorded_at: Optional[str] = None,
    clock: Optional[Callable[[], Any]] = None,
) -> dict:
    """Build one ledger event wrapping a verified governance decision."""
    verify_decision_integrity(decision)
    if not isinstance(prev_event_id, str) or not (
        prev_event_id == GENESIS_PREV_EVENT_ID
        or (prev_event_id.startswith(EVENT_ID_RE_PREFIX) and len(prev_event_id) == 68)
    ):
        raise GovernanceLedgerError(
            f"prev_event_id 必须为 'genesis' 或 {EVENT_ID_RE_PREFIX}+64hex，得到 {prev_event_id!r}"
        )
    if recorded_at is None:
        recorded_at = (
            clock().isoformat()
            if clock is not None
            else datetime.now(timezone.utc).isoformat()
        )
    event = {
        "prev_event_id": prev_event_id,
        "decision_id": decision["decision_id"],
        "kind": decision["kind"],
        "status": decision["status"],
        "decision": decision,
        "recorded_at": recorded_at,
    }
    event["event_id"] = compute_event_id(event)
    return event


def _ledger_path(ledger_dir, decision_id: str) -> Path:
    return Path(ledger_dir) / f"{decision_id}.jsonl"


def append_decision(
    decision: dict,
    ledger_dir=DEFAULT_LEDGER_DIR,
    *,
    recorded_at: Optional[str] = None,
) -> Path:
    """Append the decision as one verified event; append-only, idempotent.

    When the ledger file exists, every stored line must be byte-equal to the
    corresponding in-memory event; divergence raises
    :class:`GovernanceLedgerConflictError` (append-only: stored history is
    never rewritten). Returns the ledger path.
    """
    verify_decision_integrity(decision)
    path = _ledger_path(ledger_dir, decision["decision_id"])
    if path.exists():
        existing = path.read_bytes().splitlines()
        if len(existing) > 1:
            raise GovernanceLedgerConflictError(
                f"{path}: 文件已有 {len(existing)} 行；治理决策是单事件文档，"
                f"多行意味着存储被外部追加，拒绝写入"
            )
        if existing:
            stored = json.loads(existing[0])
            event = decision_event(decision, recorded_at=stored.get("recorded_at"))
            if stored != event:
                raise GovernanceLedgerConflictError(
                    f"{path}: 既有首行与内存事件不一致，既有记录已被改动，拒绝写入"
                )
            return path
    event = decision_event(decision, recorded_at=recorded_at)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as fh:
        fh.write(canonical_bytes(event) + b"\n")
    return path


def load_decision(path) -> dict:
    """Load a decision ledger, re-verifying the chain and embedded identity.

    Any tampered stored line (edited decision content, broken prev link,
    recomputed event_id mismatch) raises
    :class:`GovernanceLedgerTamperError`.
    """
    raw_lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not raw_lines:
        raise GovernanceLedgerError(f"{path}: 空账本文件")
    events = []
    prev_expected = GENESIS_PREV_EVENT_ID
    for idx, line in enumerate(raw_lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GovernanceLedgerTamperError(
                f"{path}: 第 {idx + 1} 行不是合法 JSON: {exc}"
            ) from exc
        if not isinstance(event, dict):
            raise GovernanceLedgerTamperError(f"{path}: 第 {idx + 1} 行期望事件对象")
        if event.get("prev_event_id") != prev_expected:
            raise GovernanceLedgerTamperError(
                f"{path}: 第 {idx + 1} 行 prev_event_id 链断裂"
                f"（期望 {prev_expected!r}，存储 {event.get('prev_event_id')!r}）"
            )
        if event.get("event_id") != compute_event_id(event):
            raise GovernanceLedgerTamperError(
                f"{path}: 第 {idx + 1} 行 event_id 与内容不符（链哈希篡改）"
            )
        decision = event.get("decision")
        if not isinstance(decision, dict) or decision.get("decision_id") != event.get("decision_id"):
            raise GovernanceLedgerTamperError(
                f"{path}: 第 {idx + 1} 行 decision.decision_id 与事件不一致"
            )
        verify_decision_integrity(decision)
        prev_expected = event["event_id"]
        events.append(event)
    first = events[0]["decision"]
    return {
        "decision_id": first["decision_id"],
        "kind": first["kind"],
        "status": first["status"],
        "decision": first,
        "events": events,
    }
