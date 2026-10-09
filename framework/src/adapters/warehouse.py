"""ADAPTER-003 (仓储): warehouse estimate exchange (SPEC-006).

Same EstimateEnvelope contract, ``warehouse_demand`` kind. The decision side
is intentionally an explicitly-labelled **reference stub**:
:func:`stub_decision_engine` is a deterministic pure function (accept/reject +
capacity-violation count) standing in for the real warehouse decision engine.
The real engine integration point is BLOCKED on the engine repo — see
SPEC-006 §8 (blocked-on-engine-repo). Zero solver dependency, like every
module under ``framework/src/adapters``.
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional

from .estimates import (
    ESTIMATE_KIND_WAREHOUSE_DEMAND,
    EnvelopeError,
    envelope_from_oof,
)

__all__ = [
    "ADAPTER_ID",
    "STUB_ENGINE_ID",
    "warehouse_envelope_from_oof",
    "stub_decision_engine",
]

ADAPTER_ID = "ADAPTER-003"

#: Explicit reference-stub identity: outputs are deterministic pure functions
#: of (envelope, capacity); they are NOT warehouse business decisions.
STUB_ENGINE_ID = "reference-stub/0.1.0"

#: Tolerance for float capacity accounting (values are rounded to 6 decimals).
_CAPACITY_EPS = 1e-9


def warehouse_envelope_from_oof(
    oof_csv_path,
    group_col: str,
    ir: Mapping,
    evidence: Mapping,
    *,
    value_unit: str = "units_per_day",
    clock: Optional[Callable] = None,
) -> dict:
    """Build a ``warehouse_demand`` EstimateEnvelope from a truth-bearing OOF CSV.

    Same contract as :func:`adapters.estimates.envelope_from_oof`; no weekday
    quantization (the p80 rule is a PJP-weekday concept).
    """
    return envelope_from_oof(
        oof_csv_path,
        group_col,
        ir,
        evidence,
        estimate_kind=ESTIMATE_KIND_WAREHOUSE_DEMAND,
        value_unit=value_unit,
        quantization=None,
        clock=clock,
    )


def stub_decision_engine(envelope: Mapping, *, capacity: int) -> dict:
    """Deterministic reference stub standing in for the real decision engine.

    Greedy allocation in stable-code (unit_id) order against a total daily
    ``capacity`` (same unit as the envelope's ``value_unit``): a unit is
    accepted iff its remaining capacity share fits; otherwise it is rejected.

    Returns a dict that self-identifies as a stub::

        {"engine": "reference-stub/0.1.0", "stub": true, "deterministic": true,
         "decision": "accept"|"reject", "capacity": int,
         "capacity_violations": <rejected count>, "accepted_units": [...],
         "rejected_units": [...], "envelope_id": ..., "capacity_unit": ...}

    Pure function of (envelope, capacity): same input -> byte-identical
    output; no clock, no randomness. This is an interface placeholder, not a
    warehouse decision — see SPEC-006 §8.
    """
    if envelope.get("estimate_kind") != ESTIMATE_KIND_WAREHOUSE_DEMAND:
        raise EnvelopeError(
            f"stub_decision_engine 需要 {ESTIMATE_KIND_WAREHOUSE_DEMAND} 信封, "
            f"得到 {envelope.get('estimate_kind')!r}"
        )
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
        raise ValueError("capacity 必须是非负 int（units_per_day）")
    units = sorted(envelope.get("units", ()), key=lambda u: str(u["unit_id"]))
    remaining = float(capacity)
    accepted, rejected = [], []
    for unit in units:
        value = float(unit["value"])
        if value <= remaining + _CAPACITY_EPS:
            accepted.append(str(unit["unit_id"]))
            remaining -= value
        else:
            rejected.append(str(unit["unit_id"]))
    return {
        "engine": STUB_ENGINE_ID,
        "stub": True,
        "deterministic": True,
        "decision": "accept" if not rejected else "reject",
        "capacity": capacity,
        "capacity_violations": len(rejected),
        "accepted_units": accepted,
        "rejected_units": rejected,
        "envelope_id": envelope.get("envelope_id"),
        "capacity_unit": envelope.get("value_unit"),
    }
