"""Candidate comparison and negative rejection (SPEC-005, plan P3-B).

``decide`` consumes an ExperimentIR plus one run result per candidate and
projects them into the shared ``DecisionOutcome`` dict (plan baseline
v0.1 §5/§12; boundary fixed by docs/adr/ADR-002). The A-line DecisionTrace
(``ir/decision_trace.py``, separate worktree) consumes this dict verbatim:
the field set below is the contract — nothing added, nothing dropped.

Rules (plan G3/G4, §10 preregistered thresholds):

1. Hard gates first (``ir.experiment_ir.run_gates``): any gate "fail" ⇒
   every candidate is rejected with ``constraint_gate_failed`` and
   ``selected_option`` is ``None``. No ranking happens under an illegal
   contract.
2. Winner = direction-optimal candidate metric value; ties keep the
   earlier candidate in IR order (deterministic, never set order).
3. Negative rejection against the preregistered baseline
   (``baseline_ref.score``, same validation protocol per G5):
   - strictly worse than baseline under direction ⇒ ``worse_than_baseline``
   - improvement below the ``soft_preferences`` entry
     ``{"type": "min_improvement_over_baseline", "threshold": <number>}``
     ⇒ ``below_min_improvement`` (the threshold is preregistered in the
     hash-frozen IR — decide reads it, never invents one)
   - every other loser ⇒ ``lost_to_selected`` (detail carries the gap).

When even the direction-optimal candidate is rejected the whole decision
is a rejection: ``selected_option`` is ``None``. The "decision rejected"
lifecycle semantics belong upstream (A-line DecisionTrace status machine);
``decide`` only reports facts.

Dependency policy: pure stdlib at runtime (``RunResult`` is imported under
``TYPE_CHECKING`` only and results are duck-typed via their manifest), so
this module never pulls numpy/sklearn/mlflow. Zero imports from the A-line
decision-trace files — the dict contract is the only interface.
"""
from __future__ import annotations

import copy
import math
import statistics
from typing import TYPE_CHECKING, Any, Mapping

from .experiment_ir import run_gates

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .runner import RunResult

__all__ = [
    "DECISION_CONTRACT_FIELDS",
    "MIN_IMPROVEMENT_TYPE",
    "REASON_BELOW_MIN_IMPROVEMENT",
    "REASON_CONSTRAINT_GATE_FAILED",
    "REASON_LOST_TO_SELECTED",
    "REASON_WORSE_THAN_BASELINE",
    "decide",
]

#: Rejection reason vocabulary (shared contract; closed).
REASON_CONSTRAINT_GATE_FAILED = "constraint_gate_failed"
REASON_WORSE_THAN_BASELINE = "worse_than_baseline"
REASON_BELOW_MIN_IMPROVEMENT = "below_min_improvement"
REASON_LOST_TO_SELECTED = "lost_to_selected"

#: ``soft_preferences`` entry type carrying the preregistered minimum
#: improvement threshold: {"type": ..., "threshold": <number>}.
MIN_IMPROVEMENT_TYPE = "min_improvement_over_baseline"

#: The exact DecisionOutcome field set (plan baseline v0.1 §5/§12): B line
#: produces it, A line consumes it. Kept as data so tests can pin the
#: contract shape ("one field more or less" is a failure).
DECISION_CONTRACT_FIELDS = frozenset(
    {
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
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _min_improvement_threshold(ir: Mapping[str, Any]) -> float | None:
    """Preregistered ``min_improvement_over_baseline`` threshold, if any.

    The ExperimentIR schema deliberately leaves ``soft_preferences`` items
    free-form, so this reader is defensive: entries without the expected
    ``type``/numeric ``threshold`` shape are ignored (a schema-legal IR can
    never crash the decision). Multiple matching entries collapse to the
    strictest (largest) threshold — the decision must satisfy every
    preregistered preference simultaneously.
    """
    thresholds: list[float] = []
    for pref in ir.get("soft_preferences") or []:
        if not isinstance(pref, Mapping):
            continue
        if pref.get("type") != MIN_IMPROVEMENT_TYPE:
            continue
        threshold = pref.get("threshold")
        if _is_number(threshold) and math.isfinite(float(threshold)):
            thresholds.append(float(threshold))
    return max(thresholds) if thresholds else None


def _fold_metric_std(canonical: Mapping[str, Any], metric_name: str) -> float | None:
    """Sample stdev over fold-level metric values, when the manifest carries
    them (``canonical["fold_metrics"][metric_name] = [v_fold0, ...]``).
    Anything else (absent, single fold, non-numeric/non-finite entries)
    yields ``None`` — callers must report the gap in ``uncertainty.note``."""
    fold_metrics = canonical.get("fold_metrics")
    if not isinstance(fold_metrics, Mapping):
        return None
    values = fold_metrics.get(metric_name)
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        return None
    numbers: list[float] = []
    for value in values:
        if not _is_number(value):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        numbers.append(number)
    return statistics.stdev(numbers)


def _improvement(value: float, baseline: float, direction: str) -> float:
    """Signed improvement of ``value`` over ``baseline`` — positive means
    better under the metric direction."""
    return baseline - value if direction == "minimize" else value - baseline


def _is_better(value: float, reference: float, direction: str) -> bool:
    return value < reference if direction == "minimize" else value > reference


def _worse_detail(metric_name: str, value: float, baseline: float,
                  direction: str) -> str:
    return (
        f"{metric_name}={value} is worse than baseline {baseline} "
        f"(direction={direction})"
    )


def _below_detail(metric_name: str, value: float, baseline: float,
                  improvement: float, threshold: float, direction: str) -> str:
    return (
        f"{metric_name}={value} improves on baseline {baseline} by "
        f"{improvement}, below the preregistered minimum improvement "
        f"{threshold} (direction={direction})"
    )


def _lost_detail(metric_name: str, value: float, winner_id: str,
                 winner_value: float, gap: float, direction: str) -> str:
    return (
        f"lost to {winner_id}: {metric_name}={value} vs selected "
        f"{winner_value}, gap={gap} (direction={direction})"
    )


def _candidate_option(candidate: Mapping[str, Any], value: float) -> dict:
    """Shared shape of ``selected_option`` entries and ``alternatives``."""
    return {
        "candidate_id": candidate["candidate_id"],
        "family": candidate["family"],
        "params": copy.deepcopy(candidate["params"]),
        "metric_value": value,
    }


def _state_snapshot(
    ir: Mapping[str, Any],
    results: Mapping[str, Any],
    ordered_ids: list[str],
) -> dict:
    """Decided-IR hash plus winner-manifest data/code triple; without a
    winner, the first available run manifest (IR candidate order), else the
    IR's own declarations. ir_content_hash always refers to the IR under
    decision (candidates may execute under rotated revisions), so the trace
    stays anchored to the authorized contract."""
    for candidate_id in ordered_ids:
        run = results.get(candidate_id)
        if run is None:
            continue
        canonical = run.manifest["canonical"]
        return {
            "ir_content_hash": ir.get("content_hash", ""),
            "data_sha256": canonical["data_sha256"],
            "code_version": canonical["code_version"],
        }
    dataset = ir.get("dataset_snapshot_ref")
    return {
        "ir_content_hash": ir.get("content_hash", ""),
        "data_sha256": dataset.get("sha256", "") if isinstance(dataset, Mapping) else "",
        "code_version": "unknown",
    }


def _validate_results(
    ir: Mapping[str, Any],
    results: Mapping[str, "RunResult"],
    ordered_ids: list[str],
    *, require_full: bool,
) -> None:
    unknown = sorted(set(results) - set(ordered_ids))
    if unknown:
        raise ValueError(
            "results carry unknown candidate_id(s): "
            f"{unknown}; IR candidates: {ordered_ids}"
        )
    if require_full:
        missing = [cid for cid in ordered_ids if cid not in results]
        if missing:
            raise ValueError(
                "decide requires one RunResult per candidate; missing: "
                f"{missing}"
            )


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def decide(
    ir: Mapping[str, Any],
    results: Mapping[str, "RunResult"],
) -> dict:
    """Project executed candidate runs into the shared DecisionOutcome.

    Args:
        ir: ExperimentIR dict (as compiled; keys per SPEC-001 schema).
        results: one run result per candidate, keyed by
            ``candidate_id`` (each produced by
            ``ir.runner.execute_experiment`` with that candidate first).

    Returns:
        The DecisionOutcome dict — exactly ``DECISION_CONTRACT_FIELDS``,
        nothing more, nothing less.

    Raises:
        ValueError: unknown/missing result keys, candidate/manifest family
            mismatch, a manifest lacking the IR metric, or an IR without
            candidates.
    """
    candidates = ir.get("candidates") or []
    if not candidates:
        raise ValueError("ExperimentIR carries no candidates to decide between")
    ordered_ids = [candidate["candidate_id"] for candidate in candidates]
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError(f"duplicate candidate_id in IR: {ordered_ids}")

    metric = ir["metric_definition_ref"]
    metric_name = metric["name"]
    direction = metric["direction"]

    # ① Hard gates first: an illegal contract is never ranked.
    gates = run_gates(ir)
    constraint_check = [
        {"gate_id": gate.gate_id, "status": gate.status} for gate in gates
    ]
    failed_gates = [gate for gate in gates if gate.status == "fail"]

    if failed_gates:
        _validate_results(ir, results, ordered_ids, require_full=False)
        detail = "; ".join(f"{g.gate_id}: {g.reason}" for g in failed_gates)
        rejected = [
            {
                "candidate_id": candidate_id,
                "reason": REASON_CONSTRAINT_GATE_FAILED,
                "detail": detail,
            }
            for candidate_id in ordered_ids
        ]
        return {
            "intent_ref": ir.get("hypothesis_ref") or ir["experiment_id"],
            "state_snapshot_ref": _state_snapshot(ir, results, ordered_ids),
            "alternatives": [
                _candidate_option(candidate, float(
                    results[candidate["candidate_id"]].manifest["canonical"]["metrics"][metric_name]
                ))
                for candidate in candidates
                if (
                    candidate["candidate_id"] in results
                    and metric_name in results[candidate["candidate_id"]]
                    .manifest["canonical"].get("metrics", {})
                )
            ],
            "constraint_check": constraint_check,
            "baseline_ref": copy.deepcopy(dict(ir["baseline_ref"])),
            "selected_option": None,
            "rejected_options_and_reasons": rejected,
            "uncertainty": {
                "metric_std": None,
                "margin": None,
                "note": (
                    "hard gate(s) failed, no ranking performed; margin and "
                    "metric_std undefined"
                ),
            },
            "decision_actor": ir.get("authorization", {}).get("authorized_by", ""),
            "authorization_ref": ir.get("authorization", {}).get("authorization_ref", ""),
            "evidence_refs": [
                results[candidate_id].oof_sha256
                for candidate_id in ordered_ids
                if candidate_id in results
            ],
            "metric_name": metric_name,
            "metric_direction": direction,
        }

    # ② Gates passed: full, consistent candidate coverage is required.
    _validate_results(ir, results, ordered_ids, require_full=True)
    for candidate in candidates:
        canonical = results[candidate["candidate_id"]].manifest["canonical"]
        if canonical.get("model_family") != candidate["family"]:
            raise ValueError(
                f"result for candidate {candidate['candidate_id']!r} reports "
                f"family {canonical.get('model_family')!r}, IR says "
                f"{candidate['family']!r}"
            )
        if metric_name not in canonical.get("metrics", {}):
            raise ValueError(
                f"result for candidate {candidate['candidate_id']!r} lacks "
                f"metric {metric_name!r} in its manifest"
            )

    values = {
        candidate["candidate_id"]: float(
            results[candidate["candidate_id"]].manifest["canonical"]["metrics"][metric_name]
        )
        for candidate in candidates
    }
    baseline = float(ir["baseline_ref"]["score"])
    threshold = _min_improvement_threshold(ir)

    # ③ Winner: direction-optimal value; ties keep the earlier IR candidate.
    winner_id = ordered_ids[0]
    for candidate_id in ordered_ids[1:]:
        if _is_better(values[candidate_id], values[winner_id], direction):
            winner_id = candidate_id
    winner_value = values[winner_id]
    winner_improvement = _improvement(winner_value, baseline, direction)

    # ④ Negative rejection, uniform rule chain (winner rule first).
    selected_option: dict | None = None
    rejected: list[dict] = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        value = values[candidate_id]
        improvement = _improvement(value, baseline, direction)
        if improvement < 0:
            rejected.append({
                "candidate_id": candidate_id,
                "reason": REASON_WORSE_THAN_BASELINE,
                "detail": _worse_detail(metric_name, value, baseline, direction),
            })
        elif threshold is not None and improvement < threshold:
            rejected.append({
                "candidate_id": candidate_id,
                "reason": REASON_BELOW_MIN_IMPROVEMENT,
                "detail": _below_detail(
                    metric_name, value, baseline, improvement, threshold, direction
                ),
            })
        elif candidate_id == winner_id:
            selected_option = _candidate_option(candidate, value)
        else:
            gap = (
                value - winner_value if direction == "minimize"
                else winner_value - value
            )
            rejected.append({
                "candidate_id": candidate_id,
                "reason": REASON_LOST_TO_SELECTED,
                "detail": _lost_detail(
                    metric_name, value, winner_id, winner_value, gap, direction
                ),
            })

    winner_canonical = results[winner_id].manifest["canonical"]
    metric_std = (
        _fold_metric_std(winner_canonical, metric_name)
        if selected_option is not None
        else None
    )
    notes = []
    if selected_option is None:
        notes.append("no option selected; margin undefined")
    else:
        notes.append(f"margin against baseline {baseline}")
    if metric_std is None:
        notes.append(
            "no fold-level metrics (manifest.canonical.fold_metrics) "
            "available; metric_std unset"
        )
    else:
        notes.append(
            f"metric_std is the sample stdev over fold-level {metric_name} values"
        )

    return {
        "intent_ref": ir.get("hypothesis_ref") or ir["experiment_id"],
        "state_snapshot_ref": {
            "ir_content_hash": ir.get("content_hash", ""),
            "data_sha256": winner_canonical["data_sha256"],
            "code_version": winner_canonical["code_version"],
        } if selected_option is not None else _state_snapshot(ir, results, ordered_ids),
        "alternatives": [
            _candidate_option(candidate, values[candidate["candidate_id"]])
            for candidate in candidates
        ],
        "constraint_check": constraint_check,
        "baseline_ref": copy.deepcopy(dict(ir["baseline_ref"])),
        "selected_option": selected_option,
        "rejected_options_and_reasons": rejected,
        "uncertainty": {
            "metric_std": metric_std,
            "margin": (
                abs(winner_value - baseline) if selected_option is not None else None
            ),
            "note": "; ".join(notes),
        },
        "decision_actor": ir.get("authorization", {}).get("authorized_by", ""),
        "authorization_ref": ir.get("authorization", {}).get("authorization_ref", ""),
        "evidence_refs": [
            results[candidate_id].oof_sha256 for candidate_id in ordered_ids
        ],
        "metric_name": metric_name,
        "metric_direction": direction,
    }
