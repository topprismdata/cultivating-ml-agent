"""SPEC-005 / plan P3-B decision tests: candidate comparison and negative
rejection.

Covers the DecisionOutcome contract shape (exactly the shared field set),
direction-correct winner selection on a real two-candidate replay (s6e5-style
fixture, ridge + hgb, one ``execute_experiment`` per candidate), the three
negative-rejection buckets against the preregistered baseline
(``worse_than_baseline`` / ``below_min_improvement`` / ``lost_to_selected``),
the hard-gate short-circuit (``constraint_gate_failed``, no ranking), the
fold-level ``metric_std`` path, and input validation.
"""
import copy
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.src.ir.decide import (
    DECISION_CONTRACT_FIELDS,
    REASON_BELOW_MIN_IMPROVEMENT,
    REASON_CONSTRAINT_GATE_FAILED,
    REASON_LOST_TO_SELECTED,
    REASON_WORSE_THAN_BASELINE,
    decide,
)
from framework.src.ir.experiment_ir import compute_content_hash, load_ir
from framework.src.ir.runner import RunResult, execute_experiment

CASE_DIR = ROOT / "replays" / "s6e5-style"
EXAMPLES_DIR = ROOT / "schemas" / "experiment-ir" / "0.1.0" / "examples"
BASELINE_SCORE = 0.35  # preregistered in the s6e5-style IR


def _fake_run(metric_value: float, *, family: str = "ridge",
              fold_metrics: dict | None = None) -> RunResult:
    """Hand-built RunResult (no retraining) with a SPEC-003-shaped manifest."""
    canonical = {
        "schema_version": "0.1.0",
        "ir_content_hash": "a" * 64,
        "code_version": "test-code-version",
        "data_sha256": "b" * 64,
        "seed": 42,
        "strategy": "time_based",
        "n_folds": 1,
        "fold_sizes": [{"train": 1948, "val": 28}],
        "metrics": {"rmsle": metric_value},
        "oof_sha256": "c" * 64,
        "model_family": family,
        "model_params": {},
    }
    if fold_metrics is not None:
        canonical["fold_metrics"] = fold_metrics
    return RunResult(
        ir={},
        manifest={"canonical": canonical, "volatile": {}},
        evidence={},
        oof_path=Path("oof.csv"),
        oof_sha256="c" * 64,
    )


@pytest.fixture(scope="module")
def replay(tmp_path_factory):
    """Real double-candidate replay: ridge and hgb on the s6e5-style fixture.

    ``execute_experiment`` runs ``candidates[0]`` only, so the second
    candidate is executed from a rotated IR revision (candidates reversed,
    ``content_hash`` recomputed — a new revision, not a mutation).
    """
    project_root = tmp_path_factory.mktemp("decide-replay")
    ir = load_ir(CASE_DIR / "experiment.ir.json")
    run_ridge = execute_experiment(ir, project_root / "ridge", seed=42,
                                   data_root=ROOT)

    ir_hgb = copy.deepcopy(ir)
    ir_hgb["candidates"] = list(reversed(ir_hgb["candidates"]))
    ir_hgb["content_hash"] = compute_content_hash(ir_hgb)
    run_hgb = execute_experiment(ir_hgb, project_root / "hgb", seed=42,
                                 data_root=ROOT)

    results = {"ridge-v1": run_ridge, "hgb-v1": run_hgb}
    return ir, results, run_ridge, run_hgb


# ---------- contract shape + direction-correct winner ----------

def test_outcome_contract_shape_and_winner(replay):
    ir, results, run_ridge, run_hgb = replay
    outcome = decide(ir, results)

    # Contract: one field more or less is a failure.
    assert set(outcome) == DECISION_CONTRACT_FIELDS
    assert len(DECISION_CONTRACT_FIELDS) == 13

    # Winner is the direction-optimal candidate (rmsle, minimize).
    ridge_value = run_ridge.canonical["metrics"]["rmsle"]
    hgb_value = run_hgb.canonical["metrics"]["rmsle"]
    best_value = min(ridge_value, hgb_value)
    winner_id = "ridge-v1" if best_value == ridge_value else "hgb-v1"
    winner_run = results[winner_id]

    selected = outcome["selected_option"]
    assert selected is not None
    assert set(selected) == {"candidate_id", "family", "params", "metric_value"}
    assert selected["candidate_id"] == winner_id
    assert selected["metric_value"] == best_value
    assert selected["family"] == winner_run.canonical["model_family"]
    assert selected["params"] == winner_run.canonical["model_params"]

    assert outcome["metric_name"] == "rmsle"
    assert outcome["metric_direction"] == "minimize"

    # Both candidates are alternatives; both beat the 0.35 baseline here,
    # so the loser is lost_to_selected with a gap detail.
    assert [alt["candidate_id"] for alt in outcome["alternatives"]] == [
        "ridge-v1", "hgb-v1"
    ]
    for alt in outcome["alternatives"]:
        assert set(alt) == {"candidate_id", "family", "params", "metric_value"}
        assert alt["metric_value"] == results[alt["candidate_id"]].canonical["metrics"]["rmsle"]

    rejected = outcome["rejected_options_and_reasons"]
    assert len(rejected) == 1
    loser = rejected[0]
    loser_value = max(ridge_value, hgb_value)
    if loser_value > BASELINE_SCORE:
        assert loser["reason"] == REASON_WORSE_THAN_BASELINE
    else:
        assert loser["reason"] == REASON_LOST_TO_SELECTED
        assert f"gap={loser_value - best_value}" in loser["detail"]
        assert winner_id in loser["detail"]
    assert set(loser) == {"candidate_id", "reason", "detail"}

    # Gates all ran and none failed.
    assert [c["gate_id"] for c in outcome["constraint_check"]] == [
        "G1_TEMPORAL_AVAILABILITY",
        "G2_SPLIT_SEPARATION",
        "G3_METRIC_DEFINITION",
        "G4_BUDGET_BOUNDS",
        "G5_BASELINE_SAME_PROTOCOL",
        "G6_RUN_AUTHORIZATION",
    ]
    assert all(
        c["status"] in ("pass", "not_applicable")
        for c in outcome["constraint_check"]
    )


def test_state_refs_uncertainty_and_provenance(replay):
    ir, results, run_ridge, run_hgb = replay
    outcome = decide(ir, results)

    best_value = min(
        run_ridge.canonical["metrics"]["rmsle"],
        run_hgb.canonical["metrics"]["rmsle"],
    )
    winner_run = run_ridge if best_value == run_ridge.canonical["metrics"]["rmsle"] else run_hgb

    # state_snapshot_ref: ir_content_hash anchors the decided IR (candidates
    # may execute under rotated revisions); data/code come from the winner.
    assert outcome["state_snapshot_ref"] == {
        "ir_content_hash": ir["content_hash"],
        "data_sha256": winner_run.canonical["data_sha256"],
        "code_version": winner_run.canonical["code_version"],
    }

    # uncertainty: margin against the preregistered baseline; the SPEC-003
    # manifest carries no fold-level metrics, so metric_std is None + note.
    uncertainty = outcome["uncertainty"]
    assert set(uncertainty) == {"metric_std", "margin", "note"}
    assert uncertainty["margin"] == abs(best_value - BASELINE_SCORE)
    assert uncertainty["metric_std"] is None
    assert uncertainty["note"]

    # evidence_refs: every candidate's oof_sha256, in IR candidate order.
    assert outcome["evidence_refs"] == [
        results["ridge-v1"].oof_sha256,
        results["hgb-v1"].oof_sha256,
    ]

    # Authorization passthrough from the IR.
    assert outcome["decision_actor"] == ir["authorization"]["authorized_by"]
    assert outcome["authorization_ref"] == ir["authorization"]["authorization_ref"]
    assert outcome["baseline_ref"] == {
        "baseline_id": ir["baseline_ref"]["baseline_id"],
        "protocol_ref": ir["baseline_ref"]["protocol_ref"],
        "score": BASELINE_SCORE,
    }

    # intent_ref: hypothesis_ref when present, else experiment_id.
    assert outcome["intent_ref"] == ir["experiment_id"]
    ir_with_intent = copy.deepcopy(ir)
    ir_with_intent["hypothesis_ref"] = "hypothesis/2026-10-09-h1.md"
    assert decide(ir_with_intent, results)["intent_ref"] == \
        "hypothesis/2026-10-09-h1.md"


# ---------- negative rejection: worse_than_baseline ----------

def test_worse_than_baseline_rejects_everything():
    ir = load_ir(CASE_DIR / "experiment.ir.json")
    # Hand-built results (no retraining): the best candidate (hgb, 0.36) is
    # still strictly worse than the 0.35 baseline under minimize.
    results = {
        "ridge-v1": _fake_run(0.40),
        "hgb-v1": _fake_run(0.36, family="hgb"),
    }
    outcome = decide(ir, results)

    assert outcome["selected_option"] is None
    assert set(outcome) == DECISION_CONTRACT_FIELDS
    reasons = {
        entry["candidate_id"]: entry["reason"]
        for entry in outcome["rejected_options_and_reasons"]
    }
    assert reasons == {
        "ridge-v1": REASON_WORSE_THAN_BASELINE,
        "hgb-v1": REASON_WORSE_THAN_BASELINE,
    }
    assert outcome["uncertainty"]["margin"] is None
    assert outcome["uncertainty"]["metric_std"] is None
    # Without a winner the state still anchors the decided IR (ir_content_hash);
    # data/code come from the first available manifest (IR candidate order).
    assert outcome["state_snapshot_ref"] == {
        "ir_content_hash": ir["content_hash"],
        "data_sha256": "b" * 64,
        "code_version": "test-code-version",
    }


# ---------- negative rejection: below_min_improvement ----------

def test_below_min_improvement_threshold():
    ir = load_ir(CASE_DIR / "experiment.ir.json")
    ir["soft_preferences"] = [
        {"type": "min_improvement_over_baseline", "threshold": 0.05}
    ]
    ir["content_hash"] = compute_content_hash(ir)
    results = {
        "ridge-v1": _fake_run(0.349),            # improvement 0.001 < 0.05
        "hgb-v1": _fake_run(0.36, family="hgb"),  # worse than baseline
    }
    outcome = decide(ir, results)

    assert outcome["selected_option"] is None
    reasons = {
        entry["candidate_id"]: entry["reason"]
        for entry in outcome["rejected_options_and_reasons"]
    }
    assert reasons == {
        "ridge-v1": REASON_BELOW_MIN_IMPROVEMENT,
        "hgb-v1": REASON_WORSE_THAN_BASELINE,
    }
    assert outcome["uncertainty"]["margin"] is None


def test_threshold_satisfied_keeps_selection():
    ir = load_ir(CASE_DIR / "experiment.ir.json")
    ir["soft_preferences"] = [
        {"type": "min_improvement_over_baseline", "threshold": 0.005}
    ]
    ir["content_hash"] = compute_content_hash(ir)
    results = {
        "ridge-v1": _fake_run(0.34),              # improvement 0.01 >= 0.005
        "hgb-v1": _fake_run(0.341, family="hgb"),  # beats baseline, loses to ridge
    }
    outcome = decide(ir, results)

    selected = outcome["selected_option"]
    assert selected is not None
    assert selected["candidate_id"] == "ridge-v1"
    assert selected["metric_value"] == 0.34
    reasons = {
        entry["candidate_id"]: entry["reason"]
        for entry in outcome["rejected_options_and_reasons"]
    }
    assert reasons == {"hgb-v1": REASON_LOST_TO_SELECTED}
    detail = next(
        entry["detail"]
        for entry in outcome["rejected_options_and_reasons"]
        if entry["candidate_id"] == "hgb-v1"
    )
    assert f"gap={0.341 - 0.34}" in detail  # 差距 detail（按位 float 文本）
    assert outcome["uncertainty"]["margin"] == abs(0.34 - BASELINE_SCORE)


# ---------- uncertainty: fold-level metric std ----------

def test_metric_std_from_fold_metrics():
    ir = load_ir(CASE_DIR / "experiment.ir.json")
    fold_values = [0.10, 0.12, 0.14]
    results = {
        "ridge-v1": _fake_run(0.11, fold_metrics={"rmsle": fold_values}),
        "hgb-v1": _fake_run(0.30, family="hgb"),
    }
    outcome = decide(ir, results)

    assert outcome["selected_option"]["candidate_id"] == "ridge-v1"
    assert outcome["uncertainty"]["metric_std"] == pytest.approx(
        statistics.stdev(fold_values)
    )
    assert "metric_std" in outcome["uncertainty"]["note"]


# ---------- hard-gate short-circuit ----------

def test_gate_fail_marks_all_candidates_constraint_gate_failed():
    ir = load_ir(EXAMPLES_DIR / "invalid-g6-authorization.json")
    outcome = decide(ir, {})  # gate-fail path runs without any results

    assert set(outcome) == DECISION_CONTRACT_FIELDS
    assert outcome["selected_option"] is None

    statuses = {c["gate_id"]: c["status"] for c in outcome["constraint_check"]}
    assert len(statuses) == 6
    assert statuses["G6_RUN_AUTHORIZATION"] == "fail"

    rejected = outcome["rejected_options_and_reasons"]
    assert [entry["candidate_id"] for entry in rejected] == [
        candidate["candidate_id"] for candidate in ir["candidates"]
    ]
    assert all(
        entry["reason"] == REASON_CONSTRAINT_GATE_FAILED for entry in rejected
    )
    assert all("G6_RUN_AUTHORIZATION" in entry["detail"] for entry in rejected)

    assert outcome["alternatives"] == []
    assert outcome["evidence_refs"] == []
    # No executed run exists: state falls back to the IR's own declarations.
    assert outcome["state_snapshot_ref"] == {
        "ir_content_hash": ir["content_hash"],
        "data_sha256": ir["dataset_snapshot_ref"]["sha256"],
        "code_version": "unknown",
    }
    assert outcome["uncertainty"] == {
        "metric_std": None,
        "margin": None,
        "note": outcome["uncertainty"]["note"],
    }
    assert outcome["uncertainty"]["note"]


# ---------- input validation ----------

def test_unknown_result_key_raises(replay):
    ir, results, run_ridge, _ = replay
    with pytest.raises(ValueError, match="unknown candidate_id"):
        decide(ir, {**results, "ghost-candidate": run_ridge})


def test_missing_candidate_result_raises(replay):
    ir, results, run_ridge, _ = replay
    with pytest.raises(ValueError, match="missing"):
        decide(ir, {"ridge-v1": run_ridge})


def test_family_mismatch_raises(replay):
    ir, results, *_ = replay
    tampered = _fake_run(0.10, family="hgb")  # IR says ridge for ridge-v1
    with pytest.raises(ValueError, match="family"):
        decide(ir, {"ridge-v1": tampered, "hgb-v1": results["hgb-v1"]})


def test_ir_without_candidates_raises():
    ir = load_ir(CASE_DIR / "experiment.ir.json")
    ir["candidates"] = []
    ir["content_hash"] = compute_content_hash(ir)
    with pytest.raises(ValueError, match="no candidates"):
        decide(ir, {})
