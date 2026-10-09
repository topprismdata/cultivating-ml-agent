"""Skill Tester gate + ledger tests (SPEC-007 §4/§5).

Covers: fixed gate rules (negative >= 1 per G4, no failed cases, non-empty
skill_id), closed reason vocabulary, suggestion-only crystallization (never
writes skills/), deterministic decision identity, report digest sensitivity,
ledger roundtrip and tamper interception for gate decisions, and policy hash
pinning on gate decisions.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.src.governance import (
    SKILL_GATE_REASONS,
    evaluate_skill_gate,
)
from framework.src.governance.skill_gate import (
    GATE_STATUS_FAIL,
    GATE_STATUS_PASS,
    GateDecision,
    compute_decision_id,
)
from framework.src.governance import (
    GovernanceError,
    GovernanceLedgerTamperError,
    append_decision,
    load_decision,
    load_policy,
)

DEFAULT_POLICY_PATH = (
    ROOT / "framework" / "src" / "governance" / "policies" / "default-policy.json"
)
SKILL_ID = "skill-demo-forecast"


@pytest.fixture(scope="module")
def policy():
    return load_policy(DEFAULT_POLICY_PATH)


def make_report(**overrides):
    report = {
        "run_id": "st-run-0001",
        "skill_id": SKILL_ID,
        "tests": [
            {"case_id": "case-pos-1", "case_type": "positive", "status": "pass"},
            {"case_id": "case-neg-1", "case_type": "negative", "status": "pass"},
        ],
    }
    report.update(overrides)
    return report


# ---------------------------------------------------------------------------
# Gate rules
# ---------------------------------------------------------------------------

class TestGateRules:
    def test_pass_with_positive_and_negative(self, policy):
        decision = evaluate_skill_gate(SKILL_ID, make_report(), policy)
        assert decision.status == GATE_STATUS_PASS
        assert decision.reasons == []
        assert decision.counts == {"tests": 2, "positive": 1, "negative": 1, "failed": 0}
        assert decision.gate == "skill_tester"
        assert decision.skill_id == SKILL_ID

    def test_missing_negative_case_fails_g4(self, policy):
        report = make_report(
            tests=[{"case_id": "case-pos-1", "case_type": "positive", "status": "pass"}]
        )
        decision = evaluate_skill_gate(SKILL_ID, report, policy)
        assert decision.status == GATE_STATUS_FAIL
        assert [r["reason"] for r in decision.reasons] == ["missing_negative_case"]
        assert decision.crystallization["suggested"] is False  # G4: 负例缺失不给结晶建议

    def test_failed_case_fails_gate(self, policy):
        report = make_report(
            tests=[
                {"case_id": "case-pos-1", "case_type": "positive", "status": "pass"},
                {"case_id": "case-neg-1", "case_type": "negative", "status": "fail"},
            ]
        )
        decision = evaluate_skill_gate(SKILL_ID, report, policy)
        assert decision.status == GATE_STATUS_FAIL
        assert [r["reason"] for r in decision.reasons] == ["failed_case_present"]
        assert "case-neg-1" in decision.reasons[0]["detail"]
        assert decision.counts["failed"] == 1

    def test_empty_skill_id_fails_gate(self, policy):
        decision = evaluate_skill_gate("", make_report(), policy)
        assert decision.status == GATE_STATUS_FAIL
        assert [r["reason"] for r in decision.reasons] == ["empty_skill_id"]
        assert decision.crystallization["suggested"] is False
        blank = evaluate_skill_gate("   ", make_report(), policy)
        assert blank.status == GATE_STATUS_FAIL

    def test_all_failures_reported_at_once(self, policy):
        report = make_report(
            tests=[
                {"case_id": "case-pos-1", "case_type": "positive", "status": "fail"},
            ]
        )
        decision = evaluate_skill_gate("", report, policy)
        assert [r["reason"] for r in decision.reasons] == [
            "empty_skill_id",
            "missing_negative_case",
            "failed_case_present",
        ]

    def test_reason_vocabulary_closed(self, policy):
        for decision in (
            evaluate_skill_gate(SKILL_ID, make_report(), policy),
            evaluate_skill_gate("", make_report(), policy),
        ):
            for entry in decision.reasons:
                assert entry["reason"] in SKILL_GATE_REASONS

    def test_malformed_report_is_contract_error_not_gate_fail(self, policy):
        with pytest.raises(GovernanceError, match="tests"):
            evaluate_skill_gate(SKILL_ID, {"tests": []}, policy)
        with pytest.raises(GovernanceError, match="case_type"):
            evaluate_skill_gate(
                SKILL_ID,
                make_report(
                    tests=[{"case_id": "c1", "case_type": "chaos", "status": "pass"}]
                ),
                policy,
            )
        with pytest.raises(GovernanceError, match="status"):
            evaluate_skill_gate(
                SKILL_ID,
                make_report(
                    tests=[{"case_id": "c1", "case_type": "negative", "status": "skipped"}]
                ),
                policy,
            )


# ---------------------------------------------------------------------------
# Crystallization is suggestion-only (never writes skills/)
# ---------------------------------------------------------------------------

class TestCrystallizationSuggestion:
    def test_suggestion_on_pass(self, policy):
        decision = evaluate_skill_gate(SKILL_ID, make_report(), policy)
        sug = decision.crystallization
        assert sug["suggested"] is True
        assert sug["skill_id"] == SKILL_ID
        assert "不自动写" in sug["note"]

    def test_no_side_effects_on_skills_dir(self, policy, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # skills/ 若被写入必然出现在 cwd
        evaluate_skill_gate(SKILL_ID, make_report(), policy)
        assert not (tmp_path / "skills").exists()
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Deterministic identity + policy pinning
# ---------------------------------------------------------------------------

class TestGateIdentity:
    def test_decision_id_deterministic_no_wall_clock(self, policy):
        a = evaluate_skill_gate(SKILL_ID, make_report(), policy)
        b = evaluate_skill_gate(SKILL_ID, make_report(), policy)
        assert a.decision_id == b.decision_id
        assert a.decision_id.startswith("gvg-")
        assert a.to_dict() == b.to_dict()

    def test_report_digest_sensitivity(self, policy):
        a = evaluate_skill_gate(SKILL_ID, make_report(), policy)
        changed = make_report(run_id="st-run-0002")
        b = evaluate_skill_gate(SKILL_ID, changed, policy)
        assert a.report_digest != b.report_digest
        assert a.decision_id != b.decision_id
        import hashlib

        expected = hashlib.sha256(
            json.dumps(make_report(), sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        assert a.report_digest == expected

    def test_policy_pins_into_decision(self, policy):
        decision = evaluate_skill_gate(SKILL_ID, make_report(), policy)
        assert decision.policy_id == policy["policy_id"]
        assert decision.policy_content_hash == policy["policy_content_hash"]

    def test_roundtrip_dict(self, policy):
        decision = evaluate_skill_gate(SKILL_ID, make_report(), policy)
        assert GateDecision.from_dict(decision.to_dict()) == decision
        assert compute_decision_id(decision.to_dict()) == decision.decision_id


# ---------------------------------------------------------------------------
# Ledger roundtrip + tamper (gate decisions)
# ---------------------------------------------------------------------------

class TestGateLedger:
    def test_roundtrip_and_tamper_interception(self, policy, tmp_path):
        decision = evaluate_skill_gate(SKILL_ID, make_report(), policy)
        path = append_decision(decision.to_dict(), tmp_path)
        assert path.name == f"{decision.decision_id}.jsonl"
        doc = load_decision(path)
        assert doc["decision_id"] == decision.decision_id
        assert doc["kind"] == "skill_gate"
        assert doc["decision"]["status"] == "pass"

        lines = path.read_bytes().splitlines()
        event = json.loads(lines[0])
        event["decision"]["status"] = "pass"  # 内容篡改（同值改写也必须破坏 event_id 才合法）
        event["decision"]["counts"]["tests"] = 99
        path.write_bytes(
            json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        )
        with pytest.raises(GovernanceLedgerTamperError):
            load_decision(path)
