"""ADAPTER-003 tests: warehouse estimate exchange (SPEC-006).

Covers: the reference stub decision engine's determinism (accept/reject +
capacity-violation count, byte-identical across calls) and
``import_domain_outcome``: business KPIs ride in the dedicated
``domain_kpis`` field and NEVER pollute the ML ``metric_name``/``metric_value``
channel (structural assertion), evidence_type mapping, and the provenance
chain pointing back at the envelope_id.
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.src.adapters import warehouse as wh
from framework.src.adapters.estimates import EnvelopeError, verify_envelope_identity
from framework.src.adapters.pjp import (
    AdapterError,
    import_domain_outcome,
    EVIDENCE_TYPE_FAILURE,
    EVIDENCE_TYPE_SUCCESS,
)

FIXED_CLOCK = lambda: datetime(2026, 10, 9, 8, 0, 0, tzinfo=timezone.utc)  # noqa: E731
IR = {"content_hash": "3f9a1c7e5b2d4a8f0c6e1d3b5a7f9e2c4d6b8a0f2e4c6d8b0a2f4e6c8d1b3a5f"}
EVIDENCE = {"evidence_id": "ev-9c4f2a8e6d1b"}


@pytest.fixture()
def warehouse_envelope(tmp_path):
    rows = []
    for code, base in (
        ("SKU-EAST-01", 120.0),
        ("SKU-EAST-02", 85.5),
        ("SKU-WEST-01", 40.25),
    ):
        for week in ("2026-09-07", "2026-09-14", "2026-09-21"):
            rows.append({"id": code, "target": base, "oof_pred": base * 1.01})
    path = tmp_path / "warehouse_oof.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return wh.warehouse_envelope_from_oof(path, "id", IR, EVIDENCE, clock=FIXED_CLOCK)


# ---------- 1. Same envelope contract, warehouse_demand kind ----------

def test_warehouse_envelope_kind_and_identity(warehouse_envelope):
    env = warehouse_envelope
    assert env["estimate_kind"] == "warehouse_demand"
    assert env["quantization_rule"] is None  # p80 weekday 规则是 PJP 概念
    assert all("quantized" not in u for u in env["units"])
    assert env["value_unit"] == "units_per_day"
    assert all(not u["unit_id"].isdigit() for u in env["units"])  # D7
    verify_envelope_identity(env)


# ---------- 2. Reference stub engine: deterministic ----------

def test_stub_engine_deterministic_and_self_labelling(warehouse_envelope):
    """同输入 -> 逐字节相同; 输出自标 reference stub（非真实仓储决策）。"""
    first = wh.stub_decision_engine(warehouse_envelope, capacity=250)
    second = wh.stub_decision_engine(warehouse_envelope, capacity=250)
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["engine"] == wh.STUB_ENGINE_ID
    assert first["stub"] is True
    assert first["deterministic"] is True
    assert first["envelope_id"] == warehouse_envelope["envelope_id"]
    assert first["capacity_unit"] == "units_per_day"


def test_stub_engine_accepts_when_capacity_covers_total(warehouse_envelope):
    # 总需求 = 1.01 * (120 + 85.5 + 40.25) = 248.2075 -> 249 容纳全部
    result = wh.stub_decision_engine(warehouse_envelope, capacity=249)
    assert result["decision"] == "accept"
    assert result["capacity_violations"] == 0
    assert result["rejected_units"] == []
    assert result["accepted_units"] == ["SKU-EAST-01", "SKU-EAST-02", "SKU-WEST-01"]


def test_stub_engine_rejects_and_counts_violations_in_stable_order(warehouse_envelope):
    # 容量只够前两家 (按稳定编码序贪心): 120*1.01 + 85.5*1.01 = 207.555 -> SKU-WEST-01 放不下
    result = wh.stub_decision_engine(warehouse_envelope, capacity=210)
    assert result["decision"] == "reject"
    assert result["capacity_violations"] == 1
    assert result["rejected_units"] == ["SKU-WEST-01"]
    assert result["accepted_units"] == ["SKU-EAST-01", "SKU-EAST-02"]
    # 零容量: 全拒
    zero = wh.stub_decision_engine(warehouse_envelope, capacity=0)
    assert zero["decision"] == "reject"
    assert zero["capacity_violations"] == 3


def test_stub_engine_rejects_wrong_kind_and_bad_capacity(warehouse_envelope):
    with pytest.raises(EnvelopeError, match="warehouse_demand"):
        wh.stub_decision_engine({"estimate_kind": "pjp_service_weekday"}, capacity=1)
    with pytest.raises(ValueError, match="capacity"):
        wh.stub_decision_engine(warehouse_envelope, capacity=-1)
    with pytest.raises(ValueError, match="capacity"):
        wh.stub_decision_engine(warehouse_envelope, capacity=1.5)


# ---------- 3. import_domain_outcome: KPI 不污染 ML 指标通道 ----------

def test_import_domain_outcome_kpis_stay_out_of_metric_fields(warehouse_envelope):
    """结构断言: 业务 KPI 只在 domain_kpis; metric_name/metric_value 钉死 null。"""
    kpis = {"total_km": 8.0, "on_time_rate": 0.97, "utilization": "88%"}
    evidence = import_domain_outcome(
        {
            "envelope_id": warehouse_envelope["envelope_id"],
            "outcome": "success",
            "domain_kpis": kpis,
            "solver_status": "OPTIMAL",
        },
        source="warehouse-engine",
    )
    assert evidence["domain_kpis"] == kpis  # 原样转录, 领域上报事实
    assert evidence["metric_name"] is None
    assert evidence["metric_value"] is None
    assert evidence["metric_version"] is None
    # KPI 键值不得以任何形式出现在 ML 指标字段
    assert "total_km" not in json.dumps(
        {k: evidence[k] for k in ("metric_name", "metric_value", "metric_version")}
    )


def test_import_domain_outcome_evidence_type_mapping(warehouse_envelope):
    base = {"envelope_id": warehouse_envelope["envelope_id"]}
    assert import_domain_outcome({**base, "outcome": "success"}, source="visitmodel")[
        "evidence_type"
    ] == EVIDENCE_TYPE_SUCCESS
    assert import_domain_outcome({**base, "outcome": "task_success"}, source="visitmodel")[
        "evidence_type"
    ] == EVIDENCE_TYPE_SUCCESS
    assert import_domain_outcome({**base, "outcome": "failure"}, source="visitmodel")[
        "evidence_type"
    ] == EVIDENCE_TYPE_FAILURE
    assert import_domain_outcome({**base, "outcome": "task_failure"}, source="visitmodel")[
        "evidence_type"
    ] == EVIDENCE_TYPE_FAILURE
    with pytest.raises(AdapterError, match="outcome"):
        import_domain_outcome({**base, "outcome": "maybe"}, source="visitmodel")


def test_import_domain_outcome_provenance_chains_back_to_envelope(warehouse_envelope):
    evidence = import_domain_outcome(
        {"envelope_id": warehouse_envelope["envelope_id"], "outcome": "success"},
        source="visitmodel",
    )
    # 显式回指 + sha256 派生链 id 双通道
    assert evidence["provenance_refs"] == {"envelope_id": warehouse_envelope["envelope_id"]}
    assert evidence["provenance_chain_id"].startswith("pch-")
    assert len(evidence["provenance_chain_id"]) == len("pch-") + 12
    assert evidence["task_id"] == warehouse_envelope["envelope_id"]  # 缺省 task 回指信封


def test_import_domain_outcome_deterministic_and_strict(warehouse_envelope):
    payload = {
        "envelope_id": warehouse_envelope["envelope_id"],
        "outcome": "failure",
        "domain_kpis": {"late_rate": 0.02},
        "timestamp": "2026-10-09T08:00:00+00:00",
    }
    first = import_domain_outcome(payload, source="warehouse-engine")
    second = import_domain_outcome(dict(payload), source="warehouse-engine")
    assert first == second  # 无 uuid4/墙钟: 纯内容派生
    assert first["evidence_id"].startswith("ev-")
    assert re.fullmatch(r"ev-[0-9a-f]{64}", first["evidence_id"])
    assert first["timestamp"] == payload["timestamp"]  # 领域上报时刻原样保留

    with pytest.raises(AdapterError, match="未知字段"):
        import_domain_outcome({**payload, "metric_value": 1.23}, source="warehouse-engine")
    with pytest.raises(AdapterError, match="envelope_id"):
        import_domain_outcome({"outcome": "success"}, source="visitmodel")
    with pytest.raises(AdapterError, match="source"):
        import_domain_outcome(
            {"envelope_id": warehouse_envelope["envelope_id"], "outcome": "success"},
            source="",
        )
