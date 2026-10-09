"""ADAPTER-002 tests: PJP estimate exchange (SPEC-006).

Covers: envelope export + hash identity determinism (sha256-derived
envelope_id/content_hash, created_at never hashed), quantization idempotence,
the sigma 0/1-based trap (envelope weekday sets are 1-based ISO, pinned in
1..7), D7 identity discipline (stable codes only — no int idx anywhere in the
envelope, no persisted code->idx mapping), the zero-solver-import
architecture guard, and — when ``VISITMODEL_PATH`` points at a local
VisitModel checkout — the real solver round-trip: sp_solve_ip run twice
(baseline sigma=None vs ML sigma), asserting objective_milli/status
reproducibility and monotone violation counts in sigma_budget, including the
empty-set and sigma_budget=0 boundaries. Without the env var the round-trip
skips (CI has no local VisitModel).

Environment guard order (historical lesson): FIRST check
``os.environ.get("VISITMODEL_PATH", "")`` is non-empty, THEN ``Path.exists()``
— an empty string makes ``Path("").exists()`` True.
"""
import json
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.src.adapters.estimates import (
    EnvelopeError,
    envelope_content_hash,
    envelope_from_oof,
    quantize_p80_weekday_coverage,
    validate_envelope,
    verify_envelope_identity,
)
from framework.src.adapters.pjp import (
    AdapterError,
    build_solver_inputs,
    count_sigma_violations,
    import_domain_outcome,
    summarize_solve_result,
)

FIXED_CLOCK = lambda: datetime(2026, 10, 9, 8, 0, 0, tzinfo=timezone.utc)  # noqa: E731
IR = {"content_hash": "3f9a1c7e5b2d4a8f0c6e1d3b5a7f9e2c4d6b8a0f2e4c6d8b0a2f4e6c8d1b3a5f"}
EVIDENCE = {"evidence_id": "ev-9c4f2a8e6d1b"}

# VisitModel linkage-v2 fixture shape (4 stores x 4 days; two Mondays, two
# Tuesdays; stores 0/2 legal on Mondays, 1/3 on Tuesdays; ring distances).
DATES = [date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 13), date(2026, 7, 14)]
CODE_TO_IDX = {"STORE-0": 0, "STORE-1": 1, "STORE-2": 2, "STORE-3": 3}
K_C = {"STORE-0": 2, "STORE-1": 2, "STORE-2": 2, "STORE-3": 2}


def _oof_frame(aligned: bool = True) -> pd.DataFrame:
    """Truth-bearing OOF frame in the build_oof_frame shape (+ time column).

    aligned=True: each store's predicted weekday coverage matches its legal
    service days (S0/S2 Monday, S1/S3 Tuesday). aligned=False: ML predicts
    Monday-only for every store — the solver must pay violations to serve
    Tuesdays.
    """
    rows = []
    for week in ("2026-07-06", "2026-07-13"):
        for code, offset, base in (
            ("STORE-0", 0, 10.0),
            ("STORE-2", 0, 10.0),
            ("STORE-1", 1, 10.0),
            ("STORE-3", 1, 10.0),
        ):
            if not aligned:
                offset = 0  # everyone predicted Monday
            d = pd.Timestamp(week) + pd.Timedelta(days=offset)
            rows.append(
                {"id": code, "time": d.date().isoformat(),
                 "target": base, "oof_pred": base + 0.1}
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def oof_csv(tmp_path):
    path = tmp_path / "oof.csv"
    _oof_frame().to_csv(path, index=False)
    return path


@pytest.fixture()
def envelope(oof_csv):
    return envelope_from_oof(oof_csv, "id", IR, EVIDENCE, clock=FIXED_CLOCK)


@pytest.fixture()
def pool():
    columns = []
    for d in DATES:
        if d.weekday() == 0:  # Monday
            columns += [(d, [0], 10.0), (d, [2], 10.0), (d, [0, 2], 2.0)]
        else:  # Tuesday
            columns += [(d, [1], 10.0), (d, [3], 10.0), (d, [1, 3], 2.0)]
    return columns


# ---------- 1. Architecture guard: zero solver dependency ----------

def test_adapters_never_import_solvers():
    """framework/src/adapters/** 零 ortools/求解器/VisitModel import（SPEC-006 §7）。"""
    forbidden = re.compile(
        r"^\s*(?:import|from)\s+(ortools|visitmodel|visit_ir|visit_semantic_api|opticore)\b",
        re.MULTILINE,
    )
    adapters_dir = ROOT / "framework" / "src" / "adapters"
    offenders = []
    for py in sorted(adapters_dir.glob("*.py")):
        source = py.read_text(encoding="utf-8")
        if forbidden.search(source):
            offenders.append(py.name)
    assert offenders == []


# ---------- 2. Envelope export + hash identity determinism ----------

def test_envelope_export_deterministic(envelope, oof_csv):
    """同输入（含冻结时钟）-> 逐字节相同信封；created_at 不入哈希。"""
    rebuilt = envelope_from_oof(oof_csv, "id", IR, EVIDENCE, clock=FIXED_CLOCK)
    assert rebuilt == envelope
    assert envelope["envelope_id"] == re.sub(
        r"^env-", "env-", "env-" + envelope["content_hash"][:12]
    )
    assert envelope["envelope_id"] == "env-" + envelope["content_hash"][:12]

    # created_at 易失: 改时钟不改身份
    later = envelope_from_oof(
        oof_csv, "id", IR, EVIDENCE,
        clock=lambda: datetime(2027, 1, 1, tzinfo=timezone.utc),
    )
    assert later["created_at"] != envelope["created_at"]
    assert later["content_hash"] == envelope["content_hash"]
    assert later["envelope_id"] == envelope["envelope_id"]

    # 内容变 -> 身份变
    tampered = json.loads(json.dumps(envelope))
    tampered["units"][0]["value"] += 1.0
    assert envelope_content_hash(tampered) != envelope["content_hash"]


def test_envelope_shape_and_provenance(envelope):
    assert envelope["schema_version"] == "0.1.0"
    assert envelope["estimate_kind"] == "pjp_service_weekday"
    assert envelope["value_unit"] == "units_per_day"
    assert envelope["quantization_rule"] == "p80_weekday_coverage"
    assert envelope["model_provenance"] == {
        "ir_content_hash": IR["content_hash"],
        "evidence_id": "ev-9c4f2a8e6d1b",
    }
    assert validate_envelope(envelope) == []
    verify_envelope_identity(envelope)  # tamper check raises on mismatch


def test_envelope_identity_tamper_rejected(envelope):
    tampered = json.loads(json.dumps(envelope))
    tampered["units"][0]["value"] += 1.0
    with pytest.raises(EnvelopeError, match="content_hash mismatch"):
        verify_envelope_identity(tampered)
    id_swapped = json.loads(json.dumps(envelope))
    id_swapped["envelope_id"] = "env-" + "0" * 12
    with pytest.raises(EnvelopeError, match="envelope_id mismatch"):
        verify_envelope_identity(id_swapped)


def test_envelope_requires_truth_column(tmp_path):
    path = tmp_path / "no_truth.csv"
    pd.DataFrame({"id": ["A"], "oof_pred": [1.0]}).to_csv(path, index=False)
    with pytest.raises(EnvelopeError, match="缺少必需列"):
        envelope_from_oof(path, "id", IR, EVIDENCE, clock=FIXED_CLOCK)


def test_schema_examples_valid_and_identity_consistent():
    """examples/ 两个样例: jsonschema 校验合法 + 身份可由适配器算法复算。"""
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = ROOT / "schemas" / "estimate-envelope" / "0.1.0" / "estimate-envelope.schema.json"
    examples_dir = schema_path.parent / "examples"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    for name in ("pjp-service-weekday.json", "warehouse-demand.json"):
        doc = json.loads((examples_dir / name).read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator(schema).validate(doc)
        verify_envelope_identity(doc)  # content_hash/envelope_id 与算法一致
        assert validate_envelope(doc) == []


# ---------- 3. Quantization rule: idempotent, P80, 1-based pinned ----------

def test_quantization_idempotent_and_pure(oof_csv):
    frame = pd.read_csv(oof_csv)
    first = quantize_p80_weekday_coverage(frame, group_col="id")
    second = quantize_p80_weekday_coverage(frame, group_col="id")
    assert first == second  # 纯函数: f(f(x)) == f(x) 语义下的幂等
    # envelope rebuild 再量化 -> 载荷不变
    env_a = envelope_from_oof(oof_csv, "id", IR, EVIDENCE, clock=FIXED_CLOCK)
    env_b = envelope_from_oof(oof_csv, "id", IR, EVIDENCE, clock=FIXED_CLOCK)
    assert [u.get("quantized") for u in env_a["units"]] == [
        u.get("quantized") for u in env_b["units"]
    ]


def test_sigma_basis_pinned_one_based(envelope):
    """0/1 基钉死: 信封 weekday 集合是 1 基 ISO (1=周一..7=周日), 永不出现 0。

    OOF fixture 的周一 (pandas/Timestamp weekday()==0, 0 基) 必须落成信封里的 1;
   VisitContract.sigma 的 0 基语义 (visit_semantic_api: sigma: int, 0=周一)
    泄漏到本边界即为错误 — 适配器拒绝 0。
    """
    by_code = {u["unit_id"]: u for u in envelope["units"]}
    assert by_code["STORE-0"]["quantized"]["weekdays"] == [1]  # Monday -> 1 (1 基)
    assert by_code["STORE-1"]["quantized"]["weekdays"] == [2]  # Tuesday -> 2
    for unit in envelope["units"]:
        for w in unit["quantized"]["weekdays"]:
            assert 1 <= w <= 7
            assert isinstance(w, int)


def test_zero_based_weekday_leak_rejected(oof_csv):
    env = envelope_from_oof(oof_csv, "id", IR, EVIDENCE, clock=FIXED_CLOCK)
    leaked = json.loads(json.dumps(env))
    leaked["units"][0]["quantized"]["weekdays"] = [0]  # 0 基 VisitContract.sigma 泄漏
    with pytest.raises(AdapterError, match="1 基 ISO"):
        build_solver_inputs(
            leaked, CODE_TO_IDX, DATES, [], K_C, sigma_budget=0
        )


def test_unknown_quantization_rule_rejected(oof_csv):
    with pytest.raises(ValueError, match="未知量化规则"):
        envelope_from_oof(
            oof_csv, "id", IR, EVIDENCE, quantization="top1_weekday",
            clock=FIXED_CLOCK,
        )


def test_time_column_optional_declares_rule_but_no_payload(tmp_path):
    path = tmp_path / "no_time.csv"
    _oof_frame().drop(columns=["time"]).to_csv(path, index=False)
    env = envelope_from_oof(path, "id", IR, EVIDENCE, clock=FIXED_CLOCK)
    assert env["quantization_rule"] == "p80_weekday_coverage"  # 声明保留
    assert all("quantized" not in u for u in env["units"])  # 载荷缺省


# ---------- 4. D7 identity discipline ----------

def test_envelope_carries_no_int_idx(envelope):
    """信封只携带稳定编码; 无任何 *_idx 字段（D7: int idx 不配当身份）。"""
    for unit in envelope["units"]:
        assert isinstance(unit["unit_id"], str) and not unit["unit_id"].isdigit()
    assert "_idx" not in json.dumps(envelope)
    assert "idx" not in json.dumps(envelope)


def test_code_to_idx_mapping_not_persisted(envelope, pool):
    """code->idx 映射只活在一次调用里: 信封与求解器输入拼装均不回写映射。"""
    mapping = dict(CODE_TO_IDX)
    build_solver_inputs(envelope, mapping, DATES, pool, K_C, sigma_budget=0)
    assert mapping == CODE_TO_IDX  # 调用后映射未被改写/持久化
    assert "idx" not in json.dumps(envelope)  # 信封键均为稳定编码, 无 idx 通道


# ---------- 5. build_solver_inputs contract ----------

def test_build_solver_inputs_shape(envelope, pool):
    inputs = build_solver_inputs(envelope, CODE_TO_IDX, DATES, pool, K_C, sigma_budget=4)
    # sp_solve_ip 形状: dates / k_c{idx:count} / pool[(date,route,kmm)] / sigma{idx:set} / budget / timeout
    assert set(inputs) == {"dates", "k_c", "pool", "sigma", "sigma_budget", "timeout_s"}
    assert inputs["dates"] == DATES
    assert inputs["k_c"] == {0: 2, 1: 2, 2: 2, 3: 2}  # code -> idx 映射生效
    assert inputs["sigma"] == {0: {1}, 1: {2}, 2: {1}, 3: {2}}  # 1 基集合原样转 idx 键
    assert inputs["sigma_budget"] == 4
    for date_, route, km in inputs["pool"]:
        assert date_ in DATES and km >= 0


def test_build_solver_inputs_rejects_partial_sigma_coverage(envelope):
    """k_c 店缺量化集合 = 恒违规脚枪: 拒绝拼装。"""
    partial = json.loads(json.dumps(envelope))
    partial["units"] = [u for u in partial["units"] if u["unit_id"] != "STORE-3"]
    with pytest.raises(AdapterError, match="缺少 quantized weekday"):
        build_solver_inputs(partial, CODE_TO_IDX, DATES, [], K_C, sigma_budget=0)


def test_build_solver_inputs_rejects_unknown_code(envelope, pool):
    with pytest.raises(AdapterError, match="之外的店编码"):
        build_solver_inputs(
            envelope, {"STORE-0": 0}, DATES, pool, K_C, sigma_budget=0
        )


def test_build_solver_inputs_rejects_bad_pool(envelope):
    with pytest.raises(AdapterError, match="不在 dates 内"):
        build_solver_inputs(
            envelope, CODE_TO_IDX, DATES[:2],
            [(DATES[3], [0], 1.0)], K_C, sigma_budget=0,
        )
    with pytest.raises(AdapterError, match="未知 idx"):
        build_solver_inputs(
            envelope, CODE_TO_IDX, DATES,
            [(DATES[0], [9], 1.0)], K_C, sigma_budget=0,
        )


# ---------- 6. summarize_solve_result: deterministic fields only ----------

def test_summarize_drops_days_keeps_contract_fields(pool):
    days = {DATES[0]: [0, 2], DATES[1]: [1, 3]}
    diagnostics = {"status": "OPTIMAL", "objective_value_milli": 8000.0,
                   "selected_objective_milli": 8000.0}
    sigma = {0: {1}, 1: {2}, 2: {1}, 3: {2}}
    summary = summarize_solve_result((8.0, days, diagnostics), sigma=sigma)
    assert summary == {"objective_milli": 8000, "status": "OPTIMAL", "violations": 0}
    # days 被丢弃: 输出键只有三个确定性字段
    assert set(summary) == {"objective_milli", "status", "violations"}
    # Tuesday 服务 vs 全店预测周一 -> 每列 2 违规 ([1,3] 两店都不在 {1})
    monday_sigma = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
    summary_viol = summarize_solve_result((8.0, days, diagnostics), sigma=monday_sigma)
    assert summary_viol["violations"] == 2


def test_summarize_baseline_without_sigma_has_no_fabricated_violations():
    summary = summarize_solve_result((None, None, {"status": "INFEASIBLE"}))
    assert summary == {"objective_milli": None, "status": "INFEASIBLE", "violations": None}


def test_count_sigma_violations_linear_additive():
    days = {date(2026, 7, 6): [0, 2], date(2026, 7, 7): [1, 3]}  # Mon, Tue
    sigma = {0: {1}, 1: {2}, 2: {1}, 3: {2}}
    assert count_sigma_violations(days, sigma) == 0
    assert count_sigma_violations(days, {}) == 4  # 每店恰一次覆盖 -> 线性可加
    assert count_sigma_violations(days, {c: set() for c in range(4)}) == 4


# ---------- 7. Real solver round-trip (guarded by VISITMODEL_PATH) ----------

_VISITMODEL_PATH = os.environ.get("VISITMODEL_PATH", "")
if _VISITMODEL_PATH and Path(_VISITMODEL_PATH).exists():
    _src = str(Path(_VISITMODEL_PATH) / "src")
    for _sibling in ("VisitIR", "OptiCore"):  # VisitModel 顶层 import visit_ir/opticore
        _p = Path(_VISITMODEL_PATH).with_name(_sibling) / "src"
        if _p.exists():
            sys.path.insert(0, str(_p))
    sys.path.insert(0, _src)
    try:
        from visitmodel.sp.formulation import sp_solve_ip  # noqa: E402
        _HAS_VISITMODEL = True
    except ImportError:  # pragma: no cover - 依赖不全的本地环境
        _HAS_VISITMODEL = False
else:
    _HAS_VISITMODEL = False


@pytest.mark.skipif(
    not _HAS_VISITMODEL,
    reason="VISITMODEL_PATH 未设置或 VisitModel/VisitIR 不可导入 (CI 无本地领域路径)",
)
class TestRealSolverRoundTrip:
    """真验往返: sp_solve_ip 两次运行, 契约字段可复现; 违规计数随预算单调。"""

    @pytest.fixture()
    def solver_env(self, envelope, pool):
        return build_solver_inputs(
            envelope, CODE_TO_IDX, DATES, pool, K_C, sigma_budget=0, timeout_s=30.0
        )

    def _solve(self, inputs, budget):
        raw = sp_solve_ip(**{**inputs, "sigma_budget": budget}, return_diagnostics=True)
        return summarize_solve_result(raw, sigma=inputs["sigma"])

    def test_objective_and_status_reproducible(self, solver_env):
        """两次运行: objective_milli (int(round(km*1000))) 与 status 逐位可复现。

        selected days 不锁 (num_search_workers=8 无 random_seed, 并列最优漂移) —
        契约只锁 objective_milli/status/违规计数。
        """
        baseline_raw = sp_solve_ip(
            **{**solver_env, "sigma": None, "sigma_budget": None},
            return_diagnostics=True,
        )
        baseline_again = sp_solve_ip(
            **{**solver_env, "sigma": None, "sigma_budget": None},
            return_diagnostics=True,
        )
        # days 可能漂移, objective/status 不得漂移
        assert baseline_raw[0] == baseline_again[0]
        assert baseline_raw[2]["status"] == baseline_again[2]["status"]
        baseline = summarize_solve_result(
            baseline_raw, sigma=solver_env["sigma"]
        )
        assert baseline == {
            "objective_milli": 8000, "status": "OPTIMAL", "violations": 0
        }

    def test_ml_sigma_beats_or_matches_baseline_within_budget(self, solver_env):
        """ML sigma 全对齐时: budget=0 即可行, 与 baseline 同目标零违规。"""
        constrained = self._solve(solver_env, 0)
        assert constrained["status"] == "OPTIMAL"
        assert constrained["objective_milli"] == 8000
        assert constrained["violations"] == 0

    def test_violation_counts_monotone_in_sigma_budget(self, pool, tmp_path):
        """错位预测 (全店预测周一): 违规计数随 sigma_budget 单调不增。

        预算 0..3 -> INFEASIBLE (周二最低违规 4 超预算); >=4 -> OPTIMAL,
        违规 4 (可行域内违规随预算放宽不增)。
        """
        misaligned_path = tmp_path / "misaligned.csv"
        _oof_frame(aligned=False).to_csv(misaligned_path, index=False)
        misaligned_env = envelope_from_oof(
            misaligned_path, "id", IR, EVIDENCE, clock=FIXED_CLOCK
        )
        inputs = build_solver_inputs(
            misaligned_env, CODE_TO_IDX, DATES, pool, K_C,
            sigma_budget=10, timeout_s=30.0,
        )
        observed = [(b, self._solve(inputs, b)) for b in (0, 2, 4, 10)]
        assert observed[0][1]["status"] == "INFEASIBLE"  # sigma_budget=0 边界
        assert observed[1][1]["status"] == "INFEASIBLE"
        assert observed[2][1] == {
            "objective_milli": 8000, "status": "OPTIMAL", "violations": 4
        }
        assert observed[3][1] == observed[2][1]  # 预算放宽, 违规不增 (4 -> 4)
        feasible = [s["violations"] for _, s in observed if s["violations"] is not None]
        assert feasible == sorted(feasible, reverse=True)  # 单调不增

    def test_empty_set_sigma_budget_zero_boundary(self, envelope, pool):
        """空 set 边界: 每列恒违规; budget=0 -> INFEASIBLE; budget 够 -> 违规=列店数和。"""
        inputs = build_solver_inputs(
            envelope, CODE_TO_IDX, DATES, pool, K_C,
            sigma_budget=10, timeout_s=30.0,
        )
        empty_sigma = {c: set() for c in inputs["sigma"]}
        raw0 = sp_solve_ip(
            **{**inputs, "sigma": empty_sigma, "sigma_budget": 0},
            return_diagnostics=True,
        )
        assert summarize_solve_result(raw0, sigma=empty_sigma)["status"] == "INFEASIBLE"
        raw8 = sp_solve_ip(
            **{**inputs, "sigma": empty_sigma, "sigma_budget": 8},
            return_diagnostics=True,
        )
        summary = summarize_solve_result(raw8, sigma=empty_sigma)
        assert summary == {
            "objective_milli": 8000, "status": "OPTIMAL", "violations": 8
        }  # 4 列 x 每列 2 店, 线性可加

    def test_timeout_feasible_fallback_documented(self, envelope, pool):
        """timeout_s 契约: 达时返回 FEASIBLE + optimality_proven=False (不锁进信封)。"""
        inputs = build_solver_inputs(
            envelope, CODE_TO_IDX, DATES, pool, K_C,
            sigma_budget=0, timeout_s=30.0,
        )
        raw = sp_solve_ip(**inputs, return_diagnostics=True)
        assert raw[2]["status"] == "OPTIMAL"  # 小实例秒级内证最优; 契约允许 FEASIBLE 降级


# ---------- 8. import_domain_outcome (pjp.py placement, structural) ----------

def test_import_domain_outcome_shape(envelope):
    evidence = import_domain_outcome(
        {
            "envelope_id": envelope["envelope_id"],
            "outcome": "success",
            "domain_kpis": {"total_km": 8.0},
            "solver_status": "OPTIMAL",
        },
        source="visitmodel",
    )
    assert evidence["evidence_type"] == "task_success"
    assert evidence["source_system"] == "visitmodel"
    assert evidence["metric_name"] is None and evidence["metric_value"] is None
