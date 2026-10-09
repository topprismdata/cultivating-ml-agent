"""ADAPTER-002 (PJP): EstimateEnvelope -> sp_solve_ip kwargs (SPEC-006).

Solver-boundary adapter. This module NEVER imports ortools / VisitModel /
VisitIR and never calls a solver: :func:`build_solver_inputs` only assembles
the exact kwargs dict of ``visitmodel.sp.formulation.sp_solve_ip`` (in-memory
dict calling convention); executing the solve and verifying it round-trip is
the caller's/test's job (guarded by ``VISITMODEL_PATH``).

Contract locked across the boundary (determinism note): CP-SAT runs with
``num_search_workers=8`` and no random seed, so tie-broken *selected days* may
drift between runs. Only ``objective_milli`` (int(round(km*1000)) — the
objective is integral in milli-km), ``status`` and the violation counts are
locked; :func:`summarize_solve_result` deliberately drops ``days``.

Identity discipline: the spec side speaks stable store codes; the solver side
speaks bare int idx. The code->idx mapping is caller-supplied per call and
never persisted anywhere (D7).
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Optional, Sequence

from .estimates import (
    ENVELOPE_ID_RE,
    canonical_bytes,
)

__all__ = [
    "ADAPTER_ID",
    "SOURCE_SYSTEM_VISITMODEL",
    "SOURCE_SYSTEM_WAREHOUSE_ENGINE",
    "EVIDENCE_TYPE_SUCCESS",
    "EVIDENCE_TYPE_FAILURE",
    "DOMAIN_OUTCOME_PROTOCOL_VERSION",
    "AdapterError",
    "build_solver_inputs",
    "summarize_solve_result",
    "count_sigma_violations",
    "import_domain_outcome",
]

ADAPTER_ID = "ADAPTER-002"

SOURCE_SYSTEM_VISITMODEL = "visitmodel"
SOURCE_SYSTEM_WAREHOUSE_ENGINE = "warehouse-engine"

EVIDENCE_TYPE_SUCCESS = "task_success"
EVIDENCE_TYPE_FAILURE = "task_failure"

#: Domain-reported outcome vocabulary -> ANF evidence_type.
_OUTCOME_MAP = {
    "success": EVIDENCE_TYPE_SUCCESS,
    "failure": EVIDENCE_TYPE_FAILURE,
    EVIDENCE_TYPE_SUCCESS: EVIDENCE_TYPE_SUCCESS,
    EVIDENCE_TYPE_FAILURE: EVIDENCE_TYPE_FAILURE,
}

DOMAIN_OUTCOME_PROTOCOL_VERSION = "0.1.0"
DOMAIN_OUTCOME_PROTOCOL = (
    "domain-outcome/0.1.0: 业务 KPI 是领域上报事实，仅存 domain_kpis 字段，"
    "永不写入 metric_name/metric_value（ML 指标通道）；evidence_id 由内容 sha256 派生。"
)

_ALLOWED_PAYLOAD_KEYS = frozenset(
    {"envelope_id", "outcome", "domain_kpis", "solver_status", "task_id", "timestamp"}
)


class AdapterError(Exception):
    """Cross-domain adapter contract violation."""


# ---------------------------------------------------------------------------
# Envelope -> sp_solve_ip kwargs
# ---------------------------------------------------------------------------

def _quantized_sigma_by_code(envelope: Mapping) -> dict:
    """{unit_id: set(1..7)} from quantized payloads (validated)."""
    sigma_by_code = {}
    for unit in envelope.get("units", ()):
        payload = unit.get("quantized")
        if payload:
            weekdays = payload["weekdays"]
            bad = [w for w in weekdays if not (1 <= w <= 7)]
            if bad:
                # 0/1 基泄漏钉死: VisitContract.sigma 是 0 基, 本边界只收 1 基 ISO。
                raise AdapterError(
                    f"unit {unit['unit_id']!r} 的 quantized.weekdays 含 {bad}: "
                    "本边界只接受 1 基 ISO 星期几 (1..7)；0 基是 VisitContract.sigma 语义，"
                    "须先 +1 转换（SPEC-006 §3）"
                )
            sigma_by_code[str(unit["unit_id"])] = {int(w) for w in weekdays}
    return sigma_by_code


def build_solver_inputs(
    envelope: Mapping,
    code_to_idx: Mapping,
    dates: Sequence,
    pool: Sequence,
    k_c: Mapping,
    *,
    sigma_budget: int,
    timeout_s: float = 60.0,
) -> dict:
    """Assemble the exact kwargs dict of ``sp_solve_ip`` (no solving here).

    - ``envelope``: EstimateEnvelope with quantized 1-based ISO weekday sets.
    - ``code_to_idx``: caller-supplied stable-code -> int idx mapping; used
      transiently, never persisted.
    - ``dates``/``pool``: solver-native (dates list; pool columns
      ``(date, route[idx], km)`` — km is the only numeric cost channel).
    - ``k_c``: code-keyed obligations ``{store_code: count}``; the adapter
      maps them through ``code_to_idx`` (identity discipline: callers speak
      codes at this boundary).
    - ``sigma_budget`` must be given together with sigma to take effect
      (sp_solve_ip semantics); required here.

    Every store in ``k_c`` must carry a quantized weekday set: the solver
    treats a store missing from sigma as always-violating
    (``w not in sigma.get(c, set())``), so partial coverage is a silent
    constraint footgun and is refused here.
    """
    if not isinstance(code_to_idx, Mapping) or not code_to_idx:
        raise AdapterError("code_to_idx 必须是非空映射（稳定编码 -> int idx）")
    idx_values = list(code_to_idx.values())
    if any(not isinstance(i, int) or isinstance(i, bool) for i in idx_values):
        raise AdapterError("code_to_idx 的值必须是 int idx")
    if len(set(idx_values)) != len(idx_values):
        raise AdapterError("code_to_idx 的 idx 值必须唯一")
    if not isinstance(sigma_budget, int) or isinstance(sigma_budget, bool) or sigma_budget < 0:
        raise AdapterError("sigma_budget 必须是非负 int")

    sigma_by_code = _quantized_sigma_by_code(envelope)

    k_c_codes = {str(code): k_c[code] for code in k_c}
    missing_sigma = sorted(set(k_c_codes) - set(sigma_by_code))
    if missing_sigma:
        raise AdapterError(
            f"k_c 中的店 {missing_sigma} 缺少 quantized weekday 集合: "
            "部分覆盖会让求解器把这些店视为恒违规，适配器拒绝拼装"
        )
    unknown_codes = sorted(
        (set(k_c_codes) | set(sigma_by_code)) - set(str(c) for c in code_to_idx)
    )
    if unknown_codes:
        raise AdapterError(f"信封/k_c 引用了 code_to_idx 之外的店编码: {unknown_codes}")

    sigma = {
        code_to_idx[code]: set(sigma_by_code[code]) for code in sigma_by_code
    }
    k_c_idx = {code_to_idx[code]: int(k) for code, k in k_c_codes.items()}

    dates = list(dates)
    pool = list(pool)
    known_idx = set(idx_values)
    for column in pool:
        date, route, km = column
        if date not in set(dates):
            raise AdapterError(f"pool 列日期 {date!r} 不在 dates 内")
        if any(c not in known_idx for c in route):
            raise AdapterError(f"pool 列 {date!r} 的 route 含未知 idx: {list(route)}")
        if not isinstance(km, (int, float)) or isinstance(km, bool) or km < 0:
            raise AdapterError(f"pool 列 {date!r} 的 km 必须是 >=0 数值（唯一成本通道）")

    return {
        "dates": dates,
        "k_c": k_c_idx,
        "pool": pool,
        "sigma": sigma,
        "sigma_budget": int(sigma_budget),
        "timeout_s": float(timeout_s),
    }


# ---------------------------------------------------------------------------
# Solver result -> deterministic summary (days deliberately dropped)
# ---------------------------------------------------------------------------

def count_sigma_violations(days: Mapping, sigma: Mapping) -> int:
    """Total predicted-service-day violations over selected columns.

    Each store is covered exactly once per column, so per-column violations
    are linearly additive (mirrors sp_solve_ip's own viol_terms construction):
    for each selected (date, route): weekday w = date.weekday()+1 (1-based
    ISO); each store c in route with w not in sigma[c] adds one violation.
    """
    total = 0
    for date, route in days.items():
        w = date.weekday() + 1
        for c in route:
            if w not in sigma.get(c, set()):
                total += 1
    return total


def summarize_solve_result(
    result,
    *,
    sigma: Optional[Mapping] = None,
) -> dict:
    """Extract ONLY the deterministic fields from a solver result.

    ``result``: the ``(objective_km, days[, diagnostics])`` return of
    ``sp_solve_ip`` (``return_diagnostics=True`` recommended). Returns::

        {"objective_milli": int|None, "status": str|None, "violations": int|None}

    ``objective_milli`` = int(round(km*1000)) — bit-deterministic (the solver
    objective itself is integral in milli-km). ``days`` (selected routes) is
    consumed for violation counting and then DISCARDED: ties may drift across
    runs (multi-worker, no seed), so days are never part of the contract.
    ``violations`` requires the ``sigma`` context (pass the solver-side dict
    from :func:`build_solver_inputs`); without it the count is None — never
    fabricated.
    """
    if not isinstance(result, tuple) or len(result) not in (2, 3):
        raise AdapterError("result 必须是 sp_solve_ip 的 (km, days[, diagnostics]) 元组")
    km, days = result[0], result[1]
    diagnostics = result[2] if len(result) == 3 else {}
    if not isinstance(diagnostics, Mapping):
        raise AdapterError("diagnostics 必须是映射")
    objective_milli = None if km is None else int(round(float(km) * 1000))
    if objective_milli is not None and diagnostics.get("objective_value_milli") is not None:
        # 契约自检: km 整数化与求解器目标毫值一致（CP-SAT 目标本身是整数毫值）
        delta = abs(objective_milli - float(diagnostics["objective_value_milli"]))
        if delta > 1.0:
            raise AdapterError(
                f"objective_milli {objective_milli} 与 diagnostics.objective_value_milli "
                f"{diagnostics['objective_value_milli']} 不一致"
            )
    violations = None
    if days is not None and sigma is not None:
        violations = count_sigma_violations(days, sigma)
    return {
        "objective_milli": objective_milli,
        "status": diagnostics.get("status"),
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# Domain outcome -> ANF evidence-envelope
# ---------------------------------------------------------------------------

def _provenance_chain_id(envelope_id: str) -> str:
    """``pch-`` + 12 hex of sha256 over the canonical envelope reference."""
    seed = {"envelope_id": envelope_id}
    return "pch-" + hashlib.sha256(canonical_bytes(seed)).hexdigest()[:12]


def import_domain_outcome(payload: Mapping, *, source: str) -> dict:
    """Fold a domain-reported outcome JSON into an ANF evidence-envelope dict.

    Strict boundary:
    - ``source`` is caller-provided (e.g. ``'visitmodel'`` /
      ``'warehouse-engine'``) and lands verbatim in ``source_system``;
    - ``payload.outcome`` maps to ``evidence_type`` ∈ {task_success,
      task_failure}; anything else is rejected;
    - business KPIs ride verbatim in ``domain_kpis`` — they are domain-reported
      facts and NEVER touch ``metric_name``/``metric_value`` (ML metric
      channel), which are pinned to null here;
    - ``evidence_id`` is sha256-derived from content (excluding evidence_id /
      timestamp — the volatile pair); no uuid4, no wall clock: the import path
      is fully deterministic;
    - provenance points back at the envelope:
      ``provenance_refs.envelope_id`` verbatim plus a ``pch-`` chain id
      derived from it.

    Unknown payload keys are rejected (import boundary strictness).
    """
    if not isinstance(source, str) or not source.strip():
        raise AdapterError("source 必须是非空字符串（如 'visitmodel'/'warehouse-engine'）")
    if not isinstance(payload, Mapping):
        raise AdapterError("payload 必须是映射")
    unknown = sorted(set(payload) - _ALLOWED_PAYLOAD_KEYS)
    if unknown:
        raise AdapterError(f"payload 含未知字段: {unknown}（import 边界严格拒绝）")
    envelope_id = payload.get("envelope_id")
    if not isinstance(envelope_id, str) or not ENVELOPE_ID_RE.match(envelope_id):
        raise AdapterError("payload.envelope_id 必须匹配 ^env-[0-9a-f]{12}$")
    outcome = payload.get("outcome")
    try:
        evidence_type = _OUTCOME_MAP[outcome]
    except (KeyError, TypeError):
        raise AdapterError(
            f"payload.outcome 必须是 success/failure/{EVIDENCE_TYPE_SUCCESS}/"
            f"{EVIDENCE_TYPE_FAILURE}，得到 {outcome!r}"
        ) from None
    domain_kpis = payload.get("domain_kpis") or {}
    if not isinstance(domain_kpis, Mapping) or any(not isinstance(k, str) for k in domain_kpis):
        raise AdapterError("payload.domain_kpis 必须是 {str: 数值|字符串} 映射")
    for key in ("solver_status", "task_id", "timestamp"):
        if payload.get(key) is not None and not isinstance(payload[key], str):
            raise AdapterError(f"payload.{key} 必须是字符串或缺省")

    content = {
        "capability_id": "cross-domain-exchange",
        "skill_id": "domain-outcome-import",
        "evidence_type": evidence_type,
        "source_system": source,
        "measurement_protocol": DOMAIN_OUTCOME_PROTOCOL,
        "protocol_version": DOMAIN_OUTCOME_PROTOCOL_VERSION,
        "executor": source,
        # ML 指标通道钉死为空: 业务 KPI 永不冒充 ML 指标（SPEC-006 §1 铁律）
        "metric_name": None,
        "metric_value": None,
        "metric_version": None,
        "provenance_chain_id": _provenance_chain_id(envelope_id),
        "provenance_refs": {"envelope_id": envelope_id},
        "task_id": payload.get("task_id") or envelope_id,
        "domain_kpis": dict(domain_kpis),
        "solver_status": payload.get("solver_status"),
        "artifact_refs": [],
        "timestamp": payload.get("timestamp"),
    }
    identity = {k: v for k, v in content.items() if k not in ("timestamp",)}
    evidence_id = "ev-" + hashlib.sha256(canonical_bytes(identity)).hexdigest()
    evidence = {"evidence_id": evidence_id}
    evidence.update(content)
    return evidence
