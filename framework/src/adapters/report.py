"""G5 cross-domain report (SPEC-006 §6): strictly separated columns.

``cross_domain_report(cases)`` renders a deterministic markdown report (with
an embedded canonical JSON block) where every case shows four strictly
separated columns:

- ML 指标 — model-side statistics (e.g. OOF metric); the ONLY place ML
  numbers appear;
- 业务 KPI — domain-reported facts, verbatim, never recomputed or mixed with
  ML metrics;
- 约束违规数 — solver-side hard-constraint violation count;
- 计算成本 — compute cost (solver wall time).

Nothing is ever computed ACROSS columns (no mixing/averaging/normalizing ML
metrics with business KPIs — plan baseline §12 iron rule). Pure function:
same cases -> byte-identical output; no clock, no randomness.
"""
from __future__ import annotations

import json
from typing import Mapping, Optional

__all__ = [
    "REPORT_TITLE",
    "COLUMN_HEADERS",
    "ReportError",
    "normalize_case",
    "cross_domain_report",
    "cross_domain_report_json",
]

REPORT_TITLE = "跨域交换报告（G5）"

#: Exactly four data columns, in order. Strictly separated, never merged.
COLUMN_HEADERS = ("ML 指标", "业务 KPI", "约束违规数", "计算成本")

_REQUIRED_CASE_KEYS = frozenset(
    {"case_id", "ml_metric", "domain_kpis", "violations", "compute_cost"}
)

_FLOATS = "%.6f"


class ReportError(Exception):
    """Cross-domain report case contract violation."""


def _fmt_float(value: float) -> str:
    return _FLOATS % float(value)


def _fmt_ml_metric(ml_metric: Optional[Mapping]) -> str:
    if ml_metric is None:
        return "n/a"
    unknown = sorted(set(ml_metric) - {"name", "value"})
    if unknown:
        raise ReportError(f"ml_metric 含未知字段: {unknown}")
    name = ml_metric.get("name")
    value = ml_metric.get("value")
    if not isinstance(name, str) or not name.strip():
        raise ReportError("ml_metric.name 必须是非空字符串")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ReportError("ml_metric.value 必须是数值")
    return f"{name}={_fmt_float(value)}"


def _fmt_kpis(domain_kpis: Mapping) -> str:
    if not domain_kpis:
        return "n/a"
    cells = []
    for key in sorted(domain_kpis):
        value = domain_kpis[key]
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ReportError(f"domain_kpis[{key!r}] 必须是数值或字符串（领域上报事实）")
        rendered = _fmt_float(value) if isinstance(value, (int, float)) else str(value)
        cells.append(f"{key}={rendered}")
    return "; ".join(cells)


def _fmt_violations(violations: Optional[int]) -> str:
    if violations is None:
        return "n/a"
    if not isinstance(violations, int) or isinstance(violations, bool):
        raise ReportError("violations 必须是 int 或 null（约束违规计数，不混算）")
    return str(violations)


def _fmt_compute_cost(compute_cost: Optional[Mapping]) -> str:
    if compute_cost is None:
        return "n/a"
    unknown = sorted(set(compute_cost) - {"wall_time_s"})
    if unknown:
        raise ReportError(f"compute_cost 含未知字段: {unknown}")
    wall_time_s = compute_cost.get("wall_time_s")
    if wall_time_s is None:
        return "n/a"
    if not isinstance(wall_time_s, (int, float)) or isinstance(wall_time_s, bool):
        raise ReportError("compute_cost.wall_time_s 必须是数值或 null")
    return f"wall_time_s={_fmt_float(wall_time_s)}"


def normalize_case(case: Mapping) -> dict:
    """Validate one case and return the canonical (JSON-ready) dict.

    Canonicalization rounds every float to 6 decimals so the JSON block is
    byte-deterministic for identical inputs.
    """
    if not isinstance(case, Mapping):
        raise ReportError("case 必须是映射")
    unknown = sorted(set(case) - _REQUIRED_CASE_KEYS)
    if unknown:
        raise ReportError(f"case 含未知字段: {unknown}")
    missing = sorted(_REQUIRED_CASE_KEYS - set(case))
    if missing:
        raise ReportError(f"case 缺少字段: {missing}")
    case_id = case["case_id"]
    if not isinstance(case_id, str) or not case_id.strip():
        raise ReportError("case_id 必须是非空字符串")
    ml_metric = case["ml_metric"]
    if ml_metric is not None and not isinstance(ml_metric, Mapping):
        raise ReportError("ml_metric 必须是映射或 null")
    domain_kpis = case["domain_kpis"]
    if not isinstance(domain_kpis, Mapping):
        raise ReportError("domain_kpis 必须是映射（可为空）")
    violations = case["violations"]
    if violations is not None and (
        not isinstance(violations, int) or isinstance(violations, bool)
    ):
        raise ReportError("violations 必须是 int 或 null")
    compute_cost = case["compute_cost"]
    if compute_cost is not None and not isinstance(compute_cost, Mapping):
        raise ReportError("compute_cost 必须是映射或 null")

    canonical_ml = None
    if ml_metric is not None:
        ml_value = ml_metric.get("value")
        if not isinstance(ml_value, (int, float)) or isinstance(ml_value, bool):
            raise ReportError("ml_metric.value 必须是数值")
        canonical_ml = {
            "name": str(ml_metric["name"]),
            "value": round(float(ml_value), 6),
        }
    canonical_kpis = {}
    for key in sorted(domain_kpis):
        value = domain_kpis[key]
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ReportError(
                f"domain_kpis[{key!r}] 必须是数值或字符串（领域上报事实，不混算）"
            )
        canonical_kpis[str(key)] = (
            round(float(value), 6) if isinstance(value, (int, float)) else str(value)
        )
    canonical_cost = None
    if compute_cost is not None and compute_cost.get("wall_time_s") is not None:
        canonical_cost = {"wall_time_s": round(float(compute_cost["wall_time_s"]), 6)}
    return {
        "case_id": case_id,
        "ml_metric": canonical_ml,
        "domain_kpis": canonical_kpis,
        "violations": violations,
        "compute_cost": canonical_cost,
    }


def cross_domain_report_json(cases: list) -> str:
    """Canonical JSON block content: sorted keys, 6-decimal floats."""
    normalized = [normalize_case(case) for case in cases]
    return json.dumps(
        {"title": REPORT_TITLE, "columns": list(COLUMN_HEADERS), "cases": normalized},
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
    )


def cross_domain_report(cases: list) -> str:
    """Render the G5 markdown report (with embedded canonical JSON block)."""
    normalized = [normalize_case(case) for case in cases]
    lines = [
        f"# {REPORT_TITLE}",
        "",
        "字段纪律：四列严格分列——ML 指标只在 ML 列；业务 KPI 是领域上报事实，"
        "原样转录不与 ML 指标混算；约束违规数与计算成本属于求解域。",
        "",
        "| 案例 | " + " | ".join(COLUMN_HEADERS) + " |",
        "|---" * (len(COLUMN_HEADERS) + 1) + "|",
    ]
    for case in normalized:
        lines.append(
            "| {case_id} | {ml} | {kpis} | {violations} | {cost} |".format(
                case_id=case["case_id"],
                ml=_fmt_ml_metric(case["ml_metric"]),
                kpis=_fmt_kpis(case["domain_kpis"]),
                violations=_fmt_violations(case["violations"]),
                cost=_fmt_compute_cost(case["compute_cost"]),
            )
        )
    lines += ["", "```json", cross_domain_report_json(cases), "```", ""]
    return "\n".join(lines)
