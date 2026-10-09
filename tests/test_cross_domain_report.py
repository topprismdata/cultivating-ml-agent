"""G5 cross-domain report tests (SPEC-006 §6).

Covers: the four strictly separated columns (ML 指标 / 业务 KPI / 约束违规数 /
计算成本 — no cross-column computation anywhere), structural JSON/Markdown
agreement, and determinism (same cases -> byte-identical output).
"""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.src.adapters.report import (
    COLUMN_HEADERS,
    ReportError,
    cross_domain_report,
    cross_domain_report_json,
    normalize_case,
)

CASES = [
    {
        "case_id": "pjp-4store",
        "ml_metric": {"name": "oof_mae", "value": 0.1234567},
        "domain_kpis": {"total_km": 8.0, "service_rate": "97%"},
        "violations": 0,
        "compute_cost": {"wall_time_s": 0.0123456},
    },
    {
        "case_id": "pjp-misaligned",
        "ml_metric": {"name": "oof_mae", "value": 0.5},
        "domain_kpis": {"total_km": 8.0},
        "violations": 4,
        "compute_cost": {"wall_time_s": 0.02},
    },
    {
        "case_id": "warehouse-demand",
        "ml_metric": None,
        "domain_kpis": {},
        "violations": None,
        "compute_cost": None,
    },
]


def _table_rows(markdown: str) -> list:
    return [
        line
        for line in markdown.splitlines()
        if line.startswith("|") and " ML 指标 " not in line and not line.startswith("|--")
    ]


# ---------- 1. Four columns, strictly separated ----------

def test_report_has_exactly_four_columns():
    markdown = cross_domain_report(CASES)
    header = next(line for line in markdown.splitlines() if line.startswith("| 案例"))
    cells = [c.strip() for c in header.strip("|").split("|")]
    assert cells == ["案例", *COLUMN_HEADERS]
    assert len(cells) == 5  # 案例 + 四列


def test_columns_do_not_mix():
    """四列不混算: ML 指标只出现在 ML 列; 业务 KPI 只出现在 KPI 列。"""
    markdown = cross_domain_report(CASES)
    rows = _table_rows(markdown)
    assert len(rows) == len(CASES)
    first = [c.strip() for c in rows[0].strip("|").split("|")]
    assert first[0] == "pjp-4store"
    # ML 列: 只有 ML 指标
    assert first[1] == "oof_mae=0.123457"
    # 业务 KPI 列: 原样转录, 领域上报事实, 不与 ML 指标合并运算 (键按字典序)
    assert first[2] == "service_rate=97%; total_km=8.000000"
    # 约束违规数 / 计算成本: 求解域字段独立成列
    assert first[3] == "0"
    assert first[4] == "wall_time_s=0.012346"
    # 任何一行都不含跨列混算产物 (如 ml 与 kpi 的加减/平均字样)
    for row in rows:
        assert "avg" not in row.lower() and "mixed" not in row.lower()


def test_null_cells_render_as_na_not_zero():
    markdown = cross_domain_report(CASES)
    last = [c.strip() for c in _table_rows(markdown)[2].strip("|").split("|")]
    assert last[1:] == ["n/a", "n/a", "n/a", "n/a"]  # None 不伪造为 0


# ---------- 2. JSON block agreement ----------

def test_json_block_matches_markdown_and_schema():
    markdown = cross_domain_report(CASES)
    block = re.search(r"```json\n(.*?)\n```", markdown, re.DOTALL).group(1)
    payload = json.loads(block)
    assert payload["columns"] == list(COLUMN_HEADERS)
    assert len(payload["cases"]) == len(CASES)
    case0 = payload["cases"][0]
    assert case0["ml_metric"] == {"name": "oof_mae", "value": 0.123457}  # 6 位舍入
    assert case0["domain_kpis"] == {"service_rate": "97%", "total_km": 8.0}
    assert case0["violations"] == 0
    assert case0["compute_cost"] == {"wall_time_s": 0.012346}
    assert cross_domain_report_json(CASES) == block


# ---------- 3. Determinism ----------

def test_report_deterministic_same_input_same_output():
    a = cross_domain_report(CASES)
    b = cross_domain_report(CASES)
    assert a == b
    import copy

    shuffled = list(reversed([copy.deepcopy(c) for c in CASES]))
    # 顺序是调用方语义的一部分: 不同顺序 = 不同报告 (无隐藏重排)
    assert cross_domain_report(shuffled) != a


# ---------- 4. Case contract validation ----------

def test_case_validation_rejects_pollution_and_mistyping():
    with pytest.raises(ReportError, match="未知字段"):
        normalize_case({**CASES[0], "metric_value": 1.0})
    with pytest.raises(ReportError, match="缺少字段"):
        normalize_case({"case_id": "x"})
    with pytest.raises(ReportError, match="violations"):
        normalize_case({**CASES[0], "violations": "none"})  # 违规数是计数, 不收字符串
    with pytest.raises(ReportError, match="ml_metric.value"):
        normalize_case({**CASES[0], "ml_metric": {"name": "x", "value": "high"}})
    with pytest.raises(ReportError, match="domain_kpis"):
        normalize_case({**CASES[0], "domain_kpis": {"k": ["list"]}})


def test_report_markdown_discipline_note_present():
    markdown = cross_domain_report(CASES[:1])
    assert "不与 ML 指标混算" in markdown  # 字段纪律声明随报告输出
