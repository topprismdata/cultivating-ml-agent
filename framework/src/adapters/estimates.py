"""EstimateEnvelope v0.1 builder + identity hashing (ADAPTER-002/003, SPEC-006).

Pure ML-side module: turns an OOF frame (``build_oof_frame`` product written to
CSV, with ground truth) into a per-store EstimateEnvelope dict. Zero solver
dependency — importing this module must never pull ortools/VisitModel.

Identity discipline (mirrors ir.decision_trace / ir.experiment_ir):
- canonical bytes = sorted keys, no insignificant space, UTF-8;
- ``created_at`` never enters any hash (injected clocks and replay cannot
  change identity);
- ``content_hash`` = sha256(canonical(envelope - {created_at, content_hash,
  envelope_id}));
- ``envelope_id`` = ``env-`` + first 12 hex of ``content_hash`` — sha256
  derived, never uuid4, never wall clock.

Sigma triple-meaning trap (SPEC-006 §3): ``units[].sigma`` here is the
residual standard deviation of the estimate; ``VisitContract.sigma`` (visit_ir
visit_semantic_api) is a 0-based weekday int (0=Monday);
``sp_solve_ip(sigma=...)`` takes 1-based ISO weekday sets (1=Monday). This
module speaks 1-based ISO in quantized payloads and rejects 0 — a 0 found in
a weekday set is 0/1-based leakage, not a legal weekday.

Quantization rule ``p80_weekday_coverage``: per store, per weekday mean of
oof_pred; the store's weekday set = weekdays whose mean >= the store's P80
quantile over weekday means. Pure function of the frame (idempotent).
"""
from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pandas as pd

from ..ir.decision_trace import canonical_bytes

__all__ = [
    "SCHEMA_VERSION",
    "ESTIMATE_KINDS",
    "ESTIMATE_KIND_PJP_SERVICE_WEEKDAY",
    "ESTIMATE_KIND_WAREHOUSE_DEMAND",
    "ESTIMATE_KIND_WAREHOUSE_PROCESSING_TIME",
    "QUANTIZATION_P80_WEEKDAY_COVERAGE",
    "ENVELOPE_ID_RE",
    "CONTENT_HASH_RE",
    "WEEKDAY_MIN",
    "WEEKDAY_MAX",
    "EnvelopeError",
    "canonical_bytes",
    "envelope_identity_body",
    "envelope_content_hash",
    "compute_envelope_id",
    "verify_envelope_identity",
    "validate_envelope",
    "quantize_p80_weekday_coverage",
    "apply_quantization",
    "envelope_from_oof",
]

SCHEMA_VERSION = "0.1.0"

ESTIMATE_KIND_PJP_SERVICE_WEEKDAY = "pjp_service_weekday"
ESTIMATE_KIND_WAREHOUSE_DEMAND = "warehouse_demand"
ESTIMATE_KIND_WAREHOUSE_PROCESSING_TIME = "warehouse_processing_time"
ESTIMATE_KINDS = (
    ESTIMATE_KIND_PJP_SERVICE_WEEKDAY,
    ESTIMATE_KIND_WAREHOUSE_DEMAND,
    ESTIMATE_KIND_WAREHOUSE_PROCESSING_TIME,
)

QUANTIZATION_P80_WEEKDAY_COVERAGE = "p80_weekday_coverage"

ENVELOPE_ID_RE = re.compile(r"^env-[0-9a-f]{12}$")
CONTENT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
EVIDENCE_ID_RE = re.compile(r"^ev-[0-9a-f]{12,64}$")
DECISION_ID_RE = re.compile(r"^dt-[0-9a-f]{12}$")

#: 1-based ISO weekdays (1=Monday .. 7=Sunday). 0 is NEVER legal here — it is
#: the 0-based VisitContract.sigma domain leaking across the boundary.
WEEKDAY_MIN = 1
WEEKDAY_MAX = 7

#: Volatile fields: never hashed. Derived fields: excluded from the identity
#: body (envelope_id is a pure function of that body, content_hash IS it).
_HASH_EXCLUDED_FIELDS = ("created_at", "content_hash", "envelope_id")

_UNIT_FIELDS = frozenset({"unit_id", "value", "sigma", "confidence", "quantized"})
_QUANTIZED_FIELDS = frozenset({"rule", "weekdays", "quantile"})
_PROVENANCE_FIELDS = frozenset({"ir_content_hash", "evidence_id", "decision_id"})

#: Floats are rounded to this many decimals at build time (declared rule,
#: mirrors the runner's ``%.6f`` OOF CSV precision). Deterministic.
FLOAT_DECIMALS = 6


class EnvelopeError(Exception):
    """EstimateEnvelope contract violation."""


def _now_iso(clock: Optional[Callable[[], Any]]) -> str:
    """ISO-8601 UTC timestamp; ``clock`` injectable, default wall clock.

    created_at is volatile and never hashed, so the default does not affect
    identity; the injection exists for byte-level determinism in tests.
    """
    value = clock() if clock is not None else datetime.now(timezone.utc)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, str) and value.strip():
        return value
    raise TypeError("clock must return datetime or ISO-8601 string")


# ---------------------------------------------------------------------------
# Identity hashing (sha256-derived; no uuid4, no wall clock)
# ---------------------------------------------------------------------------

def envelope_identity_body(envelope: Mapping[str, Any]) -> dict:
    """Envelope content covered by the hash: minus created_at/content_hash/envelope_id."""
    return {k: v for k, v in envelope.items() if k not in _HASH_EXCLUDED_FIELDS}


def envelope_content_hash(envelope: Mapping[str, Any]) -> str:
    """sha256 hexdigest of the canonical identity body (64 hex)."""
    return hashlib.sha256(canonical_bytes(envelope_identity_body(envelope))).hexdigest()


def compute_envelope_id(content_hash: str) -> str:
    """``env-`` + first 12 hex of the content hash."""
    return "env-" + content_hash[:12]


def verify_envelope_identity(envelope: Mapping[str, Any]) -> None:
    """Recompute content_hash/envelope_id and compare (tamper / clock check)."""
    content_hash = envelope_content_hash(envelope)
    if envelope.get("content_hash") != content_hash:
        raise EnvelopeError(
            "content_hash mismatch: recomputed "
            f"{content_hash!r} != stored {envelope.get('content_hash')!r}"
        )
    expected_id = compute_envelope_id(content_hash)
    if envelope.get("envelope_id") != expected_id:
        raise EnvelopeError(
            f"envelope_id mismatch: expected {expected_id!r}, "
            f"stored {envelope.get('envelope_id')!r}"
        )


# ---------------------------------------------------------------------------
# Structural validation (stdlib mirror of the JSON Schema; always available)
# ---------------------------------------------------------------------------

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_unit(unit: Mapping[str, Any], index: int) -> list:
    errors = []
    if not isinstance(unit, Mapping):
        return [f"units[{index}] 必须是对象"]
    unknown = sorted(set(unit) - _UNIT_FIELDS)
    if unknown:
        errors.append(f"units[{index}] 含未知字段: {unknown}")
    unit_id = unit.get("unit_id")
    if not isinstance(unit_id, str) or not unit_id.strip():
        errors.append(f"units[{index}].unit_id 必须是非空字符串")
    elif unit_id.isdigit():
        # D7: int idx 不配当身份 — bare digit strings are int idx in disguise.
        errors.append(f"units[{index}].unit_id {unit_id!r} 是裸数字（禁止 int idx 作身份, D7）")
    if not _is_number(unit.get("value")):
        errors.append(f"units[{index}].value 必须是有限数值")
    sigma = unit.get("sigma")
    if not _is_number(sigma) or sigma < 0:
        errors.append(f"units[{index}].sigma 必须是 >=0 的有限数值")
    confidence = unit.get("confidence")
    if confidence is not None and (not _is_number(confidence) or not 0 <= confidence <= 1):
        errors.append(f"units[{index}].confidence 必须在 [0,1]")
    quantized = unit.get("quantized")
    if quantized is not None:
        if not isinstance(quantized, Mapping):
            errors.append(f"units[{index}].quantized 必须是对象")
        else:
            unknown_q = sorted(set(quantized) - _QUANTIZED_FIELDS)
            if unknown_q:
                errors.append(f"units[{index}].quantized 含未知字段: {unknown_q}")
            rule = quantized.get("rule")
            if not isinstance(rule, str) or not rule.strip():
                errors.append(f"units[{index}].quantized.rule 必须是非空字符串")
            weekdays = quantized.get("weekdays")
            if (
                not isinstance(weekdays, list)
                or not weekdays
                or any(not isinstance(w, int) or isinstance(w, bool) for w in weekdays)
                or any(not WEEKDAY_MIN <= w <= WEEKDAY_MAX for w in weekdays)
                or len(set(weekdays)) != len(weekdays)
            ):
                errors.append(
                    f"units[{index}].quantized.weekdays 必须是 {WEEKDAY_MIN}..{WEEKDAY_MAX} "
                    "内不重复的非空整数列表（1 基 ISO；0 = 0/1 基泄漏）"
                )
            quantile = quantized.get("quantile")
            if quantile is not None and (not _is_number(quantile) or not 0 <= quantile <= 1):
                errors.append(f"units[{index}].quantized.quantile 必须在 [0,1]")
    return errors


def validate_envelope(envelope: Mapping[str, Any]) -> list:
    """Structural validation; returns a list of violation strings (empty = legal)."""
    errors = []
    if not isinstance(envelope, Mapping):
        return ["envelope 必须是对象"]
    required = (
        "schema_version",
        "envelope_id",
        "estimate_kind",
        "created_at",
        "model_provenance",
        "units",
        "quantization_rule",
        "value_unit",
        "content_hash",
    )
    for field in required:
        if field not in envelope:
            errors.append(f"缺少必填字段: {field}")
    unknown = sorted(set(envelope) - set(required))
    if unknown:
        errors.append(f"含未知顶层字段: {unknown}")
    if errors:
        return errors

    if envelope["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version 必须是 {SCHEMA_VERSION!r}")
    if not ENVELOPE_ID_RE.match(envelope["envelope_id"]):
        errors.append("envelope_id 必须匹配 ^env-[0-9a-f]{12}$")
    if not CONTENT_HASH_RE.match(envelope["content_hash"]):
        errors.append("content_hash 必须是 64 位小写 hex")
    if envelope["estimate_kind"] not in ESTIMATE_KINDS:
        errors.append(f"estimate_kind 必须属于 {ESTIMATE_KINDS}")
    if not isinstance(envelope["created_at"], str) or not envelope["created_at"].strip():
        errors.append("created_at 必须是非空字符串")
    value_unit = envelope["value_unit"]
    if not isinstance(value_unit, str) or not value_unit.strip():
        errors.append("value_unit 必须是非空字符串（如 minutes/km/units_per_day）")
    quantization_rule = envelope["quantization_rule"]
    if quantization_rule is not None and (
        not isinstance(quantization_rule, str) or not quantization_rule.strip()
    ):
        errors.append("quantization_rule 必须是非空字符串或 null")

    provenance = envelope["model_provenance"]
    if not isinstance(provenance, Mapping):
        errors.append("model_provenance 必须是对象")
    else:
        unknown_p = sorted(set(provenance) - _PROVENANCE_FIELDS)
        if unknown_p:
            errors.append(f"model_provenance 含未知字段: {unknown_p}")
        if not CONTENT_HASH_RE.match(provenance.get("ir_content_hash", "")):
            errors.append("model_provenance.ir_content_hash 必须是 64 位小写 hex")
        if not EVIDENCE_ID_RE.match(provenance.get("evidence_id", "")):
            errors.append("model_provenance.evidence_id 必须匹配 ^ev-[0-9a-f]{12,64}$")
        decision_id = provenance.get("decision_id")
        if decision_id is not None and not DECISION_ID_RE.match(decision_id):
            errors.append("model_provenance.decision_id 必须匹配 ^dt-[0-9a-f]{12}$")

    units = envelope["units"]
    if not isinstance(units, list) or not units:
        errors.append("units 必须是非空数组")
    else:
        for i, unit in enumerate(units):
            errors.extend(_validate_unit(unit, i))

    quantized_units = [
        u for u in units if isinstance(u, Mapping) and u.get("quantized") is not None
    ] if isinstance(units, list) else []
    if quantized_units and quantization_rule is None:
        errors.append("units 携带 quantized 载荷时 quantization_rule 不得为 null")
    if quantization_rule is not None:
        for i, u in enumerate(quantized_units):
            if u["quantized"].get("rule") != quantization_rule:
                errors.append(
                    f"units[{i}].quantized.rule 与信封级 quantization_rule 不一致"
                )
    return errors


# ---------------------------------------------------------------------------
# Quantization: residual sigma -> (1-based ISO weekday set)
# ---------------------------------------------------------------------------

def quantize_p80_weekday_coverage(
    frame: pd.DataFrame,
    *,
    group_col: str,
    pred_col: str = "oof_pred",
    time_col: str = "time",
    quantile: float = 0.8,
) -> dict:
    """Per store: weekday mean >= P80(weekday means) -> weekday in the set.

    Pure function of the frame → idempotent (re-application yields identical
    sets). Weekday basis conversion happens exactly here:
    ``pandas``/ISO ``weekday()`` is 0-based (Mon=0); the envelope speaks
    1-based ISO (Mon=1), matching the sp_solve_ip transport format.
    """
    if quantile < 0 or quantile > 1:
        raise ValueError("quantile 必须在 [0,1]")
    work = frame.loc[:, [group_col, time_col, pred_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col])
    work = work.dropna(subset=[pred_col])
    work["_weekday"] = (work[time_col].dt.weekday + 1).astype(int)  # 0 基 -> 1 基
    out: dict = {}
    for code, sub in work.groupby(group_col, sort=True):
        means = sub.groupby("_weekday", sort=True)[pred_col].mean()
        theta = float(np.percentile(means.to_numpy(dtype=float), quantile * 100.0))
        out[str(code)] = sorted(int(w) for w in means.index if float(means[w]) >= theta)
        # 非空性由构造保证: 最大均值 >= 任意分位数; dropna 后无 NaN 路径。
        if not out[str(code)]:  # pragma: no cover - 显式契约保险
            raise EnvelopeError(f"unit {code!r} 量化结果为空集（P80 规则下不应发生）")
    return out


def apply_quantization(
    frame: pd.DataFrame,
    *,
    group_col: str,
    rule: Optional[str],
    quantile: float = 0.8,
) -> dict:
    """Dispatch a declared quantization rule; ``None`` -> no quantized payload."""
    if rule is None:
        return {}
    if rule == QUANTIZATION_P80_WEEKDAY_COVERAGE:
        if "time" not in frame.columns:
            # time 列不可用: 量化声明保留, quantized 载荷缺省（SPEC-006 §4）
            return {}
        return quantize_p80_weekday_coverage(frame, group_col=group_col, quantile=quantile)
    raise ValueError(f"未知量化规则: {rule!r}")


# ---------------------------------------------------------------------------
# Envelope builder
# ---------------------------------------------------------------------------

def _require_provenance(ir: Mapping, evidence: Mapping) -> dict:
    if not isinstance(ir, Mapping) or not isinstance(ir.get("content_hash"), str):
        raise EnvelopeError("ir 必须是含 content_hash 的 ExperimentIR 文档")
    if not isinstance(evidence, Mapping) or not isinstance(evidence.get("evidence_id"), str):
        raise EnvelopeError("evidence 必须是含 evidence_id 的 ANF evidence-envelope")
    provenance = {
        "ir_content_hash": ir["content_hash"],
        "evidence_id": evidence["evidence_id"],
    }
    decision_id = evidence.get("decision_id")
    if decision_id is not None:
        provenance["decision_id"] = decision_id
    return provenance


def envelope_from_oof(
    oof_csv_path,
    group_col: str,
    ir: Mapping,
    evidence: Mapping,
    *,
    estimate_kind: str = ESTIMATE_KIND_PJP_SERVICE_WEEKDAY,
    value_unit: str = "units_per_day",
    quantization: Optional[str] = QUANTIZATION_P80_WEEKDAY_COVERAGE,
    quantile: float = 0.8,
    clock: Optional[Callable[[], Any]] = None,
) -> dict:
    """Aggregate a truth-bearing OOF CSV into a per-store EstimateEnvelope.

    ``oof_csv_path``: ``build_oof_frame`` product written by ``to_csv`` —
    requires columns ``group_col`` (stable store codes), ``target`` (ground
    truth), ``oof_pred``; optional ``time`` column enables weekday coverage
    quantization.

    Per unit: ``value`` = mean(oof_pred), ``sigma`` = population std of the
    residual (oof_pred - target, ddof=0), both rounded to 6 decimals.
    """
    if estimate_kind not in ESTIMATE_KINDS:
        raise EnvelopeError(f"estimate_kind 必须属于 {ESTIMATE_KINDS}")
    frame = pd.read_csv(oof_csv_path)
    missing = [c for c in (group_col, "target", "oof_pred") if c not in frame.columns]
    if missing:
        raise EnvelopeError(f"OOF 缺少必需列: {missing}（build_oof_frame 契约: 目标列 + oof_pred）")

    quantized = apply_quantization(
        frame, group_col=group_col, rule=quantization, quantile=quantile
    )

    units = []
    residuals = frame["oof_pred"] - frame["target"]
    work = frame.assign(_residual=residuals)
    for code, sub in work.groupby(group_col, sort=True):
        unit = {
            "unit_id": str(code),
            "value": round(float(sub["oof_pred"].mean()), FLOAT_DECIMALS),
            "sigma": round(float(sub["_residual"].std(ddof=0)), FLOAT_DECIMALS),
        }
        payload = quantized.get(str(code))
        if payload:
            unit["quantized"] = {
                "rule": quantization,
                "weekdays": payload,
                "quantile": quantile,
            }
        units.append(unit)
    if not units:
        raise EnvelopeError("OOF 聚合后无任何 unit（group_col 为空？）")

    envelope = {
        "schema_version": SCHEMA_VERSION,
        "estimate_kind": estimate_kind,
        "created_at": _now_iso(clock),
        "model_provenance": _require_provenance(ir, evidence),
        "units": units,
        "quantization_rule": quantization,
        "value_unit": value_unit,
    }
    content_hash = envelope_content_hash(envelope)
    envelope["envelope_id"] = compute_envelope_id(content_hash)
    envelope["content_hash"] = content_hash

    errors = validate_envelope(envelope)
    if errors:
        raise EnvelopeError("构建的 EstimateEnvelope 不合法: " + "; ".join(errors))
    return envelope
