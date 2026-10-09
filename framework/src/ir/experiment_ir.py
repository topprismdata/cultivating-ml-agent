"""ExperimentIR contract loader, hasher and hard-gate checks.

ExperimentIR is the ML-domain experiment contract: once compiled, it may
never be implicitly mutated by a natural-language agent. Any change must
produce a new revision with ``supersedes`` pointing at the previous
``content_hash``, and execution is refused whenever any hard gate fails.

Pure stdlib. ``jsonschema`` is used when installed (local 4.26 and CI);
otherwise a built-in minimal structural check takes over. This module
MUST NOT import mlflow/sklearn — it is runtime infrastructure shared by
every agent, not a training dependency.

Usage:
    from ir.experiment_ir import load_ir, authorize_execution

    ir = load_ir("schemas/experiment-ir/0.1.0/examples/valid-minimal.json")
    verdict = authorize_execution(ir)
    if not verdict.ok:
        raise PermissionError(verdict.reasons)
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:  # optional dependency: validation falls back to stdlib checks
    import jsonschema
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    jsonschema = None

SCHEMA_VERSION = "0.1.0"
SCHEMA_RELPATH = Path("schemas") / "experiment-ir" / SCHEMA_VERSION / "experiment-ir.schema.json"

STRATEGIES = ("stratified", "kfold", "time_based", "group")
DIRECTIONS = ("maximize", "minimize")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Gate ids are part of the contract; never rename or reorder.
G1_TEMPORAL_AVAILABILITY = "G1_TEMPORAL_AVAILABILITY"
G2_SPLIT_SEPARATION = "G2_SPLIT_SEPARATION"
G3_METRIC_DEFINITION = "G3_METRIC_DEFINITION"
G4_BUDGET_BOUNDS = "G4_BUDGET_BOUNDS"
G5_BASELINE_SAME_PROTOCOL = "G5_BASELINE_SAME_PROTOCOL"
G6_RUN_AUTHORIZATION = "G6_RUN_AUTHORIZATION"

_GATE_ORDER = (
    G1_TEMPORAL_AVAILABILITY,
    G2_SPLIT_SEPARATION,
    G3_METRIC_DEFINITION,
    G4_BUDGET_BOUNDS,
    G5_BASELINE_SAME_PROTOCOL,
    G6_RUN_AUTHORIZATION,
)


class IRError(Exception):
    """Base class for ExperimentIR contract violations."""


class IRSchemaError(IRError):
    """Document does not satisfy the ExperimentIR JSON schema."""


class ContentHashMismatch(IRError):
    """Stored ``content_hash`` does not match the recomputed canonical hash.

    This means the document was mutated after compilation; loading is
    refused and the fix is a new revision with ``supersedes``.
    """


@dataclass(frozen=True)
class GateResult:
    """Outcome of one hard gate."""

    gate_id: str
    status: str  # "pass" | "fail" | "not_applicable"
    reason: str

    def __post_init__(self) -> None:
        if self.status not in ("pass", "fail", "not_applicable"):
            raise ValueError(f"invalid gate status: {self.status!r}")


@dataclass(frozen=True)
class AuthorizationVerdict:
    """Aggregate verdict over all hard gates.

    ``ok=True`` (execution allowed) only when no gate has status "fail";
    "not_applicable" gates never block.
    """

    ok: bool
    failed_gates: tuple[str, ...]
    reasons: tuple[str, ...]
    gates: tuple[GateResult, ...] = field(default=())

    @property
    def blocked(self) -> bool:
        return not self.ok


# ---------------------------------------------------------------------------
# Canonical hashing
# ---------------------------------------------------------------------------

def canonical_bytes(ir: dict) -> bytes:
    """Deterministic UTF-8 JSON bytes: sorted keys, no insignificant space."""
    return json.dumps(ir, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compute_content_hash(ir: dict) -> str:
    """sha256 hexdigest of canonical bytes, excluding ``content_hash`` itself."""
    body = {k: v for k, v in ir.items() if k != "content_hash"}
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


# ---------------------------------------------------------------------------
# Schema validation (jsonschema when available, stdlib fallback otherwise)
# ---------------------------------------------------------------------------

_SCHEMA_CACHE: dict[str, Any] = {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def schema_path() -> Path:
    return _repo_root() / SCHEMA_RELPATH


def _load_schema():
    if "schema" not in _SCHEMA_CACHE:
        path = schema_path()
        if not path.is_file():
            raise IRSchemaError(f"schema file not found: {path}")
        _SCHEMA_CACHE["schema"] = json.loads(path.read_text(encoding="utf-8"))
    return _SCHEMA_CACHE["schema"]


def schema_validate(ir: Any) -> None:
    """Validate ``ir`` against the ExperimentIR schema.

    Uses jsonschema (draft 2020-12) when installed; otherwise the built-in
    minimal structural check. ISO-8601 datetime strictness is enforced by
    ``_check_datetime_fields`` on both paths (jsonschema only checks
    ``format`` as an annotation, and only when rfc3339-validator is extra-
    installed, so it can never be the source of truth here).
    """
    if jsonschema is not None:
        schema = _load_schema()
        validator = jsonschema.validators.validator_for(schema)(
            schema,
        )
        errors = sorted(validator.iter_errors(ir), key=lambda e: list(e.absolute_path))
        if errors:
            lines = [
                f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
                for e in errors
            ]
            raise IRSchemaError("ExperimentIR schema validation failed:\n" + "\n".join(lines))
    else:
        _fallback_validate(ir)
    _check_datetime_fields(ir)


def _parse_iso(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp; returns None when unparseable."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


_DATETIME_FIELDS = ("created_at",)  # top level
_DATETIME_SUBFIELDS = {
    "dataset_snapshot_ref": ("as_of", "label_cutoff"),
    "authorization": ("granted_at",),
}


def _check_datetime_fields(ir: dict) -> None:
    problems = []
    for name in _DATETIME_FIELDS:
        if name in ir and _parse_iso(ir[name]) is None:
            problems.append(f"{name}: {ir[name]!r} 不是合法 ISO-8601 时间戳")
    for key, subfields in _DATETIME_SUBFIELDS.items():
        section = ir.get(key)
        if not isinstance(section, dict):
            continue
        for name in subfields:
            if name in section and _parse_iso(section[name]) is None:
                problems.append(f"{key}.{name}: {section[name]!r} 不是合法 ISO-8601 时间戳")
    if problems:
        raise IRSchemaError("ExperimentIR schema validation failed:\n" + "\n".join(problems))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fail(problems: list[str]) -> None:
    if problems:
        raise IRSchemaError("ExperimentIR schema validation failed:\n" + "\n".join(problems))


def _check_str(problems: list[str], path: str, value: Any, min_length: int = 1) -> None:
    if not isinstance(value, str) or (min_length and len(value) < min_length):
        problems.append(f"{path}: 期望非空字符串，得到 {value!r}")


def _check_protocol(problems: list[str], p: Any, path: str) -> None:
    if not isinstance(p, dict):
        problems.append(f"{path}: 期望对象")
        return
    allowed = {"strategy", "n_folds", "time_col", "val_size_weeks", "group_col"}
    for k in p:
        if k not in allowed:
            problems.append(f"{path}.{k}: 未知字段")
    if p.get("strategy") not in STRATEGIES:
        problems.append(f"{path}.strategy: {p.get('strategy')!r} 不在 {STRATEGIES}")
    n_folds = p.get("n_folds")
    if not _is_int(n_folds) or n_folds < 1:
        problems.append(f"{path}.n_folds: 期望 >=1 的整数，得到 {n_folds!r}")
    if "time_col" in p and not isinstance(p["time_col"], str):
        problems.append(f"{path}.time_col: 期望字符串")
    if "val_size_weeks" in p and not _is_int(p["val_size_weeks"]):
        problems.append(f"{path}.val_size_weeks: 期望整数")
    if "group_col" in p and not isinstance(p["group_col"], str):
        problems.append(f"{path}.group_col: 期望字符串")


def _fallback_validate(ir: Any) -> None:
    """Minimal structural check used when jsonschema is unavailable.

    Mirrors the schema's shape: required fields, types, closed key sets and
    hash/revision patterns. Semantic legality (budget > 0, direction enum,
    temporal consistency, ...) is intentionally left to the hard gates.
    """
    problems: list[str] = []
    if not isinstance(ir, dict):
        raise IRSchemaError("ExperimentIR 必须是 JSON 对象")

    required = {
        "schema_version", "experiment_id", "task_ref", "world_state_ref",
        "objective", "dataset_snapshot_ref", "feature_set_ref",
        "validation_protocol_ref", "metric_definition_ref", "baseline_ref",
        "candidates", "hard_constraints", "budget", "expected_artifacts",
        "authorization", "source_ref", "created_at", "content_hash", "revision",
    }
    allowed = required | {"hypothesis_ref", "soft_preferences", "supersedes"}
    for k in ir:
        if k not in allowed:
            problems.append(f"{k}: 未知顶层字段")
    for k in sorted(required - ir.keys()):
        problems.append(f"{k}: 缺少必填字段")

    if ir.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version: 期望 {SCHEMA_VERSION!r}，得到 {ir.get('schema_version')!r}")
    for k in ("experiment_id", "task_ref", "world_state_ref", "objective",
              "feature_set_ref", "source_ref"):
        _check_str(problems, k, ir.get(k))
    if "hypothesis_ref" in ir:
        _check_str(problems, "hypothesis_ref", ir["hypothesis_ref"])
    for k in ("content_hash", "supersedes"):
        if k in ir and not isinstance(ir[k], str) or (k in ir and not _SHA256_RE.match(ir[k])):
            problems.append(f"{k}: 期望 64 位小写 hex sha256")
    rev = ir.get("revision")
    if not _is_int(rev) or rev < 1:
        problems.append(f"revision: 期望 >=1 的整数，得到 {rev!r}")

    ds = ir.get("dataset_snapshot_ref")
    if not isinstance(ds, dict):
        problems.append("dataset_snapshot_ref: 期望对象")
    else:
        for k in ds:
            if k not in {"uri", "sha256", "rows", "as_of", "label_cutoff"}:
                problems.append(f"dataset_snapshot_ref.{k}: 未知字段")
        _check_str(problems, "dataset_snapshot_ref.uri", ds.get("uri"))
        if not isinstance(ds.get("sha256"), str) or not _SHA256_RE.match(ds.get("sha256", "")):
            problems.append("dataset_snapshot_ref.sha256: 期望 64 位小写 hex sha256")
        if not _is_int(ds.get("rows")) or ds.get("rows") < 0:
            problems.append("dataset_snapshot_ref.rows: 期望 >=0 的整数")

    proto = ir.get("validation_protocol_ref")
    _check_protocol(problems, proto, "validation_protocol_ref")

    metric = ir.get("metric_definition_ref")
    if not isinstance(metric, dict):
        problems.append("metric_definition_ref: 期望对象")
    else:
        for k in metric:
            if k not in {"name", "direction"}:
                problems.append(f"metric_definition_ref.{k}: 未知字段")
        _check_str(problems, "metric_definition_ref.name", metric.get("name"))
        if not isinstance(metric.get("direction"), str):
            problems.append("metric_definition_ref.direction: 期望字符串")

    baseline = ir.get("baseline_ref")
    if not isinstance(baseline, dict):
        problems.append("baseline_ref: 期望对象")
    else:
        for k in baseline:
            if k not in {"baseline_id", "protocol_ref", "score"}:
                problems.append(f"baseline_ref.{k}: 未知字段")
        _check_str(problems, "baseline_ref.baseline_id", baseline.get("baseline_id"))
        _check_protocol(problems, baseline.get("protocol_ref"), "baseline_ref.protocol_ref")
        if not _is_number(baseline.get("score")):
            problems.append("baseline_ref.score: 期望数值")

    budget = ir.get("budget")
    if not isinstance(budget, dict):
        problems.append("budget: 期望对象")
    else:
        for k in budget:
            if k not in {"max_train_hours", "max_runs"}:
                problems.append(f"budget.{k}: 未知字段")
        for k in ("max_train_hours", "max_runs"):
            if not _is_number(budget.get(k)):
                problems.append(f"budget.{k}: 期望数值")

    auth = ir.get("authorization")
    if not isinstance(auth, dict):
        problems.append("authorization: 期望对象")
    else:
        for k in auth:
            if k not in {"authorized_by", "authorization_ref", "granted_at"}:
                problems.append(f"authorization.{k}: 未知字段")
        for k in ("authorized_by", "authorization_ref", "granted_at"):
            if k not in auth:
                problems.append(f"authorization.{k}: 缺少必填字段")
            elif not isinstance(auth[k], str):
                problems.append(f"authorization.{k}: 期望字符串")

    for k, item_required in (
        ("hard_constraints", {"type", "description", "check"}),
        ("candidates", {"candidate_id", "family", "params"}),
    ):
        items = ir.get(k)
        if not isinstance(items, list):
            problems.append(f"{k}: 期望数组")
            continue
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                problems.append(f"{k}[{i}]: 期望对象")
                continue
            for f in item:
                if f not in item_required:
                    problems.append(f"{k}[{i}].{f}: 未知字段")
            for f in sorted(item_required):
                if f not in item:
                    problems.append(f"{k}[{i}].{f}: 缺少必填字段")
                elif f != "params" and not (isinstance(item[f], str) and item[f]):
                    problems.append(f"{k}[{i}].{f}: 期望非空字符串")
            if k == "candidates" and "params" in item and not isinstance(item["params"], dict):
                problems.append(f"{k}[{i}].params: 期望对象")

    arts = ir.get("expected_artifacts")
    if not isinstance(arts, list) or any(not isinstance(a, str) or not a for a in arts):
        problems.append("expected_artifacts: 期望非空字符串数组")
    if "soft_preferences" in ir and not isinstance(ir["soft_preferences"], list):
        problems.append("soft_preferences: 期望数组")

    _fail(problems)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_ir(path: str | Path) -> dict:
    """Load, schema-validate and hash-check an ExperimentIR document.

    Raises IRSchemaError for schema-level illegality and
    ContentHashMismatch when the stored hash disagrees with the recomputed
    one (document mutated after compilation). Returns the parsed dict.
    """
    with open(path, encoding="utf-8") as f:
        ir = json.load(f)
    schema_validate(ir)
    stored = ir.get("content_hash")
    recomputed = compute_content_hash(ir)
    if not isinstance(stored, str) or stored.lower() != recomputed:
        raise ContentHashMismatch(
            f"content_hash 不一致: 文档记录 {stored!r}, 重算 {recomputed}; "
            "文档在编译后被改动，修正必须产生新 revision 并以 supersedes 指向前版"
        )
    return ir


# ---------------------------------------------------------------------------
# Hard gates
# ---------------------------------------------------------------------------

def _section(ir: dict, key: str) -> dict:
    value = ir.get(key)
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _g1_temporal_availability(ir: dict) -> GateResult:
    proto = _section(ir, "validation_protocol_ref")
    dataset = _section(ir, "dataset_snapshot_ref")
    strategy = proto.get("strategy")
    if strategy != "time_based":
        return GateResult(G1_TEMPORAL_AVAILABILITY, "not_applicable",
                          f"策略 {strategy!r} 非时序，无时间可用性约束")
    missing = []
    if not _text(proto.get("time_col")):
        missing.append("validation_protocol_ref.time_col")
    if proto.get("val_size_weeks") is None:
        missing.append("validation_protocol_ref.val_size_weeks")
    if dataset.get("label_cutoff") is None:
        missing.append("dataset_snapshot_ref.label_cutoff")
    if missing:
        return GateResult(G1_TEMPORAL_AVAILABILITY, "fail",
                          "time_based 策略缺少必需要素: " + ", ".join(missing))
    cutoff = _parse_iso(dataset["label_cutoff"])
    as_of = _parse_iso(dataset["as_of"])
    if cutoff is None or as_of is None:
        return GateResult(G1_TEMPORAL_AVAILABILITY, "fail",
                          "label_cutoff/as_of 无法解析为 ISO-8601 时间戳")
    try:
        temporal_ok = cutoff <= as_of
    except TypeError:  # naive vs aware datetime
        temporal_ok = False
    if not temporal_ok:
        return GateResult(G1_TEMPORAL_AVAILABILITY, "fail",
                          f"label_cutoff({dataset['label_cutoff']}) 晚于 "
                          f"as_of({dataset['as_of']})，存在未来标签泄漏")
    return GateResult(G1_TEMPORAL_AVAILABILITY, "pass",
                      "时序要素齐备且 label_cutoff <= as_of")


def _g2_split_separation(ir: dict) -> GateResult:
    proto = _section(ir, "validation_protocol_ref")
    strategy = proto.get("strategy")
    if strategy not in STRATEGIES:
        return GateResult(G2_SPLIT_SEPARATION, "fail",
                          f"未知验证策略 {strategy!r}，合法值: {STRATEGIES}")
    if strategy == "time_based":
        val_size_weeks = proto.get("val_size_weeks")
        if not _is_number(val_size_weeks) or val_size_weeks <= 0:
            return GateResult(G2_SPLIT_SEPARATION, "fail",
                              f"time_based 策略要求 val_size_weeks > 0，得到 {val_size_weeks!r}")
        return GateResult(G2_SPLIT_SEPARATION, "pass",
                          f"time_based 前向切分 val_size_weeks={val_size_weeks}")
    if strategy == "group":
        if not _text(proto.get("group_col")):
            return GateResult(G2_SPLIT_SEPARATION, "fail",
                              "group 策略要求非空 group_col")
        return GateResult(G2_SPLIT_SEPARATION, "pass",
                          f"group 切分按 group_col={proto['group_col']!r} 隔离")
    return GateResult(G2_SPLIT_SEPARATION, "pass",
                      f"策略 {strategy} 无额外分离约束")


def _g3_metric_definition(ir: dict) -> GateResult:
    metric = _section(ir, "metric_definition_ref")
    problems = []
    if not _text(metric.get("name")):
        problems.append("metric_definition_ref.name 为空")
    direction = metric.get("direction")
    if direction not in DIRECTIONS:
        problems.append(f"direction {direction!r} 非法，合法值: maximize|minimize")
    if problems:
        return GateResult(G3_METRIC_DEFINITION, "fail", "; ".join(problems))
    return GateResult(G3_METRIC_DEFINITION, "pass",
                      f"指标 {metric['name']} direction={direction}")


def _g4_budget_bounds(ir: dict) -> GateResult:
    budget = _section(ir, "budget")
    problems = []
    for key in ("max_train_hours", "max_runs"):
        value = budget.get(key)
        if not _is_number(value) or value <= 0:
            problems.append(f"budget.{key} 必须为正数，得到 {value!r}")
    if problems:
        return GateResult(G4_BUDGET_BOUNDS, "fail", "; ".join(problems))
    return GateResult(G4_BUDGET_BOUNDS, "pass",
                      f"预算合规: max_train_hours={budget['max_train_hours']}, "
                      f"max_runs={budget['max_runs']}")


def _g5_baseline_same_protocol(ir: dict) -> GateResult:
    baseline = _section(ir, "baseline_ref")
    top_protocol = ir.get("validation_protocol_ref")
    baseline_protocol = baseline.get("protocol_ref")
    if baseline_protocol == top_protocol:
        return GateResult(G5_BASELINE_SAME_PROTOCOL, "pass",
                          "基线与主实验使用全等验证协议")
    return GateResult(G5_BASELINE_SAME_PROTOCOL, "fail",
                      "baseline_ref.protocol_ref 与顶层 validation_protocol_ref "
                      "不全等，基线与候选不可比")


def _g6_run_authorization(ir: dict) -> GateResult:
    auth = _section(ir, "authorization")
    missing = []
    if not _text(auth.get("authorized_by")):
        missing.append("authorization.authorized_by")
    if not _text(auth.get("authorization_ref")):
        missing.append("authorization.authorization_ref")
    if missing:
        return GateResult(G6_RUN_AUTHORIZATION, "fail",
                          "缺少执行授权: " + ", ".join(missing))
    return GateResult(G6_RUN_AUTHORIZATION, "pass",
                      f"授权人 {auth['authorized_by']} (ref={auth['authorization_ref']})")


_GATE_FUNCS = {
    G1_TEMPORAL_AVAILABILITY: _g1_temporal_availability,
    G2_SPLIT_SEPARATION: _g2_split_separation,
    G3_METRIC_DEFINITION: _g3_metric_definition,
    G4_BUDGET_BOUNDS: _g4_budget_bounds,
    G5_BASELINE_SAME_PROTOCOL: _g5_baseline_same_protocol,
    G6_RUN_AUTHORIZATION: _g6_run_authorization,
}


def run_gates(ir: dict) -> list[GateResult]:
    """Run all six hard gates in fixed contract order."""
    return [_GATE_FUNCS[gate_id](ir) for gate_id in _GATE_ORDER]


def authorize_execution(ir: dict) -> AuthorizationVerdict:
    """Aggregate gate results into an allow/block verdict.

    Any gate with status "fail" blocks execution ("not_applicable" never
    blocks). The remedy is a new revision with ``supersedes`` pointing at
    this document's content_hash — never an in-place edit.
    """
    gates = run_gates(ir)
    failed = [g for g in gates if g.status == "fail"]
    return AuthorizationVerdict(
        ok=not failed,
        failed_gates=tuple(g.gate_id for g in failed),
        reasons=tuple(f"{g.gate_id}: {g.reason}" for g in failed),
        gates=tuple(gates),
    )
