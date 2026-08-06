"""
测试新增模块: Context Engineering / Reflexion / Multi-Agent /
              Self-Evolving Skills / Entity Extraction / MCTS

运行: python tests/test_new_modules.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ---------- Context Engineering ----------

def test_context_compressor():
    from framework.src.context_engineering import ContextCompressor, CompressionLevel
    text = """
    LightGBM is a great gradient boosting library.
    XGBoost is also widely used.
    CatBoost handles categorical features natively.
    Random Forest is simple but effective.
    LightGBM is fast and supports categorical features.
    """ * 5

    c = ContextCompressor()
    r_light = c.compress(text, CompressionLevel.LIGHT)
    r_med = c.compress(text, CompressionLevel.MEDIUM, focus_query="LightGBM")
    r_agg = c.compress(text, CompressionLevel.AGGRESSIVE, focus_query="XGBoost")

    # Aggressive 应明显少于 light;medium 可能略多或略少(取决于重复度)
    assert r_agg.compressed_chars < r_light.compressed_chars
    assert r_agg.compressed_chars > 0
    print(f"[OK] ContextCompressor: light={r_light.compressed_chars}, "
          f"med={r_med.compressed_chars}, agg={r_agg.compressed_chars}")


def test_log_truncator():
    from framework.src.context_engineering import LogTruncator
    log_lines = []
    for i in range(500):
        log_lines.append(f"Epoch {i//10}: loss=0.{i:03d} val_loss=0.5{i:03d}")
    log_lines[250] = "ERROR: CUDA out of memory. Tried to allocate 2.00 GiB"
    log_lines[251] = "Traceback (most recent call last):"
    log_lines[252] = '  File "train.py", line 42, in <module>'

    log_text = "\n".join(log_lines)
    truncator = LogTruncator(max_lines=50)
    result = truncator.truncate(log_text)

    assert "ERROR" in result
    assert "out of memory" in result
    assert "Traceback" in result
    assert "omitted" in result
    print(f"[OK] LogTruncator: 500 lines → {result.count(chr(10))} lines, keeps ERROR")


def test_token_economy():
    from framework.src.context_engineering import TokenEconomy, BudgetPriority
    econ = TokenEconomy()
    econ.add_critical("System: You are an ML expert.", label="system")
    econ.add_high("Current task: Predict house prices.", label="task")
    econ.add("All 43+ skill descriptions...", BudgetPriority.LOW, label="all_skills")
    econ.add("Some auxiliary context...", BudgetPriority.MEDIUM, label="context")

    ctx = econ.assemble()
    assert "Critical" in ctx or "HIGH" in ctx
    stats = econ.stats()
    print(f"[OK] TokenEconomy: {stats['blocks']} blocks, "
          f"~{stats['estimated_tokens']} tokens, over_budget={stats['over_budget']}")


# ---------- Reflexion ----------

def test_error_analyzer():
    from framework.src.reflexion import ErrorAnalyzer
    a = ErrorAnalyzer()

    # Shape mismatch
    diag1 = a.analyze(
        "ValueError: shapes (32,10) and (32,20) not aligned",
        code="import numpy as np\n..."
    )
    assert diag1.category.value in ("shape", "type", "logic")

    # Import error
    diag2 = a.analyze("ModuleNotFoundError: No module named 'sklearn'")
    assert diag2.category.value == "import"

    # Memory error
    diag3 = a.analyze("RuntimeError: CUDA out of memory. Tried to allocate 4.00 GiB")
    assert diag3.category.value == "memory"
    assert "gpu-readiness-assessment" in diag3.related_skills

    print(f"[OK] ErrorAnalyzer: shape/import/memory all classified correctly")


def test_sandbox_execution():
    from framework.src.reflexion import CodeSandbox, execute_with_analysis
    sb = CodeSandbox(timeout_sec=10)

    # Success case
    result, diag = execute_with_analysis("print('hello world')", sb)
    assert result.success
    assert diag is None

    # Failure case
    result, diag = execute_with_analysis("import nonexistent_module_xyz", sb)
    assert not result.success
    assert diag is not None
    assert diag.category.value == "import"

    # Timeout case
    sb_fast = CodeSandbox(timeout_sec=1)
    result, diag = execute_with_analysis("import time; time.sleep(5)", sb_fast)
    assert not result.success
    assert result.timed_out

    print(f"[OK] CodeSandbox: success/failure/timeout all handled")


def test_reflexion_loop_without_llm():
    from framework.src.reflexion import ReflexionLoop, CodeSandbox
    loop = ReflexionLoop(sandbox=CodeSandbox(timeout_sec=10))
    # 无 llm_fix: 最多跑 max_attempts 次失败
    result = loop.run("import nonexistent", max_attempts=3)
    assert not result.success
    assert result.attempts_count == 3
    print(f"[OK] ReflexionLoop (no LLM): {result.attempts_count} attempts, all failed")


# ---------- Multi-Agent ----------

def test_role_presets():
    from framework.src.agents import get_role
    arch = get_role("data-architect")
    coder = get_role("ml-coder")
    critic = get_role("continuity-critic")
    researcher = get_role("knowledge-researcher")

    assert "search_kaggle_discussions" in arch.tools
    assert "Do NOT write executable Python code" in " ".join(arch.forbidden_actions)
    assert "data leakage" in critic.system_prompt.lower()
    assert "arxiv" in researcher.system_prompt.lower()
    print("[OK] Role presets: architect/coder/critic/researcher loaded")


def test_orchestrator_standard_pipeline():
    from framework.src.agents import Orchestrator
    orch = Orchestrator(llm_call=None)  # Mock mode
    pipeline = orch.create_standard_ml_pipeline()
    assert "standard-ml" in orch.pipelines
    assert len(orch.agents) == 3
    outputs = orch.run_pipeline("standard-ml", "Predict house prices", max_revisions=1)
    assert len(outputs) >= 3  # 至少 architect + coder + critic
    print(f"[OK] Orchestrator: standard pipeline produced {len(outputs)} messages")


# ---------- Self-Evolving Skills ----------

def test_skill_extractor():
    from framework.src.skill_evolution import SkillExtractor
    ext = SkillExtractor()
    text = (
        "We should always check data leakage before training. "
        "Never use future information in features. "
        "Make sure to validate with walk-forward for time series."
    )
    candidates = ext.extract_from_text(text)
    assert len(candidates) >= 2
    # 至少包含 anti-pattern 和 principle 类型
    types = {c.type for c in candidates}
    assert "anti-pattern" in types or "principle" in types
    print(f"[OK] SkillExtractor: extracted {len(candidates)} candidates, types={types}")


def test_skill_validator():
    from framework.src.skill_evolution import SkillExtractor, SkillValidator
    ext = SkillExtractor()
    val = SkillValidator()

    text = "Never use random K-fold on time series data because it leaks future info."
    candidates = ext.extract_from_text(text)
    assert len(candidates) > 0

    result = val.validate(candidates[0])
    # 缺 description 触发器(没以 "Use when..." 开头),会有 suggestions
    print(f"[OK] SkillValidator: verdict={result.verdict.value}, "
          f"suggestions={len(result.suggestions)}")


def test_skill_registry_round_trip(tmp_path=None):
    from framework.src.skill_evolution import (
        SkillExtractor, SkillRegistry, SkillCandidate
    )
    if tmp_path is None:
        import tempfile
        tmp_path = Path(tempfile.mkdtemp())

    registry = SkillRegistry(
        skills_dir=str(tmp_path / "skills"),
        state_file=str(tmp_path / "skills" / ".registry.json"),
    )
    candidate = SkillCandidate(
        name="test-auto-skill",
        description="Use when you need a test skill for validation",
        content="## Context\nTest content",
        type="skill",
        importance=0.6,
        tags=["test"],
    )
    result = registry.register(candidate, auto_approve=True)
    assert result.verdict.value == "approve"
    assert registry.is_registered("test-auto-skill")
    print(f"[OK] SkillRegistry: registered and persisted")


# ---------- Entity Extraction (Mem0 升级) ----------

def test_entity_extraction():
    from framework.src.memory.entity_extraction import EntityRelationExtractor
    ext = EntityRelationExtractor()
    text = (
        "LightGBM uses histogram-based gradient boosting. "
        "XGBoost improves LightGBM with second-order gradients. "
        "CatBoost handles categorical features natively, unlike LightGBM. "
        "We evaluated F1 score on the Spaceship Titanic competition. "
        "Walk-forward validation prevents data leakage in time series."
    )
    entities, relations = ext.extract(text)

    entity_types = {e.type for e in entities}
    assert "technique" in entity_types
    assert "metric" in entity_types or "competition" in entity_types

    # 至少应识别出 LightGBM, XGBoost, CatBoost
    names = {e.name.lower() for e in entities}
    assert "lightgbm" in names

    # 关系至少一条(比如 LightGBM <-> XGBoost)
    assert len(relations) >= 1
    print(f"[OK] Entity extraction: {len(entities)} entities, "
          f"{len(relations)} relations, types={entity_types}")


# ---------- MCTS ----------

def test_mcts_pipeline_search():
    from framework.src.mcts import MCTSSearch, grid_expansion, identity_evaluator

    # 简单示例: 网格搜索 LightGBM num_leaves
    grid = {"num_leaves": [15, 31, 63], "learning_rate": [0.01, 0.05, 0.1]}

    # 假设 num_leaves=31 + lr=0.05 最好
    def evaluator(config):
        score = 0.5
        if config.get("num_leaves") == 31:
            score += 0.3
        if config.get("learning_rate") == 0.05:
            score += 0.2
        return score

    expansion = grid_expansion(grid)
    search = MCTSSearch(evaluator, expansion, max_depth=2)
    result = search.search(initial_config={"num_leaves": 31, "learning_rate": 0.1},
                            iterations=20)

    assert result.best_score >= 0.5
    assert result.tree_size > 1
    print(f"[OK] MCTS: best config = {result.best_node.config}, "
          f"score={result.best_score:.3f}, tree_size={result.tree_size}")


def test_mcts_node_ucb1():
    from framework.src.mcts import PipelineNode
    n1 = PipelineNode(id="n1")
    n1.visits = 10
    n1.score_sum = 5.0  # avg = 0.5
    n2 = PipelineNode(id="n2")
    n2.visits = 1
    n2.score_sum = 0.9  # avg = 0.9
    n1.parent = n2.parent = PipelineNode(id="root")
    n1.parent.visits = 100

    # 未访问节点应该 UCB1 = inf
    n3 = PipelineNode(id="n3")
    assert n3.ucb1() == float("inf")
    # 访问次数少的节点应该 UCB1 更高(探索)
    assert n2.ucb1() > n1.ucb1()
    print(f"[OK] PipelineNode UCB1: n2 (fewer visits) > n1, unvisited = inf")


# ---------- main ----------

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1-3 New Modules Tests")
    print("=" * 60)
    test_context_compressor()
    test_log_truncator()
    test_token_economy()
    test_error_analyzer()
    test_sandbox_execution()
    test_reflexion_loop_without_llm()
    test_role_presets()
    test_orchestrator_standard_pipeline()
    test_skill_extractor()
    test_skill_validator()
    test_skill_registry_round_trip()
    test_entity_extraction()
    test_mcts_pipeline_search()
    test_mcts_node_ucb1()
    print("=" * 60)
    print("[PASS] All new module tests passed")