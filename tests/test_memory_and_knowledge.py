"""
测试分层记忆和知识聚合器(离线部分,避开网络)

运行: python -m pytest tests/test_memory_and_knowledge.py -v
或:   python tests/test_memory_and_knowledge.py
"""
import sys
import tempfile
from pathlib import Path

# Windows 控制台编码兼容
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 让 import 能找到 framework/src
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.src.memory import (
    MemoryItem, WorkingMemory, ArchivalStore, MemoryHierarchy,
)
from framework.src.memory.recall import multi_strategy_recall, RecallConfig


# ---------- WorkingMemory 测试 ----------

def test_working_memory_lru_eviction():
    wm = WorkingMemory(max_items=3)
    for i in range(5):
        wm.add(MemoryItem(id=f"item-{i}", content=f"content {i}"))
    assert len(wm.items) == 3, f"Expected 3 items, got {len(wm.items)}"
    # LRU 应该淘汰 item-0 和 item-1
    ids = {i.id for i in wm.items}
    assert "item-4" in ids and "item-3" in ids and "item-2" in ids
    print("✓ WorkingMemory LRU eviction works")


def test_working_memory_dedup():
    wm = WorkingMemory()
    wm.add(MemoryItem(id="x", content="first"))
    wm.add(MemoryItem(id="x", content="second"))
    assert len(wm.items) == 1
    assert wm.items[0].content == "second"
    print("✓ WorkingMemory dedup works")


# ---------- ArchivalStore 测试 ----------

def test_archival_search_keyword():
    store = ArchivalStore(okf_dir="nonexistent")
    store.add(MemoryItem(id="a", content="time series forecasting with transformer", importance=0.8))
    store.add(MemoryItem(id="b", content="catboost gradient boosting tuning", importance=0.6))
    store.add(MemoryItem(id="c", content="transformer attention mechanism explained", importance=0.7))

    results = store.search("transformer", k=3)
    assert len(results) >= 2
    assert "transformer" in results[0].content.lower()
    print(f"✓ ArchivalStore keyword search returns {len(results)} hits")


def test_archival_search_type_filter():
    store = ArchivalStore(okf_dir="nonexistent")
    store.add(MemoryItem(id="s1", content="use catboost first", type="skill", importance=0.8))
    store.add(MemoryItem(id="p1", content="catboost is good", type="principle", importance=0.9))
    results = store.search("catboost", type_filter="skill")
    assert all(r.type == "skill" for r in results)
    print("✓ ArchivalStore type filter works")


def test_archival_md_indexing():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "skills").mkdir()
        (d / "skills" / "catboost.md").write_text(
            "---\ntype: skill\nimportance: 0.8\ntags: tabular,boosting\n---\n"
            "# CatBoost First\n\nFor tabular data, prefer CatBoost.",
            encoding="utf-8",
        )
        store = ArchivalStore(okf_dir=str(d))
        n = store.index_directory()
        assert n >= 1
        assert any("catboost" in i.id.lower() for i in store.items.values())
        print(f"✓ ArchivalStore indexed {n} markdown files")


# ---------- multi_strategy_recall 测试 ----------

def test_multi_strategy_recall():
    store = ArchivalStore(okf_dir="nonexistent")
    store.add(MemoryItem(id="fresh", content="feature engineering saturation", importance=0.9,
                          last_accessed="2026-08-05T00:00:00"))
    store.add(MemoryItem(id="old", content="feature engineering old stuff", importance=0.9,
                          last_accessed="2024-01-01T00:00:00"))

    cfg = RecallConfig(use_recency=True, recency_decay_days=30.0)
    results = multi_strategy_recall(store, "feature engineering", cfg)
    assert results[0].id == "fresh", f"Expected 'fresh' first, got {results[0].id}"
    print("✓ Multi-strategy recall prefers fresh items")


# ---------- MemoryHierarchy 测试 ----------

def test_hierarchy_recall_promotes_to_walking():
    h = MemoryHierarchy(okf_dir="nonexistent")
    h.archival.add(MemoryItem(id="x", content="test", importance=0.8))
    hits = h.recall("test")
    assert len(hits) == 1
    assert any(i.id == "x" for i in h.working.items)
    print("✓ MemoryHierarchy auto-promotes recalled items to working")


def test_hierarchy_remember_promotion():
    h = MemoryHierarchy(okf_dir="nonexistent")
    h.remember(MemoryItem(id="important", content="x", importance=0.9))
    h.remember(MemoryItem(id="trivial", content="y", importance=0.3))
    # important 应该进 archival
    assert "important" in h.archival.items
    assert "trivial" not in h.archival.items
    print("✓ MemoryHierarchy importance-based promotion works")


# ---------- aggregator 测试(只测基础结构,跳过网络) ----------

def test_knowledge_report_empty():
    from framework.src.knowledge.aggregator import KnowledgeReport
    r = KnowledgeReport(query="test")
    assert r.total_count == 0
    md = r.to_markdown()
    assert "test" in md
    print("✓ KnowledgeReport empty rendering works")


def test_knowledge_report_with_data():
    from framework.src.knowledge.aggregator import KnowledgeReport
    from framework.src.knowledge import ArxivPaper, KaggleDiscussion, SSPaper
    r = KnowledgeReport(query="test")
    r.arxiv_papers = [ArxivPaper(
        arxiv_id="2401.00001", title="Test", authors=["A"], abstract="x",
        categories=["cs.LG"], published="2024-01-01", updated="2024-01-01",
        pdf_url="x", abs_url="x", year=2024,
    )]
    r.kaggle_discs = [KaggleDiscussion(title="D", url="u", competition="c")]
    r.scholar_papers = [SSPaper(
        paper_id="x", title="S", abstract="x", year=2024, venue="",
        citation_count=10,
        reference_count=20, influential_citation_count=5,
        authors=["A"], url="u",
    )]
    md = r.to_markdown()
    assert "Test" in md and "D" in md and "S" in md
    print(f"✓ KnowledgeReport combined rendering works ({r.total_count} items)")


# ---------- main ----------

if __name__ == "__main__":
    print("=" * 60)
    print("Memory & Knowledge Tests (offline)")
    print("=" * 60)
    test_working_memory_lru_eviction()
    test_working_memory_dedup()
    test_archival_search_keyword()
    test_archival_search_type_filter()
    test_archival_md_indexing()
    test_multi_strategy_recall()
    test_hierarchy_recall_promotes_to_walking()
    test_hierarchy_remember_promotion()
    test_knowledge_report_empty()
    test_knowledge_report_with_data()
    print("=" * 60)
    print("✅ All offline tests passed")