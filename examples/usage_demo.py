"""
使用示例 — Memory Hierarchy + Knowledge Layer + MCP Server

运行:
    python examples/usage_demo.py

需要:
    - framework/ 已安装(pip install -e .)
    - 网络连接(arxiv + Kaggle 部分会真实调用)
    - 已有 docs/ml-agent-memory/ 内容(可选)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.src.memory import MemoryHierarchy, MemoryItem
from framework.src.knowledge import KnowledgeAggregator


def demo_memory_hierarchy():
    """P0: 分层记忆"""
    print("=" * 60)
    print("Demo 1: Memory Hierarchy")
    print("=" * 60)

    mem = MemoryHierarchy(okf_dir=str(ROOT / "docs" / "ml-agent-memory"))
    indexed = mem.bootstrap(extra_dirs=[str(ROOT / "skills" / "examples")])
    print(f"Indexed {indexed} items from vault + skills")

    # Recall
    items = mem.recall("tabular feature engineering", k=3)
    print(f"\nRecalled {len(items)} items:")
    for item in items:
        print(f"  [{item.type}] {item.id} (imp={item.importance:.2f})")

    # Snapshot
    print("\n--- Working Memory Snapshot (first 500 chars) ---")
    print(mem.snapshot(max_chars=500))

    # Remember
    print("\n--- Remembering new finding ---")
    mem.remember(MemoryItem(
        id="demo-finding-2026-08-05",
        content="Demo: TabPFN works well for small tabular datasets (<10K rows).",
        type="experiment",
        importance=0.7,
        tags=["tabular", "tabpfn", "demo"],
    ), persist=True)
    print(f"Stats: {mem.stats()}")


def demo_live_knowledge():
    """P1: 实时论文 + Kaggle 搜索"""
    print("\n" + "=" * 60)
    print("Demo 2: Live Knowledge Search")
    print("=" * 60)

    agg = KnowledgeAggregator()

    # Try a small search
    try:
        # Semantic Scholar — fast, low rate limit
        print("\n[Semantic Scholar] Searching 'TabPFN tabular'...")
        ss = agg.scholar
        papers = ss.search("TabPFN tabular prior-fitted network", max_results=3)
        for p in papers:
            print(f"  [{p.year}] {p.title}")
            print(f"    Citations: {p.citation_count} | TLDR: {p.tldr or '(no TLDR)'}")
    except Exception as e:
        print(f"  [SK] {e}")

    try:
        # arxiv — needs 3s rate limit
        print("\n[arxiv] Searching 'time series foundation model'...")
        ax = agg.arxiv
        papers = ax.search("time series foundation model", max_results=3, years_back=2)
        for p in papers:
            print(f"  [{p.year}] {p.title}")
            print(f"    Authors: {', '.join(p.authors[:2])}")
            print(f"    PDF: {p.pdf_url}")
    except Exception as e:
        print(f"  [SK] {e}")

    try:
        # Kaggle
        print("\n[Kaggle] Searching 'CatBoost ensemble'...")
        kf = agg.kaggle
        discs = kf.search_global("CatBoost ensemble", max_results=3)
        for d in discs:
            print(f"  [{d.votes}↑ {d.comments_count}💬] {d.title}")
    except Exception as e:
        print(f"  [SK] {e}")


def demo_aggregator():
    """P1: 统一报告"""
    print("\n" + "=" * 60)
    print("Demo 3: Unified Knowledge Report")
    print("=" * 60)

    agg = KnowledgeAggregator(okf_dir=str(ROOT / "docs" / "ml-agent-memory"))
    report = agg.search_all("walk-forward validation time series", max_per_source=3)
    print(f"Query: '{report.query}'")
    print(f"Total: {report.total_count} (scholar: {len(report.scholar_papers)}, "
          f"arxiv: {len(report.arxiv_papers)}, kaggle: {len(report.kaggle_discs)})")

    # Save to vault
    if report.total_count > 0:
        path = agg.write_report_to_vault(report)
        print(f"\nSaved to: {path}")


def demo_recall_vs_search():
    """对比:memory recall vs live search"""
    print("\n" + "=" * 60)
    print("Demo 4: Recall (vault) vs Search (live)")
    print("=" * 60)

    mem = MemoryHierarchy(okf_dir=str(ROOT / "docs" / "ml-agent-memory"))
    mem.bootstrap(extra_dirs=[str(ROOT / "skills" / "examples")])

    # Recall from vault
    print("\nRecall from vault (instant, may be stale):")
    items = mem.recall("feature engineering", k=3)
    for item in items:
        print(f"  [{item.type}] {item.id}")

    # Live search
    print("\nLive search from arxiv (slower, always current):")
    try:
        agg = KnowledgeAggregator()
        papers = agg.arxiv.search("feature engineering machine learning", max_results=3, years_back=3)
        for p in papers:
            print(f"  [{p.year}] {p.title}")
    except Exception as e:
        print(f"  [SK] {e}")


if __name__ == "__main__":
    demo_memory_hierarchy()
    demo_live_knowledge()
    demo_aggregator()
    demo_recall_vs_search()
    print("\n" + "=" * 60)
    print("All demos complete")
    print("=" * 60)