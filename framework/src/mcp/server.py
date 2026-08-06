"""
MCP Server — 把 memory + knowledge 能力暴露为 MCP tools

启动方式:
    python -m framework.src.mcp.server

依赖:
    pip install mcp  (Anthropic MCP SDK)

工具清单(agent 可直接调用):
    论文/论坛:
      - search_papers(query)              → arxiv + Semantic Scholar 合并
      - search_kaggle_discussions(query)  → Kaggle forums
      - search_arxiv_recent(category, days)
      - get_paper_citations(arxiv_id)
    记忆:
      - recall_skill(query, k)            → 从 vault 召回相关 skill
      - remember_experience(content, type, importance)
      - snapshot_memory()
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# 让 import 能找到 framework/src
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[3]  # framework/src/mcp/server.py → repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    from mcp.server.stdio import stdio_server
except ImportError:
    print("MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
    raise

from framework.src.knowledge import (
    ArxivSearch, KaggleForum, SemanticScholar,
    KnowledgeAggregator,
)
from framework.src.memory import MemoryHierarchy, recall_skills_for_query


# ---------- Server 实例 ----------

app = Server("cultivating-ml-agent")

# 全局单例(每次请求复用,避免重复初始化)
_memory: MemoryHierarchy | None = None
_aggregator: KnowledgeAggregator | None = None


def get_memory() -> MemoryHierarchy:
    global _memory
    if _memory is None:
        _memory = MemoryHierarchy(okf_dir="docs/ml-agent-memory")
        _memory.bootstrap(extra_dirs=["skills/examples", "docs"])
    return _memory


def get_aggregator() -> KnowledgeAggregator:
    global _aggregator
    if _aggregator is None:
        _aggregator = KnowledgeAggregator()
    return _aggregator


# ---------- Tool 定义 ----------

TOOLS = [
    Tool(
        name="search_papers",
        description=(
            "Search academic papers from arxiv + Semantic Scholar. "
            "Returns combined results with titles, authors, abstracts, TLDR. "
            "Use when starting a new ML technique exploration, or when current "
            "skills don't cover a topic."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (e.g. 'time series forecasting transformer')"},
                "max_results": {"type": "integer", "default": 10, "description": "Max papers to return"},
                "years_back": {"type": "integer", "default": 5, "description": "Limit to recent N years"},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="search_kaggle_discussions",
        description=(
            "Search Kaggle forum discussions globally or by competition. "
            "Returns top-voted discussions with snippets. "
            "Use when looking for top-solution tricks, hyperparameter advice, "
            "or CV-LB gap discussions."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "competition_slug": {"type": "string", "default": "", "description": "Optional Kaggle competition slug (e.g. 'spaceship-titanic')"},
                "max_results": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="search_arxiv_recent",
        description=(
            "Get recent papers from arxiv by category. "
            "Use when you want the latest state-of-the-art in a specific area."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "category": {"type": "string", "default": "cs.LG", "description": "arxiv category (cs.LG, stat.ML, cs.CV, etc.)"},
                "days": {"type": "integer", "default": 7, "description": "Look back N days"},
                "max_results": {"type": "integer", "default": 20},
            },
        },
    ),
    Tool(
        name="get_paper_citations",
        description=(
            "Get papers that cite a given arxiv paper (Semantic Scholar). "
            "Use to find follow-up work and benchmark results."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "arxiv ID (e.g. '2106.05234') or Semantic Scholar ID"},
                "max_results": {"type": "integer", "default": 20},
            },
            "required": ["paper_id"],
        },
    ),
    Tool(
        name="recall_skill",
        description=(
            "Recall relevant skills/principles from the knowledge vault. "
            "Returns top-k items ranked by multi-strategy score "
            "(keyword + importance + recency + frequency). "
            "Use at decision points to leverage accumulated experience."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query describing the situation"},
                "k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="remember_experience",
        description=(
            "Persist a new experience/skill/anti-pattern to memory. "
            "Use after learning something valuable to share with future sessions."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "type": {"type": "string", "default": "skill", "enum": ["skill", "principle", "anti-pattern", "experiment", "session"]},
                "importance": {"type": "number", "default": 0.6, "minimum": 0, "maximum": 1},
                "tags": {"type": "array", "items": {"type": "string"}, "default": []},
                "persist": {"type": "boolean", "default": True, "description": "Write to vault on disk"},
            },
            "required": ["content"],
        },
    ),
    Tool(
        name="snapshot_memory",
        description=(
            "Get current working memory snapshot as text. "
            "Use when you need to see what's currently in context."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "max_chars": {"type": "integer", "default": 8000},
            },
        },
    ),
    Tool(
        name="knowledge_search",
        description=(
            "ONE-STOP search: papers + Kaggle discussions + vault skills combined. "
            "Returns a unified markdown report. Use when starting a new task "
            "or exploring an unfamiliar technique."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_per_source": {"type": "integer", "default": 10},
                "write_to_vault": {"type": "boolean", "default": False, "description": "Persist results for future reference"},
            },
            "required": ["query"],
        },
    ),
]


@app.list_tools()
async def list_tools() -> List[Tool]:
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    try:
        if name == "search_papers":
            agg = get_aggregator()
            papers = agg.search_papers_only(
                arguments["query"],
                max_results=arguments.get("max_results", 10),
                years_back=arguments.get("years_back", 5),
            )
            # 优先返回 Semantic Scholar 格式(有 TLDR)
            lines = [f"# Papers for: {arguments['query']}\n\n"]
            for p in papers:
                if isinstance(p, SSPaper):
                    lines.append(p.to_markdown())
                elif isinstance(p, ArxivPaper):
                    lines.append(p.to_markdown())
            return [TextContent(type="text", text="".join(lines))]

        elif name == "search_kaggle_discussions":
            kf = get_aggregator().kaggle
            slug = arguments.get("competition_slug", "").strip()
            if slug:
                discs = kf.search_competition(slug, arguments["query"])
            else:
                discs = kf.search_global(arguments["query"],
                                          arguments.get("max_results", 10))
            return [TextContent(type="text", text="".join(d.to_markdown() for d in discs))]

        elif name == "search_arxiv_recent":
            ax = get_aggregator().arxiv
            papers = ax.search_recent(
                category=arguments.get("category", "cs.LG"),
                days=arguments.get("days", 7),
                max_results=arguments.get("max_results", 20),
            )
            return [TextContent(type="text", text="".join(p.to_markdown() for p in papers))]

        elif name == "get_paper_citations":
            ss = get_aggregator().scholar
            citing = ss.get_citations(arguments["paper_id"],
                                      arguments.get("max_results", 20))
            return [TextContent(type="text", text="".join(p.to_markdown() for p in citing))]

        elif name == "recall_skill":
            mem = get_memory()
            items = mem.recall(arguments["query"], k=arguments.get("k", 5))
            lines = [f"# Recalled {len(items)} items for: {arguments['query']}\n\n"]
            for item in items:
                lines.append(f"## [{item.type}] {item.id} (imp={item.importance:.2f})\n{item.content[:1500]}\n\n")
            return [TextContent(type="text", text="".join(lines))]

        elif name == "remember_experience":
            from framework.src.memory import MemoryItem
            from datetime import datetime
            mem = get_memory()
            item = MemoryItem(
                id=arguments.get("id", f"auto-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
                content=arguments["content"],
                type=arguments.get("type", "skill"),
                importance=arguments.get("importance", 0.6),
                tags=arguments.get("tags", []),
            )
            mem.remember(item, persist=arguments.get("persist", True))
            return [TextContent(type="text", text=f"✅ Remembered: {item.id} (importance={item.importance})")]

        elif name == "snapshot_memory":
            mem = get_memory()
            return [TextContent(type="text", text=mem.snapshot(arguments.get("max_chars", 8000)))]

        elif name == "knowledge_search":
            agg = get_aggregator()
            report = agg.search_all(
                arguments["query"],
                max_per_source=arguments.get("max_per_source", 10),
            )
            text = report.to_markdown()
            if arguments.get("write_to_vault", False):
                path = agg.write_report_to_vault(report)
                text += f"\n\n---\n**📝 Persisted to**: `{path}`"
            return [TextContent(type="text", text=text)]

        else:
            return [TextContent(type="text", text=f"❌ Unknown tool: {name}")]

    except Exception as e:
        return [TextContent(type="text", text=f"❌ Tool error in {name}: {e}\n{type(e).__name__}")]


# ---------- 启动入口 ----------

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())