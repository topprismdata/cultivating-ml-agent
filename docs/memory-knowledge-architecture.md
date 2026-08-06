# Memory & Knowledge Architecture

> **Last Updated**: 2026-08-05 | **Version**: 1.0 | **Status**: Implemented

## TL;DR

The project now has **two new layers** beyond static skills:

1. **Memory Hierarchy** (P0): 3-layer MemGPT-style memory — working context ↔ archival OKF ↔ filesystem
2. **Knowledge Layer** (P1): Real-time search across **arxiv**, **Semantic Scholar**, and **Kaggle discussions**, all exposed as MCP tools

Combined, they turn a static skill library (43+) into a **living knowledge base** that gets smarter with every search.

---

## 1. The Problem

Before this work, the project had:
- 43+ skills (markdown files, manually maintained)
- 16 principles (Layer 3 wisdom)
- OKF knowledge graph (edges between concepts)

**Issues**:
- Stale: knowledge cutoff = January 2026
- Slow: reading SKILL.md on demand wastes context
- No live data: missing papers from Feb-Aug 2026, latest Kaggle tricks
- Can't grow: new skills added by hand

## 3. Architecture

```
┌────────────────────────────────────────────────────────────┐
│ Agent Working Context (≤8000 chars, LRU)                   │
│   - Current task state                                     │
│   - Top-5 recalled skills (from Memory Hierarchy)          │
│   - Live search results (from Knowledge Layer)             │
└────────────────────────────────────────────────────────────┘
        ↕ recall / remember
┌────────────────────────────────────────────────────────────┐
│ Memory Hierarchy (offline, fast)                           │
│   - Working: 20 items max, LRU                             │
│   - Archival: OKF graph, multi-strategy recall             │
│   - Long-term: filesystem (auto-search/ directory)         │
└────────────────────────────────────────────────────────────┘
        ↕ search / write
┌────────────────────────────────────────────────────────────┐
│ Knowledge Sources (online, rate-limited)                   │
│   - arxiv.org API (papers, no key needed)                  │
│   - Semantic Scholar (papers + citations, optional key)    │
│   - Kaggle Discussions (forums, no key needed)             │
└────────────────────────────────────────────────────────────┘
        ↕ unified search
┌────────────────────────────────────────────────────────────┐
│ MCP Server (stdio protocol)                                │
│   Exposes 8 tools: search_papers, search_kaggle_discussions,│
│   search_arxiv_recent, get_paper_citations, recall_skill,  │
│   remember_experience, snapshot_memory, knowledge_search   │
└────────────────────────────────────────────────────────────┘
```

## 4. File Layout

```
framework/src/
├── memory/                    # Memory Hierarchy
│   ├── hierarchy.py          # 3-layer core
│   ├── recall.py             # Multi-strategy recall
│   └── okf_index.py          # OKF graph traversal
└── knowledge/                 # Live Knowledge
    ├── arxiv_search.py       # arxiv.org API client
    ├── kaggle_forum.py       # Kaggle discussions
    ├── semantic_scholar.py   # Semantic Scholar
    └── aggregator.py         # Unified search + report

framework/src/mcp/
└── server.py                  # MCP server exposing 8 tools

skills/examples/
├── memory-hierarchy-management/SKILL.md
├── arxiv-paper-search/SKILL.md
└── kaggle-discussion-search/SKILL.md

tests/
└── test_memory_and_knowledge.py  # 10 offline tests

docs/
└── memory-knowledge-architecture.md  # this file
```

## 5. Key Design Decisions

### 5.1 Why 3 layers, not 1?

Single-layer (everything in context) doesn't scale past ~30 items. Single-layer + filesystem (skills as files) wastes context on every retrieval. **3 layers balance speed, size, and relevance.**

### 5.2 Why multi-strategy recall?

Keyword alone misses semantic similarity. Recency alone buries important old lessons. **Multi-strategy (keyword + importance + recency + frequency)** matches human-like recall.

### 5.3 Why arxiv + Semantic Scholar?

- **arxiv**: latest, no key needed, but no citation graph
- **Semantic Scholar**: citations, TLDR, but rate-limited
- Combined: best of both worlds

### 5.4 Why MCP?

MCP is the 2025 standard for agent ↔ tool integration. Any agent (Claude Code, Cursor, etc.) can use our 8 tools out of the box, **without reading 500-line skill docs**.

### 5.5 Why no embedding search?

Embeddings require:
- Pre-computing embeddings for all 43+ skills (one-time)
- Embedding model download (sentence-transformers ~400MB)
- Network for query embedding

Keyword + multi-strategy works fine for <100 skills. **Re-evaluate at 100+**.

## 6. Usage

### From Python

```python
from framework.src.memory import MemoryHierarchy, MemoryItem
from framework.src.knowledge import KnowledgeAggregator

# 1. Initialize memory
mem = MemoryHierarchy(okf_dir="docs/ml-agent-memory")
mem.bootstrap(extra_dirs=["skills/examples"])

# 2. Recall at decision point
items = mem.recall("how to validate time series", k=5)

# 3. Live search
agg = KnowledgeAggregator()
report = agg.search_all("Chronos time series foundation", max_per_source=10)

# 4. Persist what we learned
mem.remember(MemoryItem(
    id="chronos-experiment-2026-08-05",
    content="Chronos-2 beat baseline by 4.7x on Store Sales",
    type="experiment",
    importance=0.8,
), persist=True)
```

### From MCP (Claude Code / Cursor / etc.)

Add to your MCP server config:

```json
{
  "mcpServers": {
    "cultivating-ml-agent": {
      "command": "python",
      "args": ["-m", "framework.src.mcp.server"]
    }
  }
}
```

Then in agent:

```
→ search_papers("TabPFN tabular")
→ search_kaggle_discussions("competition=spaceship-titanic query=CatBoost ensemble")
→ recall_skill("how to detect feature engineering saturation")
→ remember_experience(content="...", type="experiment", importance=0.8)
```

## 7. Testing

```bash
python tests/test_memory_and_knowledge.py
# 10 offline tests, all passing
```

Online tests (arxiv, Kaggle) require network; document at top of each file.

## 8. Future Roadmap

| Direction | Expected Value | Effort |
|---|---|---|
| Embedding-based recall (sentence-transformers) | +20% recall quality | Medium |
| LLM reranker (Claude Haiku for top-k) | +15% precision | Medium |
| Auto skill extraction from search results | Continuous growth | High |
| Integration with MLZero / AIDE | Code-agent compatibility | High |
| Caching layer (avoid duplicate fetches) | 3x speedup | Low |

## 9. Open Questions

- When to **forget**? Current: never (archival grows unbounded)
- When to **deduplicate** skills? If two skills say the same thing?
- How to detect **contradictory** evidence between arxiv and Kaggle?

These are open problems in the field; not unique to this project.

---

*See also*: `framework/src/memory/`, `framework/src/knowledge/`, `skills/examples/memory-hierarchy-management/`, `skills/examples/arxiv-paper-search/`, `skills/examples/kaggle-discussion-search/`