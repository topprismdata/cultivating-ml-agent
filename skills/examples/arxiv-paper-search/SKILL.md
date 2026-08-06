---
name: arxiv-paper-search
description: |
  Use when encountering a new ML technique not covered by existing 43+ skills, when needing citations for a paper/report, when comparing recent (last 6 months) approaches to a known problem, or when the existing knowledge feels stale. Triggers on phrases like "latest paper on X", "SOTA in Y", "what does the literature say about Z".
---

# Arxiv Paper Search (Real-Time Knowledge)

## Context
Static skills freeze at knowledge-cutoff date. ML papers come out daily. This skill enables the agent to **fetch fresh academic knowledge** directly from arxiv and Semantic Scholar at decision time. Essential for staying current with rapidly evolving fields (LLM agents, time series, multi-modal learning).

The core insight: **skills + live search = always-current knowledge**. Don't memorize; learn how to fetch.

## Guidance

### Basic Usage

```python
from framework.src.knowledge import ArxivSearch

search = ArxivSearch()  # respects arxiv's 3s rate limit
papers = search.search("time series forecasting transformer", max_results=10)

for p in papers:
    print(f"[{p.year}] {p.title}")
    print(f"  Authors: {', '.join(p.authors[:3])}")
    print(f"  PDF: {p.pdf_url}")
```

### Recent Papers by Category

```python
# "What came out in cs.LG this week?"
recent = search.search_recent(category="cs.LG", days=7, max_results=20)
```

### Use Semantic Scholar for Citations & TLDR

```python
from framework.src.knowledge import SemanticScholar

ss = SemanticScholar()  # add api_key= for higher rate limits
papers = ss.search("tabular foundation models", max_results=10)

for p in papers:
    if p.tldr:
        print(f"TLDR: {p.tldr}")  # auto-generated 1-sentence summary
    print(f"Cited by: {p.citation_count} (influential: {p.influential_citation_count})")
```

### Aggregate Everything

```python
from framework.src.knowledge import KnowledgeAggregator

agg = KnowledgeAggregator()
report = agg.search_all("cross-competition feature transfer", max_per_source=10)
print(report.to_markdown())  # papers + kaggle discussions combined

# Optionally persist to vault for future recall
path = agg.write_report_to_vault(report)  # writes to docs/ml-agent-memory/auto-search/
```

### Decision Workflow

```
1. Start a new technique exploration?
   → search_arxiv_recent(category="cs.LG", days=14)
2. Need to compare approaches?
   → search_papers(query) — Semantic Scholar returns TLDR + citations
3. Found an interesting one? Check follow-ups:
   → get_paper_citations(arxiv_id) — who built on it?
4. Combine with Kaggle discussions:
   → knowledge_search(query) — one-stop papers + forum
5. Persist findings:
   → write_report_to_vault(report)
   → mem.remember(MemoryItem(importance=0.8, type="experiment"))
```

## Why This Matters

| Without Live Search | With Live Search |
|---|---|
| Stuck with cutoff knowledge | Always current |
| Reinventing wheels | Build on latest |
| Can't cite sources | Reference arxiv IDs |
| Miss new techniques (e.g. TabPFN, Chronos-2) | Detect them within days |

Real example from this project: `store-sales-darts-chronos-blend` skill (v0.8.x) was enabled by **searching arxiv for "Chronos time series foundation model"** when it came out — without live search, we would have missed it.

## When to Apply

### When to Use
- Starting any task where the relevant technique is <2 years old
- Need academic citations in a writeup
- Comparing multiple approaches to a known problem
- Reading a paper and wanting to find follow-ups
- User explicitly asks "what's the latest on X?"
- Checking if a "novel" idea has been published before

### When NOT to Use
- Classical techniques (linear regression, basic GBDT) — knowledge is stable
- Pure implementation details (no academic content)
- When network is unavailable — fallback to existing skills
- Privacy-sensitive contexts — arxiv search is logged

## Notes
- **Rate limits**: arxiv asks for 3s between requests; Semantic Scholar 100/5min (no key), 5000/5min (with key)
- **Cite properly**: Always include arxiv_id and PDF URL in any report
- **Combine with vault**: After finding relevant papers, persist to memory with importance=0.7+
- **Cross-check with Kaggle**: An arxiv paper may be theoretically great but Kaggle discussions reveal what actually works in practice
- **Cache if possible**: Add `cache_dir` to avoid re-fetching in same session
- See also: `kaggle-discussion-search`, `memory-hierarchy-management`

## References
- Implementation: `framework/src/knowledge/arxiv_search.py`, `semantic_scholar.py`
- Aggregator: `framework/src/knowledge/aggregator.py`
- arxiv API: https://arxiv.org/help/api
- Semantic Scholar API: https://www.semanticscholar.org/product/api
- Related skill: `kaggle-discussion-search` (complementary data source)