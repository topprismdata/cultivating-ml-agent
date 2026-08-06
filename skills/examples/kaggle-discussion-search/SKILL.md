---
name: kaggle-discussion-search
description: |
  Use when looking for top-solution tricks on a Kaggle competition, when needing CV-LB gap discussions, when validating a strategy against community experience, or when seeking hyperparameter advice from competition winners. Triggers on phrases like "what did top scorers do", "CV-LB gap in this competition", "winning trick for X", or when starting any Kaggle competition.
---

# Kaggle Discussion Search (Community Knowledge)

## Context
Kaggle discussions contain **non-official, practitioner-tested wisdom** that often beats academic papers in practical value. Top solution write-ups, hyperparameter choices, CV-LB gap analyses — all are public. This skill teaches the agent to mine this gold mine at decision time, especially when starting a new competition or validating an existing skill against community evidence.

The core insight: **paper says "5-10% improvement", Kaggle discussion says "actually 2% on LB, watch out for X"**. Different signal, complementary.

## Guidance

### Basic Usage

```python
from framework.src.knowledge import KaggleForum

kf = KaggleForum()
# Get top-voted discussions for a specific competition
top_discs = kf.get_top_solutions("spaceship-titanic", top_n=20)

for d in top_discs:
    print(f"[{d.votes}↑ {d.comments_count}💬] {d.title}")
    print(f"  by {d.author}: {d.url}")
```

### Search by Keyword

```python
# Find discussions mentioning a specific technique
discs = kf.search_competition(
    competition_slug="store-sales-time-series-forecasting",
    query="CatBoost ensemble",
    sort_by="votes"
)
```

### Global Search (Cross-Competition)

```python
# Search across all Kaggle discussions
discs = kf.search_global("CV-LB gap", max_results=20)
```

### Decision Workflow

```
Starting new Kaggle competition?
  ↓
1. kf.get_top_solutions(competition_slug, top_n=30)
   → Read the top-voted discussions
   → Note: hyperparameter choices, CV strategies, common pitfalls
  ↓
2. kf.search_competition(slug, query="LB score CV correlation")
   → Find CV-LB gap analysis specific to this competition
  ↓
3. Cross-reference with skills:
   → cv-lb-gap-acknowledgment (general principle)
   → time-series-walk-forward-validation (if time series)
  ↓
4. After submission, search for related discussions:
   → kf.search_competition(slug, query="your technique here")
   → Validate your approach
```

### Combine with Papers

```python
from framework.src.knowledge import KnowledgeAggregator

agg = KnowledgeAggregator()
# One-shot: papers + Kaggle + vault combined
report = agg.search_all("feature engineering saturation", max_per_source=10)
# report.scholar_papers, report.arxiv_papers, report.kaggle_discs
```

## Why This Matters

| Source | Strength | Weakness |
|---|---|---|
| arxiv paper | Novel, rigorous, citable | Often idealistic, not battle-tested |
| **Kaggle discussion** | **Real-world validated, hyperparameter-specific** | **Ancedotal, scattered** |
| Skills (static) | Curated, distilled | Cutoff knowledge |

Real example: The `feature-engineering-saturation-detection` skill was created after seeing **multiple Kaggle discussions** warning about feature engineering saturation in tabular competitions — combining community signal with formal theory.

## When to Apply

### When to Use
- Starting any new Kaggle competition (read top solutions first)
- Looking for hyperparameter advice from competition winners
- Validating whether your CV score will translate to LB
- Finding "non-obvious" tricks (post-processing, ensemble weights, etc.)
- Comparing your approach to community consensus

### When NOT to Use
- Pure research / academic projects (no Kaggle context)
- When network unavailable
- For deep technical theory (use arxiv instead)
- Privacy-sensitive contexts

## Notes
- **Sort by votes** first — top-voted discussions are usually most informative
- **Read comments, not just post** — often the real insight is in comments
- **Cross-check across competitions** — same author posting on multiple competitions is a strong signal
- **Combine with paper search** — best insights usually combine theory (arxiv) + practice (Kaggle)
- **Cache results** in vault (importance=0.7+) for future reference
- See also: `arxiv-paper-search`, `cv-lb-gap-acknowledgment`, `memory-hierarchy-management`

## References
- Implementation: `framework/src/knowledge/kaggle_forum.py`
- Aggregator: `framework/src/knowledge/aggregator.py`
- Kaggle discussions: https://www.kaggle.com/discussions
- Real skills enabled by Kaggle mining: `feature-engineering-saturation-detection`, `cv-lb-gap-acknowledgment`, `ml-sweet-spot`
- Related skill: `arxiv-paper-search` (complementary data source)