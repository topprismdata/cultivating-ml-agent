---
name: memory-hierarchy-management
description: |
  Use when context window is getting crowded, when agent needs to recall past experiences, or when starting a long-running task that spans multiple sessions. Triggers when you have 43+ skills and can't fit them all in context, when you need to decide "what should be in working memory right now", or when persisting new learnings for future sessions.
---

# Memory Hierarchy Management (MemGPT-Style)

## Context
With 43+ skills and growing, stuffing everything into the agent's context window is impossible and wasteful. The naive approach (read SKILL.md on demand) is slow and lossy. This skill provides a 3-layer memory architecture: **Working Memory** (limited, in-context), **Archival Store** (unlimited, OKF-indexed), **Long-term Storage** (filesystem). Recall is multi-strategy (keyword + importance + recency + frequency).

The core insight: **context is RAM, vault is disk**. We treat them differently.

## Guidance

### The 3 Layers

```
Working Memory (in-context, ≤8000 chars)
    ↑↓ auto-promote / LRU evict
Archival Store (OKF graph, unlimited)
    ↑↓ write to filesystem on persist
Long-term Storage (docs/ml-agent-memory/auto-search/)
```

### When to Recall

```python
from framework.src.memory import MemoryHierarchy

mem = MemoryHierarchy(okf_dir="docs/ml-agent-memory")
mem.bootstrap(extra_dirs=["skills/examples", "docs"])

# At decision points — recall relevant skills
items = mem.recall("time series walk-forward validation", k=5)
for item in items:
    print(f"[{item.type}] {item.id} (imp={item.importance})")
```

### When to Remember

```python
from framework.src.memory import MemoryItem

# High-importance → auto-archive
mem.remember(MemoryItem(
    id="auto-finding-2026-08-05-catboost-vs-xgboost",
    content="In S6E2, CatBoost OOF 0.8124 beat XGBoost 0.8003 (+0.012)",
    type="experiment",
    importance=0.8,
    tags=["tabular","catboost","s6e2"],
), persist=True)
```

### Multi-Strategy Recall (Advanced)

```python
from framework.src.memory.recall import multi_strategy_recall, RecallConfig

cfg = RecallConfig(
    use_keyword=True,
    use_recency=True,       # Decay old items
    use_importance=True,    # Promote high-imp
    use_access_frequency=True,  # Frequently-used stays
    recency_decay_days=30,
    final_top_k=5,
)
results = multi_strategy_recall(mem.archival, "tabular feature engineering", cfg)
```

## Why This Matters

Without hierarchy:
- All 43 skills need to be in context (~25K chars) → leaves no room for actual work
- Agent forgets what worked yesterday
- No way to know "which skill is most relevant right now"

With hierarchy:
- Only 5-20 items in working memory at any time
- Skills auto-promoted based on multi-strategy score
- Important findings auto-persist to vault

**ROI**: 3-5x reduction in context usage; 2x improvement in skill relevance at decision points.

## When to Apply

### When to Use
- Starting any new ML competition/task
- After learning something valuable worth preserving
- When context window feels crowded
- When you need to recall past experience ("have we tried X before?")

### When NOT to Use
- Truly novel task with no history (no skills exist yet)
- One-off quick question
- When working memory isn't full (overhead not worth it)

## Notes
- **Importance threshold**: ≥0.7 items auto-archive; <0.5 stay in working only
- **Recall vs Read**: recall = multi-strategy ranking; read = direct fetch (when you know exactly which skill you want)
- **Bootstrap once per session**: indexing 43+ skills is slow; cache the index
- **Combine with knowledge_search**: for new techniques not in vault, also call `arxiv_search` or `kaggle_search`
- See also: `runtime-self-reflection` (uses memory hierarchy for anti-pattern storage)

## References
- Architecture: `framework/src/memory/hierarchy.py`
- Recall strategies: `framework/src/memory/recall.py`
- OKF integration: `framework/src/memory/okf_index.py`
- Inspired by: MemGPT (Packer et al. 2023), Letta framework