---
name: mem0-dynamic-graph-memory
description: |
  Use when extracting entities and relationships from text, when building dynamic knowledge graphs, when wanting to upgrade from static OKF to entity-aware memory, or when analyzing papers/discussions for concepts. Triggers when ingesting new papers, processing Kaggle writeups, or when existing skills can't capture connections between concepts.
---

# Mem0 Dynamic Graph Memory (Entity-Relation Extraction)

## Context
Static OKF stores predefined edges. Mem0/Letta (2025) extract **entities + relations dynamically** from text. agy suggested upgrading our existing `EntityRelationExtractor` as P2: enables automatic concept discovery instead of manual OKF curation.

The core insight: **knowledge graphs are useful only if they reflect real relationships**, and those relationships must be extracted, not hardcoded.

## Guidance

### Extract from Paper/Discussion

```python
from framework.src.memory.entity_extraction import EntityRelationExtractor

ext = EntityRelationExtractor()
text = """
LightGBM uses histogram-based gradient boosting.
XGBoost improves LightGBM with second-order gradients.
CatBoost handles categorical features natively, unlike LightGBM.
We evaluated F1 score on Spaceship Titanic.
Walk-forward validation prevents data leakage in time series.
"""

entities, relations = ext.extract(text)
# → 8 entities (techniques, metrics, competitions, concepts)
# → relations: "XGBoost improves LightGBM", etc.

for e in entities:
    print(f"{e.type}: {e.name} ({e.mentions} mentions)")
for r in relations:
    print(f"{r.source} --{r.relation}--> {r.target}")
```

### With LLM Enhancement

```python
def my_llm(prompt: str) -> str:
    return openai_client.chat.completions.create(...).choices[0].message.content

ext = EntityRelationExtractor(llm_call=my_llm)
entities, relations = ext.extract(text)
# → 规则抽取 + LLM 抽取合并去重
```

### Integrate with OKF Index

```python
# 已存在 OKFIndex,可叠加 entity edges
from framework.src.memory.okf_index import OKFIndex
from framework.src.memory.entity_extraction import EntityRelationExtractor

okf = OKFIndex(okf_dir="docs/ml-agent-memory")
okf.build()

ext = EntityRelationExtractor()
entities, relations = ext.extract(paper_abstract)

# 把 entity edges 添加到 OKF 图
for e in entities:
    okf.items[f"entity:{e.name.lower()}"] = MemoryItem(
        id=f"entity:{e.name.lower()}",
        content=f"{e.type}: {e.name}",
        type="entity",
        importance=0.6,
    )
```

## Why This Matters

| Static OKF | Dynamic Entity-Relation |
|---|---|
| Edges hardcoded in markdown | Edges extracted from text |
| Stale (doesn't reflect new content) | Always current |
| Misses implicit connections | Finds "X improves Y" relations |
| Manual curation required | Auto-scaling |

agy: "Mem0 dynamic graph memory enables cross-Competition knowledge transfer at scale."

## When to Apply

### When to Use
- Ingesting new paper or discussion
- Building knowledge graph from corpus
- Analyzing relationships between techniques
- When OKF misses connections you see in papers

### When NOT to Use
- Short texts (LLM overhead not worth it)
- Highly specialized domain (entity patterns need extension)
- Real-time scenarios (extraction takes seconds)

## Notes
- **Pattern-based fallback**: works without LLM, slightly lower recall
- **Type coverage**: technique / tool / metric / competition / concept
- **Dedup is critical**: same entity mentioned multiple times = single node
- **Confidence scoring**: relation extraction includes evidence snippet
- See also: `memory-hierarchy-management`, `time-series-walk-forward-validation`

## References
- Implementation: `framework/src/memory/entity_extraction.py`
- Inspired by: Mem0 (2024), Letta/MemGPT (2023), GLiNER (NER model)
- Pairing: `framework/src/memory/okf_index.py` (existing static OKF)