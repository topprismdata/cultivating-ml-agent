---
name: self-evolving-skills
description: |
  Use when adding new skills to the library, when extracting insights from completed experiments, when consolidating lessons learned, or when running a project retrospective. Triggers after a successful Kaggle competition, after major model breakthrough, or when "we should remember this" comes up.
---

# Self-Evolving Skills (Voyager-Style)

## Context
Static skill libraries grow only when humans add entries. Voyager (Wang et al. NeurIPS 2023) showed agents that **auto-extract and register new skills** grow 100+ skills without manual work. agy verified this as P1 for ML agents: 43+ skills → 100+ with quality gates.

The core insight: **every experiment is a candidate skill**; quality gating prevents library pollution.

## Guidance

### Extract from Text/Conversation

```python
from framework.src.skill_evolution import SkillExtractor

ext = SkillExtractor()
candidates = ext.extract_from_text("""
    We should always check data leakage before training.
    Never use future information in features.
    Make sure to use walk-forward validation for time series.
""")
# → 3 candidates: 1 skill, 1 anti-pattern, 1 principle
```

### Extract from Experiment Log

```python
# 假设实验记录在 EXPERIMENTS.md
candidates = ext.extract_from_experiments_md(experiments_md_content)
# 自动按 importance 排序, top 20 入选
```

### Validate Before Registration

```python
from framework.src.skill_evolution import SkillValidator

val = SkillValidator()
for c in candidates:
    result = val.validate(c, existing_descriptions=existing)
    # verdict: APPROVE / REVISE / REJECT
    # issues: ['Too similar...', 'Missing frontmatter...']
    # similarity_to_existing: 0.0 - 1.0
```

### Register to MCP Library

```python
from framework.src.skill_evolution import SkillRegistry

registry = SkillRegistry(skills_dir="skills/examples")
for c in candidates:
    result = registry.register(c)
    if result.verdict.value == "approve":
        print(f"Registered: {c.name}")
    # 写入 skills/examples/{name}/SKILL.md
```

### Auto-Approve (Production Danger)

```python
# 生产环境慎用 — 应该人工 review
registry.register(candidate, auto_approve=True)
```

## Why This Matters

Without self-evolving:
- Skills grow slowly (43+ → manual additions)
- Lessons forgotten after session ends
- Same mistakes repeated

With self-evolving:
- 100+ skills in 6 months vs 5
- Lessons persist with proper templates
- Quality gates prevent bloat

agy: "Self-Evolving Skills is **mid-high ROI**; agents automatically grow from 43+ to 100+."

## When to Apply

### When to Use
- After successful Kaggle competition
- End of major experiment (consolidate learnings)
- Project retrospective
- When "we learned X" emerges repeatedly

### When NOT to Use
- One-off observations (not worth a skill)
- Without Validator (always validate)
- When existing skills cover the lesson

## Notes
- **Always validate first**: quality > quantity; bad skills hurt recall
- **Similarity >0.85 = duplicate**: reject or revise
- **Description must start with "Use when..."**: ensures triggering
- **Importance ≥0.7**: auto-archive to vault
- **Tags matter**: helps recall filtering
- See also: `memory-hierarchy-management` (where new skills land)

## References
- Implementation: `framework/src/skill_evolution/`
- Inspired by: Voyager (Wang et al. NeurIPS 2023), AIDE auto-skill-gen
- Quality gating: validator checks template + similarity + length
- Pairing skill: `claudeception` (already exists, similar pattern)