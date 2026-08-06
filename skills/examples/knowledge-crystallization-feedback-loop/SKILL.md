---
name: knowledge-crystallization-feedback-loop
description: |
  Use when: (1) You've completed a competition or experiment and want to extract
  reusable knowledge, (2) Your agent keeps repeating the same mistakes across
  competitions because lessons weren't crystallized, (3) You have 50+ memory files
  but can't find relevant knowledge when starting a new task, (4) You need to
  decide what to remember vs what to forget. Covers the complete crystallization
  cycle: Experiment → Identify Outcome → Extract Pattern → Classify (feedback/
  learned/project/reference) → Store with triggers → Activate on match. Includes
  the "3-layer architecture" (L1 core, L2 domain skills, L3 cross-domain
  principles) and the "forgetting protocol" for stale knowledge. Validated across
  4+ months, 20+ competitions, 109+ crystallized memories.
---

# Knowledge Crystallization Feedback Loop

## Problem
AI agents that compete in multiple ML competitions generate enormous experience,
but without systematic crystallization:
- Same mistakes repeat (e.g., submitting without format check — 5 times)
- Hard-won insights are lost between sessions
- Knowledge files grow to 100+ items with no organization
- Retrieval fails when needed (can't find relevant lesson for new competition)

## The Crystallization Cycle

```
    ┌─────────────────────────────────────────┐
    │ 1. EXPERIMENT                            │
    │    Run competition / try approach        │
    └──────────────┬──────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────┐
    │ 2. IDENTIFY OUTCOME                      │
    │    Success? Failure? Marginal? Dead end? │
    └──────────────┬──────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────┐
    │ 3. EXTRACT PATTERN                       │
    │    What's the 1-sentence rule?           │
    │    What evidence supports it?            │
    │    When does it apply? When NOT?         │
    └──────────────┬──────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────┐
    │ 4. CLASSIFY                              │
    │    feedback = "don't do X" (anti-pattern)│
    │    learned  = "do X for result Y"        │
    │    reference = "X is located at Y"       │
    └──────────────┬──────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────┐
    │ 5. STORE WITH TRIGGERS                   │
    │    Frontmatter description = when to use │
    │    Tags = search keywords                │
    │    Links = [[related-skills]]            │
    └──────────────┬──────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────┐
    │ 6. ACTIVATE ON MATCH                     │
    │    New task matches trigger → load skill │
    │    Verify still current → apply or update│
    └─────────────────────────────────────────┘
```

## The 3-Layer Knowledge Architecture

### L1: Core Capabilities (Monthly Update)
Basic ML competencies that rarely change:
- Cross-validation setup
- Feature engineering basics
- Model selection (GBDT vs NN vs ensemble)
- Submission format validation

### L2: Domain Skills (Per-Competition Update)
Competition-specific patterns organized as SKILL.md files:
- `autogluon-first` — when to use AutoGluon
- `trueskill-simulation-competition-strategy` — simulation competitions
- `kaggle-oof-lb-validation-protocol` — validation methodology
- 31+ skills total, each with trigger conditions + evidence

### L3: Cross-Domain Principles (Quarterly Review)
Abstract lessons that transfer across ALL competitions:
- `work_smart_not_hard` — external data ROI ~7× self-training
- `local_optimum_trap` — 3× <0.001 improvement → pivot
- `controlled_variable` — one change per experiment
- `signal_dilution` — correlated models > 0.97 = worse ensemble

## What to Crystallize vs What to Forget

### ALWAYS Crystallize
| Outcome | Type | Example |
|---------|------|---------|
| Failed approach (3+ tries) | feedback | "Anti-Psychic modifications to v29 PTCG agent lose -174 to -289 LB" |
| Successful breakthrough | learned | "CatBoost-heavy blend (0.7 CAT) beats equal blend by +2 points" |
| Meta-insight | reference | "ROGII requires ravaghi/wellbore artifacts + koolbox-offline datasets" |
| Cross-competition pattern | L3 principle | "OOF-LB gap is asymmetric: small data -0.087, large data +0.007" |

### NEVER Crystallize (Forget Instead)
| What | Why |
|------|-----|
| Specific code snippets | Can be re-derived from reading code |
| Git history / commit messages | `git log` is authoritative |
| Debugging recipes | The fix is in the code |
| File paths / project structure | Read current state instead |
| Ephemeral task details | Session-specific, no reuse value |
| CLAUDE.md contents | Already loaded each session |

## The Forgetting Protocol

Memory files become stale. Review quarterly:

```
For each memory file:
  1. Is the fact still true? (verify against current state)
  2. Was it superseded by a newer finding?
  3. Is the referenced file/path still valid?
  
  If NO to any → UPDATE or DELETE
```

**Example**: "task017 is best implemented with 1-node Conv (900 params)" — if a
new technique reduces to 90 params, UPDATE the memory, don't create a new one.

## Activation Triggers: Writing Good Descriptions

The `description` field in frontmatter is the ACTIVATION TRIGGER. It determines
whether the skill is loaded for a new task.

**Bad description** (too vague):
```yaml
description: "Tips for Kaggle competitions"
```

**Good description** (specific triggers):
```yaml
description: |
  Use when: (1) Your OOF score improved but LB didn't, (2) You need to decide
  whether to submit based on OOF alone, (3) OOF/LB gap larger than 1%.
  Covers: 5 sources of OOF/LB gap, confirmation protocol, 3-strike rule.
```

## AutoMem Integration (Graph + Vector Memory)

For agents with AutoMem (FalkorDB + Qdrant) deployed:
1. **Store**: key decisions → AutoMem graph with typed relationships
2. **Recall**: semantic query returns related decisions + alternatives considered
3. **Multi-hop**: "why PostgreSQL?" → finds the decision + alternatives + principle
4. **Consolidation**: merge duplicate memories, strengthen important ones

**vs File-based memory**:
| | File-based (Markdown) | AutoMem (Graph+Vector) |
|---|---|---|
| Retrieval | Keyword grep | Semantic search |
| Relationships | Manual [[links]] | Automatic graph edges |
| Cross-session | Read files | REST API / MCP |
| Setup cost | Zero | Docker stack (FalkorDB + Qdrant) |
| Best for | Small (<50 items) | Large (100+ items) |

## Evidence

- 4+ months of Claude Code usage
- 20+ competitions crystallized into 31+ skills
- 109+ individual memory entries (AutoMem)
- Key validated patterns:
  - "0.6 × BEST_PUBLIC" rule (5+ competitions, mean ratio 0.99)
  - "3-strike rule" for OOF plateau (TPS May 2022, 27 configs converged)
  - "Re-submit = reset convergence" (PTCG, 8+ submissions tracked)
  - "Pipeline trim for code competitions" (ROGII, Pipeline A OOF 10.38)

## Common Crystallization Failures

| Failure | Example | Fix |
|---------|---------|-----|
| Too specific | "v35 anti-Psychic loses 289 LB" | Generalize: "scoring modifications to converged agents hurt" |
| Too abstract | "Be careful with submissions" | Specify: "Re-submitting resets TrueSkill μ to 600" |
| No trigger | "House Prices 0.11750" | Add: "Use when: regression competition, blending models" |
| No evidence | "Blending usually helps" | Add: "s6e7: CAT-heavy 0.949 > equal 0.947 > LGB-only 0.943" |
| Stale | "Use GLM-4-air for Industrial T1" | Update or delete if superseded |
