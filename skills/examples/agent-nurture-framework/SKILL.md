---
name: agent-nurture-framework
description: |
  Agent Nurture Framework: a systematic methodology for training AI agents from novice to expert
  through conversational knowledge crystallization. Use when: (1) designing an agent training pipeline,
  (2) consolidating fragmented skills into organized knowledge architecture, (3) integrating external
  learning resources (books, notebooks) into agent memory, (4) measuring agent capability growth over time.
  Based on 2 months of real-world experimentation with 156 skills and 51 memory documents across
  multiple ML competitions and projects.
status: active
---

# Agent Nurture Framework: From Novice to Expert Through Knowledge Crystallization

A practical framework for systematically developing AI agent capabilities, derived from 2 months of
real-world experimentation. This document covers the complete lifecycle: knowledge acquisition,
skill extraction, fragmentation management, knowledge consolidation, and progress measurement.

## Theoretical Foundation

This framework is grounded in **Nurture-First Development (NFD)** (Zhang, 2026, arXiv:2603.10808),
which proposes that domain-expert agents should be grown through structured conversational interaction
rather than pre-programmed. The core mechanism is the **Knowledge Crystallization Cycle**: fragmented
knowledge embedded in operational dialogue is periodically consolidated into structured, reusable assets.

Key insight from our experience: An agent went from needing 2 weeks to achieve top 10% in a Kaggle
competition (S6E2, February) to achieving the same in 24 hours (S6E4, April) -- a 14x speedup
attributable entirely to accumulated and crystallized knowledge.

---

## Part 1: The Learning Pipeline

### 1.1 Five-Stage Learning Loop

```
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │  ┌─────────┐    ┌──────────┐    ┌──────────┐           │
    │  │ 1. STUDY │───▶│ 2. VERIFY│───▶│ 3. APPLY │           │
    │  │ (Theory) │    │(Notebook)│    │(Practice)│           │
    │  └─────────┘    └──────────┘    └────┬─────┘           │
    │       ▲                              │                   │
    │       │                              ▼                   │
    │  ┌─────────┐    ┌──────────┐    ┌──────────┐           │
    │  │ 5. PLAN  │◀──│ 4. EXTRACT│◀──│ Encounter│           │
    │  │ (Next)   │    │(Crystallize)│  │ Problems │           │
    │  └─────────┘    └──────────┘    └──────────┘           │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
```

**Stage 1: Study (Theory Input)**
- Input: ML books, research papers, documentation, tutorials
- Action: Agent reads and comprehends, stores as memory documents
- Output: `memory/<topic>_learned.md` files

**Stage 2: Verify (Notebook Integration)**
- Input: Google Colab/Jupyter notebooks, code exercises
- Action: Convert theoretical knowledge into runnable experiments
- Output: Verified patterns, identified gotchas, edge cases documented
- **Critical**: This is where "book knowledge" becomes "practical knowledge"

**Stage 3: Apply (Real-World Practice)**
- Input: Kaggle competitions, real projects, production tasks
- Action: Apply accumulated knowledge to novel problems
- Output: Performance metrics, bug discoveries, workflow insights

**Stage 4: Extract (Knowledge Crystallization)**
- Input: Session experiences, debug traces, solutions discovered
- Action: Claudeception skill extracts reusable patterns
- Output: New skills created, existing skills updated
- **Trigger conditions**: Non-obvious debugging, workarounds, configuration insights, trial-and-error successes

**Stage 5: Plan (Gap Analysis)**
- Input: Capability matrix assessment, competition results
- Action: Identify knowledge gaps, select next learning resources
- Output: Directed study plan, notebook experiments to run

### 1.2 The Role of External Resources

| Resource Type | Role in Pipeline | Value Metric |
|---------------|-----------------|--------------|
| ML Books | Fill gaps in capability matrix | Did it enable solving a previously unsolvable problem? |
| Google Notebooks | Accelerate theory-to-practice cycle | Did it reduce experiment iteration time? |
| Kaggle Forums/Notebooks | Discover new techniques | Did it improve competition score? |
| Research Papers | Validate or challenge assumptions | Did it prevent a wrong approach? |
| Past Skills/Memory | Avoid repeating mistakes | Did it save debugging time? |

---

## Part 2: Three-Layer Knowledge Architecture

### 2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│ L1: Core Capabilities (Stable, rarely changes)          │
│                                                         │
│   ml-expert, data-science-assistant,                    │
│   kaggle-competition-best-practices,                    │
│   tool skills (mcp-builder, claude-native-agent)        │
│                                                         │
│   Characteristic: General knowledge, domain-agnostic    │
│   Update frequency: Monthly or less                     │
├─────────────────────────────────────────────────────────┤
│ L2: Domain Skills (Changes with experience)             │
│                                                         │
│   kaggle/*, reid/*, brain/*, infrastructure/*           │
│                                                         │
│   Characteristic: Domain-specific, project-specific     │
│   Update frequency: Per competition or project          │
│   Consolidation: Merge fragmented skills quarterly      │
├─────────────────────────────────────────────────────────┤
│ L3: Contextual Memory (Ephemeral, regularly cleaned)    │
│                                                         │
│   memory/s6e2_*, memory/s6e4_*, experiment logs         │
│                                                         │
│   Characteristic: Task-specific, time-sensitive          │
│   Lifecycle: Create → Crystallize → Archive/Delete      │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Layer Characteristics

**L1 (Core):**
- These are "textbook" level skills that encode fundamental knowledge
- They should be comprehensive (100+ lines) and well-structured
- Examples: `ml-expert`, `data-science-assistant`, `kaggle-competition-best-practices`
- Maintenance: Add new techniques when validated; rarely refactor

**L2 (Domain):**
- These are "playbook" level skills for specific domains
- They start as individual bug-fix or technique skills (fragmented)
- Over time, related skills should be **merged** into comprehensive domain skills
- Rule of thumb: If >5 skills share the same prefix, consider consolidation

**L3 (Memory):**
- These are "scratchpad" notes from current work
- They should be **periodically crystallized** into L1 or L2 skills
- If a memory file hasn't been referenced in 2+ weeks, archive it
- Never let memory files accumulate without crystallization

---

## Part 3: Managing Skill Fragmentation

### 3.1 The Fragmentation Problem

As an agent learns, it naturally accumulates many small, specific skills:
- Bug fixes (e.g., `sklearn-target-encoder-multiclass-shape`)
- Workarounds (e.g., `python-output-buffering`)
- Domain snippets (e.g., `kaggle-auc-binary-submission-bug`)

This creates **semantic noise**: when a problem arises, the skill matching system may surface
multiple similar but slightly different skills, causing confusion.

### 3.2 When to Keep Skills Separate vs. Merge

**Keep separate when:**
- Different trigger conditions (different error messages, different contexts)
- Skills are from different domains (ML bug vs. infrastructure issue)
- The skill is the canonical reference for a specific problem

**Merge when:**
- Same domain prefix (e.g., 13 Re-ID skills → 2 comprehensive skills)
- Trigger conditions overlap significantly
- No skill has been auto-triggered in 30+ days (low specificity)
- Skills share >50% of their solution content

### 3.3 Consolidation Strategy

```
Phase 1: Audit (Weekly)
  - Count skills by prefix/category
  - Identify clusters with >5 skills
  - Flag skills not triggered in 30 days

Phase 2: Merge (Per cluster)
  - Group related skills
  - Identify the "primary" skill (most comprehensive or most triggered)
  - Absorb secondary skills' trigger conditions and solutions
  - Add "See also" references for edge cases
  - Delete merged skills

Phase 3: Validate
  - Verify merged skill covers all original trigger conditions
  - Test that description enables semantic matching
  - Update SKILL_INDEX if maintained
```

### 3.4 Example Consolidation: Kaggle Skills

**Before (15 fragmented skills):**
```
kaggle-auc-binary-submission-bug
kaggle-data-format-first
kaggle-dataset-large-file-upload
kaggle-feature-boundary
kaggle-mlflow-tracking
kaggle-optimal-blending
kaggle-playground-external-data-validation
kaggle-quantized-model-limitations
kaggle-reid-csv-label-loading
kaggle-reid-submission-workflow
kaggle-submission-id-reset-index-bug
kaggle-top-performer-replication
kaggle-top-solution-replication
kaggle-competition-best-practices  (already comprehensive)
kaggle                             (submission limiter)
```

**After (3-4 consolidated skills):**
```
kaggle-competition-best-practices  (expanded: includes data-format, external-data, feature-boundary)
kaggle-submission-toolkit          (merged: auc-bug, reset-index-bug, large-upload, reid-workflow)
kaggle-ensemble-strategies         (merged: optimal-blending, top-performer-replication, top-solution-replication)
kaggle                             (submission limiter - keep as-is)
```

---

## Part 4: Notebook Integration Patterns

### 4.1 Why Notebooks Matter

Google Colab / Jupyter notebooks serve a unique role in the agent training pipeline:
- They are **executable documentation** -- theory becomes runnable code
- They provide **isolated experimentation** -- try without breaking production
- They enable **cross-platform verification** -- test on GPU/TPU environments

### 4.2 Integration Workflow

```
1. Agent reads ML book chapter → creates memory file with key concepts
2. User shares notebook implementing those concepts → agent verifies understanding
3. Agent identifies patterns/gotchas not in the book → extracts as skills
4. Patterns crystallize into L1/L2 skills → available for future tasks
```

### 4.3 Notebook-to-Skill Extraction Template

When processing a notebook, extract knowledge in this structure:

```markdown
## From Notebook: [notebook name]

### Key Technique Learned
[What new capability does this notebook teach?]

### Non-Obvious Insights
[What did the notebook reveal that isn't obvious from documentation?]

### Gotchas Discovered
[What errors or issues arose during execution?]

### Reusable Patterns
[What code patterns can be extracted for future use?]

### Integration with Existing Skills
[How does this relate to what the agent already knows?]
```

### 4.4 Notebook as Verification Layer

Notebooks also serve as the **verification step** between theory and practice:

```
Theory (Book/Paper) → "This technique should work for X"
    ↓
Notebook → "Confirmed: technique works, but requires Y adjustment for this dataset"
    ↓
Skill → "Use technique X with adjustment Y when encountering scenario Z"
```

Without the notebook verification step, the agent might memorize theoretical knowledge
without understanding practical constraints, leading to naive application.

---

## Part 5: Progress Measurement

### 5.1 Capability Matrix

Track agent capability across dimensions (rate 1-5):

```markdown
## Capability Assessment Template

| Dimension          | Rating | Evidence                                    |
|--------------------|--------|---------------------------------------------|
| Feature Engineering| 1-5    | Can create TE, pairwise, polynomial features|
| Model Training     | 1-5    | Can tune XGB/LGB/CB/HGB effectively        |
| Ensemble Methods   | 1-5    | Can implement hill-climbing, stacking       |
| Debugging          | 1-5    | Can diagnose training issues quickly        |
| Experiment Mgmt    | 1-5    | Can track and compare experiments           |
| Domain Knowledge   | 1-5    | Understands medical/tabular/Re-ID domains   |
| Infrastructure     | 1-5    | Can handle WSL/GPU/remote environments      |
```

Rating scale:
- **1**: Needs step-by-step guidance
- **2**: Can follow known patterns
- **3**: Can adapt patterns to new situations
- **4**: Can discover and validate new approaches
- **5**: Can innovate beyond documented techniques

### 5.2 Knowledge Growth Metrics

Track these metrics periodically:

```bash
# Total skills (should grow, then plateau as consolidation begins)
ls ~/.claude/skills/*/SKILL.md | wc -l

# Skills by category (should shift from many small to fewer large)
ls ~/.claude/skills/ | sed 's/-.*//' | sort | uniq -c | sort -rn

# Memory files (should cycle: create → crystallize → archive)
ls ~/.claude/memory/ | wc -l

# Stale memory files (not updated in 2+ weeks)
find ~/.claude/memory/ -name "*.md" -mtime +14 | wc -l

# Competition performance trend
# Track CV and LB scores across competitions/versions
```

### 5.3 Efficiency Indicators

| Indicator | What It Measures | Target |
|-----------|-----------------|--------|
| Time to top 10% | Overall agent capability | Decreasing over competitions |
| Skills triggered per task | Knowledge activation rate | High (most relevant skills surface) |
| New skills per session | Learning rate | Steady (1-3 per session) |
| Skills consolidated per month | Knowledge maturity | Increasing (signals crystallization) |
| Repeated mistakes rate | Learning effectiveness | Decreasing toward zero |

---

## Part 6: Implementation Guide for Other Agents

### 6.1 Minimum Viable Setup

To replicate this framework with any AI agent, you need:

1. **Skill extraction mechanism**: A skill that creates other skills (like Claudeception)
2. **Persistent memory**: Files or a database that survives session boundaries
3. **Semantic skill matching**: Ability to find relevant skills by description
4. **Three-layer structure**: Core / Domain / Contextual knowledge separation

### 6.2 Adaptation for Different Platforms

| Platform | Skill System | Memory System | Notes |
|----------|-------------|---------------|-------|
| Claude Code | ~/.claude/skills/ | ~/.claude/memory/ | Native skill matching via description |
| Codex CLI | ~/.codex/skills/ | ~/.codex/memory/ | Adapt skill format to Codex conventions |
| Gemini CLI | Skills via GEMINI.md | Local files | May need different trigger mechanism |
| Custom Agent | Any markdown files | Any persistent storage | Implement your own matching |

### 6.3 Critical Success Factors

From our experience, these factors determined success:

1. **Extraction discipline**: Extract skills immediately after discovery, not later
2. **Quality over quantity**: A well-written comprehensive skill beats 5 fragmented ones
3. **Regular consolidation**: Schedule monthly skill audits and merges
4. **Memory hygiene**: Archive or delete stale memory files; don't let them accumulate
5. **External resources**: Books and notebooks are most valuable when they fill identified gaps
6. **Competitive feedback**: Kaggle LB scores provide objective measurement of capability growth

---

## Part 7: Case Study — 2 Months of Agent Development

### Timeline

| Period | Competition | Time to Top 10% | Skills | Key Learning |
|--------|-------------|-----------------|--------|-------------|
| Feb 2026 | S6E2 (Medical) | ~2 weeks | ~10 | Basic ML pipeline, debugging patterns |
| Mar 2026 | WorldQuant BRAIN | Ongoing | ~60 | Feature engineering, alpha expressions |
| Apr 2026 | S6E4 (Irrigation) | ~24 hours | 156 | Reuse of crystallized knowledge |

### What Changed Between S6E2 and S6E4

**Eliminated time sinks:**
- Research phase: 0 hours (vs 3+ days for S6E2) -- knowledge already in skills
- Wrong directions: 0 (vs 5+ dead ends in S6E2) -- past mistakes documented as skills
- Infrastructure issues: ~30 min (vs 2+ days for S6E2) -- WSL/GPU skills pre-built
- Code template: ~1 hour (vs 3+ days for S6E2) -- previous scripts serve as templates

**Knowledge reuse:**
- 15 kaggle-* skills directly applied
- 18 ML training skills directly applied
- 6 infrastructure skills directly applied
- `ml-expert` and `data-science-assistant` as foundation

**This demonstrates the core principle**: Crystallized knowledge compounds over time. Each
competition makes the agent faster at the next one, not just because it "remembers" but
because knowledge is actively organized and retrievable.

---

## Part 8: Key Skills and Their Application Patterns

This section documents the specific skills that power the nurture framework, including how
they interact and when to use each one. Other agents can replicate these patterns on their
own platforms.

### 8.1 Claudeception — The Knowledge Extraction Engine

**What it does**: Automatically extracts reusable knowledge from work sessions and creates
new skills. This is the "crystallization" mechanism that makes the entire framework work.

**When it triggers**:
- After fixing a non-obvious bug (solution required >10 min investigation)
- After discovering a workaround for a tool/framework limitation
- After trial-and-error success (tried multiple approaches before finding what works)
- After finding a configuration insight specific to the project
- When user explicitly asks "save this as a skill" or "what did we learn?"

**How it works in practice**:
```
1. Problem encountered during work
2. Solution found through investigation/experimentation
3. Claudeception evaluates: Is this knowledge reusable? Non-trivial? Specific? Verified?
4. If yes → Creates new skill with:
   - YAML frontmatter (name, description with trigger conditions, version, date)
   - Problem/Context section
   - Step-by-step Solution
   - Verification method
   - Notes and caveats
5. Skill saved to ~/.claude/skills/<name>/SKILL.md
6. Immediately available for future sessions
```

**Real examples from our experience**:
- `python-output-buffering`: After wasting 45 min on "why is my log file empty?", extracted
  the knowledge that `nohup python` uses full buffering → always use `python3 -u`
- `catboost-multicore-config`: After CatBoost took 8 hours for 5 folds on CPU, discovered
  `task_type='CPU'` with `thread_count=-1` plus specificbootstrap_type settings
- `kaggle-auc-binary-submission-bug`: After CV-LB mismatch, discovered that submitting
  binary 0/1 instead of probabilities causes silent accuracy degradation

**Integration pattern**: Claudeception is the "Stage 4: Extract" of the learning loop.
Without it, knowledge remains trapped in conversation context and is lost when sessions end.

### 8.2 Memory (Knowledge Graph) — Long-Term Context Storage

**What it does**: Stores structured knowledge across sessions using Zettelkasten-style
knowledge graph with bidirectional links.

**When to use**:
- Before starting a new task → query for similar past tasks
- After discovering architectural decisions → store as ADR (Architecture Decision Record)
- When connecting new learnings to past experiments → create bidirectional links

**How it differs from skills**:
```
Memory files:                          Skills:
- Unstructured notes                   - Structured (YAML + sections)
- Session-specific context             - Reusable across sessions
- Temporary (crystallize or archive)   - Persistent
- Store "what happened"                - Store "how to solve"
- Example: s6e2_optuna_key_insights    - Example: kaggle-optimal-blending
```

**Lifecycle**:
```
Create (during work) → Reference (in later sessions) → Crystallize (into skill) → Archive
```

### 8.3 ML-Expert — The Core Knowledge Foundation

**What it does**: Comprehensive ML/DL knowledge base covering the full spectrum from data
processing to model deployment. Acts as the L1 "core capability" layer.

**When it triggers**: Any ML-related task — model training, data preprocessing, feature
engineering, hyperparameter tuning, evaluation.

**Content structure**:
```
1. Data Processing & Feature Engineering
2. Traditional ML (scikit-learn, XGBoost, LightGBM, CatBoost)
3. Deep Learning (PyTorch, TensorFlow, architectures)
4. Model Evaluation & Optimization
5. Experiment Management (MLflow, tracking)
6. Domain-Specific patterns
```

**Integration pattern**: ML-Expert is the "Stage 1: Study" foundation. When new ML books
or papers are read, their knowledge should be absorbed into ml-expert or spawn new
specialized skills.

### 8.4 Kaggle-Competition-Best-Practices — Competition Methodology

**What it does**: Comprehensive competition workflow from data exploration to final
submission, with accumulated strategies from 4 competitions.

**When it triggers**: Starting any Kaggle competition, or needing competition-specific
strategies.

**Key patterns encoded**:
```
1. Data Format First — verify data format before any research
2. External Data Validation — validate external datasets before use
3. Feature Boundary Testing — systematic feature engineering with boundary checks
4. Optimal Blending — 80/20 rule for ensemble construction
5. Top Performer Replication — systematic methodology to learn from leaders
6. Submission Workflow — complete pipeline from model to Kaggle submission
```

### 8.5 Skill-Creator — Skill Quality Assurance

**What it does**: Ensures new skills follow proper structure and quality standards.

**When to use**: When Claudeception creates a new skill, skill-creator validates:
- YAML frontmatter completeness
- Description contains specific trigger conditions
- Solution has been verified to work
- No sensitive information included

**Quality criteria checklist**:
```
[ ] Description contains specific trigger conditions
[ ] Solution has been verified to work
[ ] Content is specific enough to be actionable
[ ] Content is general enough to be reusable
[ ] No sensitive information (credentials, internal URLs)
[ ] Skill doesn't duplicate existing documentation
```

### 8.6 Continuous-Learning — Session-End Knowledge Review

**What it does**: Reviews entire session at completion to identify all extractable knowledge.

**When it triggers**: End of session, or when user runs `/claudeception`.

**Process**:
```
1. Review conversation history for extractable knowledge
2. List potential skills with brief justifications
3. Prioritize by value and reusability (typically 1-3 per session)
4. Extract top candidates
5. Report what was created and why
```

### 8.7 How Skills Interact — The Complete Workflow

```
User starts a new ML task
    │
    ├──▶ ml-expert activated (L1 core knowledge)
    │       Provides foundational ML concepts and patterns
    │
    ├──▶ kaggle-competition-best-practices activated (L2 domain)
    │       Provides competition-specific workflow
    │
    ├──▶ memory queried (L3 context)
    │       Retrieves similar past tasks and lessons learned
    │
    ├──▶ Domain-specific skills triggered as needed
    │       e.g., catboost-multicore-config, ensemble-model-correlation-trap
    │
    ├──▶ Problem encountered during work
    │       │
    │       ├──▶ search-complex-problems triggers web search
    │       │       Finds current solutions online
    │       │
    │       ├──▶ investigate performs systematic debugging
    │       │       Root cause analysis and fix
    │       │
    │       └──▶ Claudeception extracts new skill
    │               Knowledge crystallized for future use
    │
    └──▶ Session ends
            │
            └──▶ continuous-learning reviews session
                    Final extraction and quality check
```

### 8.8 Adapting These Skills for Other Agents

| Skill Pattern | Core Mechanism | How to Replicate |
|---------------|---------------|-----------------|
| Claudeception | Self-referential skill creation | Implement a "meta-skill" that can create new skills based on session learnings |
| Memory | Persistent file storage with links | Use any structured storage (JSON, markdown, SQLite) with cross-referencing |
| ML-Expert | Comprehensive domain knowledge | Start with a broad foundation skill, expand through study + crystallization |
| Best-Practices | Workflow methodology | Document processes after each successful project, refine over iterations |
| Skill-Creator | Quality validation | Implement schema validation for new skills (required fields, trigger conditions) |
| Continuous-Learning | Session review | Run a "retrospective" at end of each session, evaluate what was learned |

---

## References

- Zhang, L. (2026). "Nurture-First Agent Development: Building Domain-Expert AI Agents Through Conversational Knowledge Crystallization." arXiv:2603.10808
- "Modular Intelligence: A Skill-Based Paradigm for Scalable AI Agent Architecture." ResearchGate, 2026.
- "Rethinking AI Agents: From Fragmentation to Composable Skills." LinkedIn, 2026.
- "Don't Just Build Agents, Build Memory-Augmented AI Agents." MongoDB, 2026.
- "Your AI Agent Has Amnesia. And You Designed It That Way." dev.to, 2026.
- "Building an AI-Ready Knowledge Base: Best Practices for 2026." Rezolve.ai, 2026.
