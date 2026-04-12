---
name: three-layer-wisdom-extraction
description: |
  Use when: (1) User says "总结经验", "复盘", "extract methodology", "what did we learn",
  "抽象智慧", or asks to reflect on a completed project, (2) A non-obvious breakthrough,
  major failure, or counterintuitive discovery has just occurred, (3) User asks "why does
  this pattern keep appearing" or "what's the deeper principle here", (4) Before starting
  a new project when user wants to review past wisdom. Do NOT trigger for routine bug fixes
  — use claudeception instead.
---

# Three-Layer Wisdom Extraction

Most knowledge capture stops at "what worked and what didn't." But the most valuable
insights — principles that transfer across entirely different domains — require deliberate
abstraction. This skill provides a structured process for that abstraction.

The three layers build on each other: you can't abstract universal principles from thin air,
you need the concrete timeline first, then the domain insights, then the philosophical leap.

## When This Skill is Worth Using

Not every task warrants three-layer extraction. It's worth the effort when:

- The experience involved a **non-obvious breakthrough** that required >30 minutes of investigation
- A **failed approach** revealed something surprising about the problem structure
- The solution involved a **counterintuitive choice** (e.g., making things worse before better)
- Multiple attempts followed a **recognizable pattern** (workaround trap, local optimum, etc.)

If the experience was straightforward ("followed docs, it worked"), skip this skill.

## The Three Layers

### Layer 1: Breakthrough Path

The raw timeline. What was tried, what happened, in what order.

Gather by reviewing conversation history, commit logs, or asking the user. Focus on
**decision points** — moments where a different choice would have led to a different outcome.

**Output**: A chronological list of attempts with outcomes and key metrics.

### Layer 2: Domain Knowledge

Field-specific insights. What patterns, diagnostics, and anti-patterns emerged?

This is what `claudeception` captures as skills. If claudeception has already run, reuse its
output here. The key additions: boundary conditions (when does the insight NOT apply?) and
diagnostics (what signal would have revealed this earlier?).

**Output**: Structured domain knowledge (decision trees, comparison tables, or skill files).

### Layer 3: Universal Principles

Cross-domain patterns extracted via the five abstraction questions below. These are the
highest-value output because they transfer to completely different contexts.

**Output**: 3-7 named principles, each with: one-sentence statement, explanation of why it
holds across domains, and concrete examples from at least 2 different fields.

## The Five Abstraction Questions (Core Tool)

For each Layer 2 insight, apply these questions systematically:

1. **INVERSION**: What is the OPPOSITE of this insight, and when would that opposite be correct?
   This reveals context-dependency. If the opposite is never correct, the insight is trivial.

2. **GENERALIZATION**: Strip away domain-specific terms. What is the abstract structure?
   "Stale lag features cause underprediction" → "Training-inference distribution mismatch
   causes systematic bias toward the training distribution."

3. **TRANSFER**: Where else does this abstract pattern appear? Find at least 2 other domains.
   "Same tool has opposite effects in different contexts" → medicine (drug interactions),
   cooking (ingredient combinations), economics (policy interactions).

4. **PARADOX**: What apparent contradiction does this insight resolve?
   "Geo blend helps one model type but destroys another" → "Effectiveness depends on
   interaction with the system, not just the intervention itself."

5. **META**: What does this reveal about the problem-solving process itself?
   "The workaround that works blocks the fundamental fix" → "Local optima in solution
   space are more dangerous than failures, because they satisfy."

## Multi-Perspective Analysis

Before finalizing Layer 3 principles, analyze from multiple perspectives (adapted from
Common Wisdom Model, Grossmann 2020, and Multi-Actor Insight Extraction, Nature 2025):

- **Pragmatist**: Is this principle actually actionable, or just intellectually satisfying?
- **Skeptic**: What evidence would DISPROVE this principle? If none exists, it may be unfalsifiable.
- **Cross-domain validator**: Does this truly transfer, or are the "other domain" examples
  forced analogies?

If a principle doesn't survive all three perspectives, it's probably still Layer 2.

## Validation Checklist

A Layer 3 principle passes quality checks when ALL of these hold:

- [ ] **Domain independence**: Explainable to someone outside the field without jargon
- [ ] **Predictive**: Would knowing this have changed your approach earlier?
- [ ] **Multi-domain**: At least 2 other domains where it applies
- [ ] **Non-trivial**: Not something a reasonable person would already assume
- [ ] **Actionable**: Suggests a concrete change in approach
- [ ] **Falsifiable**: Can describe evidence that would disprove it

If a principle fails any check, it's probably still Layer 2. Try generalizing further or
discard it.

## Extraction Process

### Step 1: Gather Layer 1

Review the conversation or project history. List each major attempt with:
- What was tried
- Result (success / failure / partial)
- Key metric or signal
- What prompted the next attempt

### Step 2: Extract Layer 2

For each breakthrough or significant failure in Layer 1:
- What domain-specific insight enabled or caused it?
- What diagnostic would catch this earlier next time?
- What are the boundary conditions (when does this NOT apply)?

If claudeception skills already exist for this domain, reference them rather than duplicating.

### Step 3: Abstract Layer 3

Take the 2-3 most impactful Layer 2 insights and apply all five abstraction questions to each.
Run multi-perspective analysis on each candidate principle. Discard any that don't pass the
validation checklist. Keep 3-7 principles max.

### Step 4: Format and Present

Present the three layers to the user in this format:

```
## Layer 1: Breakthrough Path
[timeline]

## Layer 2: Domain Knowledge
[references to existing skills or new insights]

## Layer 3: Universal Principles
[each principle with: name, one-line statement, why it transfers, domain examples]
```

Ask the user to review and challenge the Layer 3 principles. Their pushback often reveals
whether a principle is genuinely universal or still domain-specific.

## Integration with Existing Skills

| Skill | Layer | What it captures | Output format |
|-------|-------|-----------------|---------------|
| claudeception | Layer 2 | Domain-specific tactics | SKILL.md files |
| this skill | Layer 3 | Universal principles | Inline in conversation, or saved as wisdom skill |
| skill-refresh | Meta | Maintains both over time | Updates/deletes stale skills |

When running this skill, check for existing claudeception skills first — they provide
ready-made Layer 2 content. See `references/principles-from-competitions.md` for examples
extracted from past projects.

## Anti-Patterns

- **Over-extraction**: Not every experience yields wisdom. If you can't find genuine
  cross-domain transfer, stop at Layer 2. Forcing Layer 3 produces vacuous platitudes.

- **Renaming is not abstracting**: "Stale features cause underprediction" → "Bad features
  cause bad predictions" is NOT a Layer 3 abstraction. It's just vaguer. Real abstraction
  changes the structure: "Training-inference distribution mismatch causes systematic bias."

- **Too many principles**: More than 7 suggests you haven't abstracted enough. Look for
  meta-principles that subsume several specific ones.

- **Forced analogies**: "This is like X in domain Y" where the analogy only works on the
  surface. Real transfer requires structural similarity, not just surface resemblance.

## References

- Academic basis: Reflexion (Shinn 2023), ExpeL (Zhao 2024), Generative Agents (Park 2023)
- Cognitive science: Common Wisdom Model (Grossmann 2020) — wisdom as perspectival meta-cognition
- Multi-actor validation: Multi-Actor Insight Extraction (Nature Scientific Reports, 2025)
- Analogical transfer: Structure-Mapping Theory (Gentner 1983), Transferable Meta-Learning (Kang 2023)
- Charlie Munger's "latticework of mental models" framework
- See `references/principles-from-competitions.md` for worked examples
