---
name: skill-governance-and-automation
description: |
  Use when: (1) Your skill repository has grown to 30+ skills and quality is
  inconsistent, (2) You need to audit skills for stale references, broken links,
  or missing evidence, (3) You want to automatically generate standardized
  summary cards for each skill, (4) You're onboarding a new agent and need a
  quick inventory of available skills. Provides: skill audit checklist (6
  quality dimensions), automated skill card template, staleness detection
  protocol, and skill lifecycle management (create → validate → maintain →
  deprecate). Inspired by NVIDIA's skill-card-generator, adapted for
  community skill repositories.
---

# Skill Governance and Automation

## Problem
As a skill repository grows (30+ skills), quality drifts:
- Descriptions become vague (poor activation triggers)
- Evidence sections go stale (cites old competition results)
- Cross-references break ([[links]] to renamed/deleted skills)
- Skills overlap or contradict each other
- No standard summary for quick inventory

## The 6-Dimension Skill Audit

For each skill, check:

### Dimension 1: Activation Trigger Quality
```
□ Description starts with "Use when: (1)..." (specific triggers)
□ At least 2 concrete trigger conditions
□ At least 1 "Do NOT use for" exclusion
□ Trigger conditions are testable (not subjective)

BAD: "Tips for Kaggle competitions"
GOOD: "Use when: (1) OOF improved but LB didn't, (2) gap > 1%"
```

### Dimension 2: Evidence Currency
```
□ Evidence section cites real numbers (not "usually helps")
□ At least 1 competition/experiment as source
□ Evidence date < 6 months old (or marked as "classic, still valid")
□ If competition-specific: competition still active or lesson still applies

STALE: "Validated on S6E4 (2026-05)" → check if still true
CURRENT: "Validated on 20+ competitions through 2026-07"
```

### Dimension 3: Solution Actionability
```
□ Solution has concrete steps (not just principles)
□ Code examples or commands provided where relevant
□ Decision trees or flowcharts for complex decisions
□ Anti-patterns section (what NOT to do + why)

VAGUE: "Be careful with ensembles"
ACTIONABLE: "If model correlation > 0.97, remove the weaker model"
```

### Dimension 4: Cross-Reference Integrity
```
□ All [[skill-name]] links point to existing skills
□ No circular dependencies (A → B → A)
□ Related skills are actually related (not just same domain)
□ No two skills cover the same scope without differentiation
```

### Dimension 5: Scope Boundaries
```
□ Clear scope: what this skill covers
□ Clear exclusions: what this skill does NOT cover
□ No overlap with another skill > 30%
□ If overlap exists: one is the "primary", other links to it
```

### Dimension 6: Format Compliance
```
□ Frontmatter: name, description present and accurate
□ Title: H1 heading matches name
□ Structure: Problem → Context → Solution → Evidence
□ Length: 50-300 lines (not a 1000-line wall, not a 5-line stub)
□ Tags: relevant search keywords
```

## Skill Card Template (Auto-Generatable)

For each skill, generate a standardized 5-line card:

```markdown
| Skill | Trigger | Key Rule | Evidence | Last Validated |
|-------|---------|----------|----------|----------------|
| autogluon-first | New tabular competition | Run AutoGluon best_quality as step 1 | House Prices CV 0.1180 | 2026-06 |
```

Automation script concept:
```python
# For each skills/examples/*/SKILL.md:
#   1. Parse frontmatter (name, description)
#   2. Extract first "Use when" from description
#   3. Extract first concrete rule from body
#   4. Extract first evidence line
#   5. Check file modification date
#   6. Output summary card
```

## Skill Lifecycle

```
CREATE:
  1. Experiment produces reusable insight
  2. Write SKILL.md following format compliance (Dimension 6)
  3. Validate: run through all 6 audit dimensions
  4. Add to README skills table
  5. Cross-reference from related skills

MAINTAIN:
  1. Quarterly: re-run 6-dimension audit
  2. After new experiments: update evidence if relevant
  3. When referenced skill changes: update cross-references
  4. When trigger conditions change: update description

DEPRECATE:
  1. Mark with ⚠️ in README
  2. Add "SUPERSEDED BY: [[new-skill-name]]" at top
  3. Keep file for historical reference (don't delete)
  4. Remove from active cross-references
```

## Staleness Detection

A skill is STALE if:
- Last evidence > 12 months old AND no "classic" marker
- Referenced competition has ended AND lesson doesn't generalize
- Referenced tool/library has major version change
- New evidence contradicts the skill's rule

Staleness action:
1. Verify: is the rule still true? (test against current state)
2. If YES → add "classic, still valid" marker + recent evidence
3. If NO → update rule or deprecate

## Evidence

- Repository grew from 0 → 36+ skills over 4 months
- Audit caught: 3 stale evidence sections, 2 broken cross-references, 1 scope overlap
- NVIDIA's skill-card-generator inspired the automated card concept
- Format evolved from ad-hoc markdown to structured frontmatter + 6 dimensions
