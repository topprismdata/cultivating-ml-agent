---
name: meta-optimize
description: Periodically analyze memory/ for staleness, contradictions, missing entries, and coverage gaps. Suggests updates.
type: learned
---

# Meta-Optimize Memory Health Check

> Periodically scan memory/ for staleness, contradictions, missing entries. **Read-only by design** — apply changes via `/meta-apply` which gates through cross-model jury.

## When to Use

- **Weekly** during long projects
- **Before** a major project phase change
- **After** a competition completes
- **When** you suspect memory is becoming stale
- **When** you can't remember if a principle still holds

## The Rule

**Memory decays. Periodically run `/meta-optimize` to identify what needs refreshing.**

Memory without maintenance becomes a graveyard of stale principles. This skill catches that.

## How to Apply

```
1. Run: bash .claude/hooks/meta_optimize.sh
2. Review the 7 sections:
   - Staleness (> 90 days untouched)
   - Oversized files (> 500 lines)
   - Missing index entries
   - Possible contradictions
   - Coverage gaps (experiment topics not extracted)
   - Stale dead-ends (> 365 days)
   - Broken internal links
3. For each issue, decide: refresh, archive, or remove
4. To apply changes, use /meta-apply (cross-model jury required)
```

## The 7 Sections Explained

| Section | Catches | Action |
|---------|---------|--------|
| Staleness | Files not touched in 90+ days | Refresh or remove |
| Oversized | Files > 500 lines (violates 500-line rule) | Split into resources/ |
| Missing index | Files not in MEMORY.md | Add to index |
| Contradictions | Principles with "never/don't/forbidden" | Verify still valid |
| Coverage gaps | Breakthrough experiments not extracted | Extract to skill |
| Stale dead-ends | Confirmed failures > 365 days | May be obsolete (re-test or remove) |
| Broken links | Markdown links to non-existent files | Fix or remove |

## Read-Only by Design

`meta_optimize.sh` is **read-only**. It reports but never modifies. This is intentional (ARIS design):

> "Self-evolution layer (`/meta-optimize`): analyzes logs and proposes SKILL.md patches; now read-only with landing gated by cross-model jury via the new `/meta-apply` skill"

Why: agents modifying their own memory can reinforce errors. External review (cross-model jury) catches bad updates.

## Anti-Patterns

- ❌ Modifying memory directly from `/meta-optimize` output (use `/meta-apply`)
- ❌ Running weekly on tiny memory (< 10 files) — overhead exceeds value
- ❌ Ignoring stale entries "because they're old" — old ≠ wrong, but they need re-verification
- ❌ Bulk-accepting all suggestions without review

## Empirical Evidence

**ARIS (wanshuiyin, 11.1k stars)** uses this exact pattern:
- `/meta-optimize` analyzes logs and proposes SKILL.md patches
- Read-only with landing gated by cross-model jury
- New `/meta-apply` skill applies changes after review
- Provenance tracking via `tools/provenance.py` — "author-family ≠ reviewer-family enforced on the stamp"

## Output Example

```
═══ Staleness (> 90 days untouched) ═══
  memory/skills/cv-strategy.md — 145 days

═══ Oversized Files (> 500 lines) ═══
  memory/principles/16-principles.md — 587 lines (split into resources/)

═══ Coverage Gaps ═══
  exp_2026-04-15_f27-breakthrough — Breakthrough but no skill extracted

═══ Summary ═══
  Stale (>90d):      1
  Oversized (>500L): 1
  Coverage gaps:     1
  → To apply changes, use /meta-apply (cross-model review required)
```
