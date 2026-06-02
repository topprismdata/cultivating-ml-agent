---
description: Analyze memory/ for staleness, contradictions, missing entries. Read-only.
---

# /meta-optimize — Memory Health Check

> Read-only analysis of memory/. Reports issues. Does not modify anything.

## Usage

```
/meta-optimize
/meta-optimize --json
```

## What This Does

Runs `bash .claude/hooks/meta_optimize.sh` which scans `memory/` and reports:

1. **Staleness** — files not modified in 90+ days
2. **Oversized files** — > 500 lines (violates 500-line rule)
3. **Missing index entries** — files not in MEMORY.md
4. **Possible contradictions** — principles with conflicting language
5. **Coverage gaps** — breakthrough experiments without extracted skills
6. **Stale dead-ends** — feedback entries > 365 days old
7. **Broken internal links** — markdown links to non-existent files

## Output

Human-readable report. For machine-readable output, use `--json`.

## Workflow After Running

```
1. /meta-optimize           # see issues
2. For each issue, decide: refresh / archive / remove
3. /meta-apply <changes>    # cross-model jury required for landing
```

## Why Read-Only

This skill **never modifies memory directly**. From ARIS design:
> "self-evolution is read-only with landing gated by cross-model jury"

Agents modifying their own memory can reinforce errors. External review catches this.

## When to Use

- **Weekly** during long projects
- **Before** major phase changes
- **After** a competition completes
- **When** you suspect memory is becoming stale

## When NOT to Use

- Daily (overhead)
- On tiny memory (< 10 files)
- When actively debugging a problem (not the time)
