---
description: Apply /meta-optimize suggestions after cross-model jury approval. Gates memory writes through external review.
---

# /meta-apply — Apply Memory Updates

> **Gated memory writes.** Sends proposed changes to cross-model jury (agy/gemini/codex/ollama) for approval. Only lands changes the jury validates.

## Usage

```
/meta-apply                    # Interactive: review each suggestion
/meta-apply --all              # Apply all after jury approval
/meta-apply <file>             # Apply to specific file
```

## The Flow

```
1. /meta-optimize runs analysis
2. You review the report
3. /meta-apply proposes changes
4. Each change is sent to cross-model jury (agy)
5. Jury says: APPROVE / REJECT / REVISE
6. Only APPROVED changes land
7. Failed changes get logged to memory/meta-apply-rejected.md
```

## Why Jury Gating

From ARIS design:
> "Self-evolution layer (`/meta-optimize`): analyzes logs and proposes SKILL.md patches; now read-only with **landing gated by cross-model jury** via the new `/meta-apply` skill"

Without a jury:
- Agent reinforces its own errors
- Stale principles never get challenged
- Conflicts slip through

## When to Use

- After `/meta-optimize` reports issues you agree with
- When you want to update skills based on new evidence
- When refreshing dead-end entries

## When NOT to Use

- During an active competition (memory changes are noise)
- On principles you just wrote (let them age first)
- If the changes are trivial formatting (overhead)

## Jury Output Format

The jury receives:
```
PROPOSED CHANGE: <file>
<before>
<after>
REASON: <why this change>
```

And returns one of:
- **APPROVE** — apply as-is
- **REJECT** — don't apply, here's why
- **REVISE** — here's a better version

## Provenance Tracking

Every approved change is logged to `memory/meta-apply-log.md`:
```markdown
## 2026-06-01
- File: memory/skills/500-line-rule.md
- Change: added example
- Author: meta-optimize
- Jury: agy (gemini-1.5-pro)
- Status: APPROVED
```

This ensures auditability: "who changed what when".

## Anti-Patterns

- ❌ `--all` without reading the report (rubber-stamping)
- ❌ Ignoring REJECT verdicts (defeats purpose)
- ❌ Skipping provenance tracking
- ❌ Updating principles you just wrote (insufficient reflection time)
