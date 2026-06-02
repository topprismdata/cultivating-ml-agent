---
description: Update the dev docs triple for the current task
---

# /dev-docs-update — Refresh Dev Docs

> Append latest progress to dev/active/<task>-tasks.md. Call this before /clear or session end.

## Usage

```
/dev-docs-update [task-slug]
```

If no slug given, use the most recently modified triple in dev/active/.

## What This Does

1. Find the active task triple (by slug or recency)
2. Append a "Session summary" section to context.md with:
   - What was done this session
   - Decisions made
   - Open questions
3. Mark completed items in tasks.md
4. Confirm to the user

## When To Use This

- Before `/clear` (preserve context)
- Before ending a long session
- After a breakthrough (capture reasoning)
- After a dead end (so future sessions know)