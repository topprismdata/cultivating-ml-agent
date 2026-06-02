---
name: memory-index-reader
description: Read memory/MEMORY.md at session start to know what context exists
type: feedback
---

# Memory Index Reader

> First action of any session: read `memory/MEMORY.md` to know what context exists.

## When to Use

**Always** at session start, before any other action.

## The Rule

**Read `memory/MEMORY.md` first thing in a session.** It's the index of all accumulated knowledge.

## Why

- Memory without an index is just files. The index tells you what's worth reading.
- 30 seconds of index reading saves 30 minutes of file searching.
- Without it, you may repeat work that's already documented.

## How to Apply

```
1. Open memory/MEMORY.md
2. Note the entries and their one-line hooks
3. Identify which entries are relevant to current task
4. Read only those (don't read everything)
5. If you create new memory, add an index entry
```

## What Goes in MEMORY.md

Each entry should be one line: `- [Title](file.md) — one-line description`

Keep it concise — long descriptions defeat the purpose.

## Anti-Patterns

- ❌ Skipping the index "because I know the project"
- ❌ Creating memory files without adding to index
- ❌ Putting detailed descriptions in the index (it's an index, not a doc)
- ❌ Reading all memory files at every session start (slow)
