---
description: Run the Grader agent on the current submission for a competition
---

# /grade — Validate and Grade Submission

> Run the Grader agent on `submissions/<competition>/submission.csv`.

## Usage

```
/grade <competition-slug>
```

Example: `/grade tps-may-2022`

## What This Does

1. Confirms `submissions/<competition>/submission.csv` exists
2. Validates format (header, row count, no empty cells)
3. Runs `bash .claude/hooks/grade_submission.sh <slug>` for full validation
4. If mlebench is installed, runs `mlebench grade` and reports score
5. Emits PASS / WARN / FAIL verdict
6. Suggests next steps

## Team Pattern

This command invokes the **Grader agent** role, which is the read-only validator in the Builder/Grader team pattern. Builder produces the file; Grader validates it.

## When To Use

- After Builder agent finishes producing submission
- Before manually submitting to leaderboard
- After any change to the submission
- When debugging a grade failure

## Anti-Patterns

- ❌ Don't run mlebench grade repeatedly on the same file (wastes quota)
- ❌ Don't modify the submission from this command (Grader is read-only)
- ❌ Don't ignore WARN (format OK but score below expectation)
