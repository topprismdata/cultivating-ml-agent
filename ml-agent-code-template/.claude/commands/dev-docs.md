---
description: Create dev docs triple (plan/context/tasks) for a multi-step task
---

# /dev-docs — Create Dev Docs for a Task

> Initialize plan/context/tasks triple for a long-running task. Survives context resets.

## Usage

```
/dev-docs <task-slug> [description]
```

Example: `/dev-docs ml-improvements "Improve OOF AUC by 0.001 on TPS May"`

## What This Does

1. Creates `dev/active/<task-slug>-plan.md` — what & why
2. Creates `dev/active/<task-slug>-context.md` — decisions & history
3. Creates `dev/active/<task-slug>-tasks.md` — working checklist

Each file is loaded into context at session start if it exists, so your work survives /clear.

## Steps

1. If `$ARGUMENTS` is empty, ask the user for the task slug and short description
2. Slug must be kebab-case (lowercase, hyphens)
3. Copy from `dev/active/TEMPLATE-{plan,context,tasks}.md`
4. Replace `<task-name>` and `<description>` placeholders
5. Update the plan's Goal section with the user's intent
6. Tell the user the files are ready

## When To Use This

- Starting a multi-step task (>30 min)
- Working on something that might be interrupted
- Switching between tasks in the same session
- Anything you want to survive a context reset

## Anti-Patterns

- ❌ Don't use for single-shot tasks (just do the work)
- ❌ Don't use for trivial fixes
- ❌ Don't use without a clear goal