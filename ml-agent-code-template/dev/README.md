# dev/active/ — Dev Docs for Active Work

> Survives context resets. Loaded into context at session start.

## What is this?

Dev docs is a triple-file pattern for multi-step tasks:

- `<slug>-plan.md` — what & why (goal, approach, acceptance criteria)
- `<slug>-context.md` — decisions & history (why code looks the way it does)
- `<slug>-tasks.md` — working checklist (in progress, done, blocked)

## Usage

1. **Start a task**: `/dev-docs <slug> <description>`
2. **During work**: edit tasks.md as you go
3. **Before /clear or session end**: `/dev-docs-update`
4. **Resume later**: I auto-load active triples at session start

## Naming

- Use kebab-case: `ml-improvements`, `denoising-postproc`
- One triple per task
- Delete the triple when the task is done (move to `dev/archive/` if you want to keep history)

## Anti-Patterns

- ❌ Don't create a triple for every tiny fix
- ❌ Don't let the triple grow to 1000+ lines (split if needed)
- ❌ Don't skip the context.md (decisions matter for future-you)