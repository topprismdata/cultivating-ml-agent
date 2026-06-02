# ML Agent Code Template

A reusable Claude Code ML agent project template with hooks, rules, skills, and auto-activation.

> **Borrowed from**: diet103/claude-code-infrastructure-showcase (9.7k ⭐), wanshuiyin/ARIS (11.1k ⭐), disler/claude-code-hooks-mastery
> **Validated by**: 6 Gold + 2 Silver MLE-Bench competition results

## Quick Start

```bash
git clone <this-repo> my-ml-project
cd my-ml-project
bash SETUP.sh              # Installs venv, hooks, skills
claude                     # Start a session
```

The SETUP script will:
1. Create a Python venv with numpy, pandas, sklearn, lightgbm, scipy
2. Copy hooks to `~/.claude/hooks/` (with timestamped backup of existing)
3. Merge hook config into `~/.claude/settings.json` (idempotent)
4. Copy skills to `~/.claude/skills/`

## Features

### Auto-Activation (P0)

Patterns in your prompt auto-suggest relevant skills. Try:

```
"let me try a new feature engineering approach"
"should I add external data?"
"what's the best CV strategy?"
```

You'll see a `💡 Activate skill: <name>` reminder in the response context.

Configuration: `.claude/skill-rules.json`

### Hooks (Auto-Run)

| Event | Hook | Purpose |
|-------|------|---------|
| PreToolUse (Bash) | `git_risk_hook.sh` | Block destructive commands |
| UserPromptSubmit | `skill_activation_hook.sh` | Auto-suggest skills |
| UserPromptSubmit | `cross_review_trigger.sh` | Remind to use /review on submit/push |
| PostToolUse | `skill_size_check.sh` | 500-line rule for skills |
| SessionStart | `session_start_hook.sh` | Load active dev docs |
| Stop | `stop_audit.sh` | Scan for console.log/secrets |

### Dev Docs Pattern (P1)

For multi-step tasks, `/dev-docs <slug>` creates a triple:
- `<slug>-plan.md` — what & why
- `<slug>-context.md` — decisions & history
- `<slug>-tasks.md` — working checklist

Survives `/clear` and session resets. Auto-loaded at session start.

### Team Pattern: Builder + Grader (P1)

- **Builder** (`.claude/agents/builder.md`) — produces `submissions/<slug>/submission.csv`
- **Grader** (`.claude/agents/grader.md`) — validates format + runs `mlebench grade`
- `/grade <slug>` slash command runs the Grader

Read-only Grader prevents wasted `mlebench grade` quota.

### Memory System

- `MEMORY.md` — index of all accumulated knowledge
- `principles/16-principles.md` — cross-domain ML principles
- `skills/` — reusable techniques (4 included)
- `experiments/` — per-experiment logs
- `competitions/` — per-competition retrospectives
- `feedback_no_recheck_confirmed_dead.md` — known dead ends

### Commands (Slash)

| Command | Purpose |
|---------|---------|
| `/dev-docs <slug>` | Create dev docs triple |
| `/dev-docs-update` | Refresh dev docs before /clear |
| `/grade <slug>` | Run Grader on current submission |
| `/review <topic>` | Cross-model review via Antigravity/Gemini/Codex/Ollama |

## Project Structure

```
my-ml-project/
├── .claude/
│   ├── CLAUDE.md            # Global instructions
│   ├── skill-rules.json     # Auto-activation rules
│   ├── hooks/               # 5 hooks (install via SETUP.sh)
│   ├── rules/               # 9 rules (coding, git, testing, etc.)
│   ├── skills/              # Custom skills (check-dead-ends, etc.)
│   ├── agents/              # Builder, Grader
│   └── commands/            # /dev-docs, /grade
├── memory/
│   ├── MEMORY.md            # Index
│   ├── principles/          # 16 principles
│   ├── skills/              # 5 skills
│   ├── experiments/         # Per-experiment logs
│   ├── competitions/        # Per-competition logs
│   └── feedback_no_recheck_confirmed_dead.md
├── dev/
│   ├── active/              # Active dev docs (auto-loaded)
│   └── archive/             # Completed triples
├── submissions/             # Output submissions
├── logs/                    # Experiment logs
├── SETUP.sh                 # One-time installer
├── COMPARISON.md            # Research: vs similar GitHub projects
└── CHANGELOG.md             # Version history
```

## Customization

1. **CLAUDE.md** — add project-specific rules (data paths, competition names, key parameters)
2. **skill-rules.json** — add patterns for your domain
3. **memory/principles/** — add project-specific patterns
4. **memory/skills/** — add reusable techniques

## Comparison vs Similar Projects

See `COMPARISON.md` for detailed comparison with 8 top Claude Code template repos.

**Our unique value**: Only template with 6 Gold + 2 Silver MLE-Bench results validated.

## Versioning

See `CHANGELOG.md`.

## License

MIT. Based on MLE-Bench agent experiments.
