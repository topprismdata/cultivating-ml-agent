# Comparison With Similar GitHub Projects

> Researched 2026-05-31. Comparison against top Claude Code template repos.

## The Landscape (top projects)

| Repo | Stars | Focus | ML/Kaggle? |
|------|-------|-------|-----------|
| hesreallyhim/awesome-claude-code | 45.4k | Curated list | No (directory) |
| **wanshuiyin/ARIS** | **11.1k** | Autonomous ML research | Yes (research papers, not Kaggle) |
| diet103/claude-code-infrastructure-showcase | 9.7k | Auto-activation + dev docs | No (TypeScript microservices) |
| ChrisWiles/claude-code-showcase | 5.9k | Full-stack with GitHub Actions | No |
| disler/claude-code-hooks-mastery | 3.7k | All 13 hook events | No (educational) |
| rohitg00/awesome-claude-code-toolkit | 1.9k | 20 hooks + 135 agents + 42 commands | No |
| **Ashfaqbs/software-dev-ai-claude-toolkit** | 17 | 9 rules + 8 cmds + 5 agents + 13 skills + 4 hooks + 4 MCP | No (full-stack) |
| muhammad-bu/claude-code-unified-agents | 6 | 54 agents with orchestration | Some AI/ML |
| **Our template (this repo)** | — | **Kaggle/ML competition agent** | **Yes (8 competitions validated)** |

## Detailed Comparison

### 1. ARIS (Auto-Research-In-Sleep)

**What they have:**
- 77 skills + 54 helpers (v0.4.15)
- Cross-model review loop (Claude + external critic)
- Research Wiki with provenance tracking
- Self-evolution via `/meta-optimize` and `/meta-apply`
- Anti-self-poisoning capture filter

**What we have that they don't:**
- **Kaggle competition focus** — they automate research papers, we automate competition pipelines
- **8 competitions validated** with concrete gold/silver scores
- **MLE-Bench domain knowledge** (data paths, grading, submission format)
- **Confirmed dead ends table** — 25+ entries with evidence
- **Stacking ceiling formula** — empirically derived from TPS May 2022

**What they have that we don't:**
- Cross-model review loop (uses external LLM as critic)
- Self-evolution (`/meta-optimize` analyzes logs and proposes SKILL.md patches)
- Research Wiki (more sophisticated than our flat markdown)
- Anti-self-poisoning filter (prevents the agent from reinforcing its own errors)

**Synergy opportunities:**
- Adopt cross-model review for new competition experiments (use Codex/Gemini to critique our approach)
- Adopt `/meta-optimize` to suggest memory updates after a run
- Borrow anti-self-poisoning: when extracting principles, require 2+ independent experiments to confirm

### 2. diet103/claude-code-infrastructure-showcase

**What they have:**
- Auto-activation via `skill-rules.json` (skills activate on prompt patterns)
- 5 production skills with modular `resources/` directories
- 500-line rule (each file < 500 lines, progressive disclosure)
- Dev docs pattern: `[task]-plan.md`, `[task]-context.md`, `[task]-tasks.md`
- 10 specialized agents

**What we have that they don't:**
- ML/Kaggle domain (they're TypeScript microservices)
- Empirical validation (8 competition results)
- Domain-specific memory (experiments, competitions, dead-ends structure)
- Anti-pattern documentation with evidence

**What they have that we don't:**
- **Auto-activation** (UserPromptSubmit hook reads prompt and suggests skills) — high ROI feature
- **Modular skills with resources/** — 500-line rule prevents context bloat
- **Dev docs pattern** — survives context resets; our memory structure serves this but less formally
- **Skill-rules.json** — declarative way to map prompt patterns to skills

**Synergy opportunities:**
- Add `skill-rules.json` to auto-suggest `check-dead-ends` skill on experiment-start prompts
- Adopt 500-line rule for new skills
- Adopt dev docs pattern: `dev/active/<task>-{plan,context,tasks}.md` for ongoing work

### 3. Ashfaqbs/software-dev-ai-claude-toolkit

**What they have:**
- 9 rules (incl. java-springboot, python-backend-ai, postgres, kafka)
- 8 slash commands (`/plan`, `/code-review`, `/tdd`, `/build-fix`, etc.)
- 5 agents (planner, code-reviewer, security-reviewer, architect, tdd-guide)
- 13 skills (incl. springboot-patterns, jpa-patterns, eval-harness)
- 4 hooks (pre/post tool use, dangerous command blocking)
- 4 MCP servers (context7, memory, sequential-thinking, github)
- Installer script that merges with existing `~/.claude/`

**What we have that they don't:**
- ML/Kaggle domain expertise
- 8 competition-validated results
- Empirical principle documentation
- Confirmed dead ends with experiment evidence

**What they have that we don't:**
- **Installer script** that intelligently merges with existing config (we have SETUP.sh but it doesn't merge)
- **Multiple domain rules** (Spring Boot, React, etc.) — we focus on ML
- **MCP server integration** in template (we leave it to user)
- **Standardized slash commands** (`/plan`, `/code-review`, `/tdd`) — we don't define these

**Synergy opportunities:**
- Adopt the installer pattern: backup existing `~/.claude/`, then merge rules/hooks
- Add standard `/plan`, `/code-review`, `/tdd` slash commands
- Document which MCP servers to add (memory, sequential-thinking)

### 4. disler/claude-code-hooks-mastery

**What they have:**
- All 13 hook events covered (single-file Python scripts)
- 9 status-line variants
- 8 output styles
- TTS feedback with priority queue
- Security gating examples
- Team-based builder/validator pattern

**What we have that they don't:**
- ML/Kaggle domain
- Empirically validated skills/principles
- Memory/knowledge management

**What they have that we don't:**
- **Full hook coverage** — they show all 13 events; we use 3 (PreToolUse, PostToolUse, Stop)
- **Status lines** — useful for at-a-glance agent state
- **TTS feedback** — accessibility/UX feature
- **Team pattern** (builder + validator) — applicable to ML: builder creates submission, validator runs `mlebench grade`

**Synergy opportunities:**
- Cover more hook events: UserPromptSubmit (for skill auto-activation), Notification, PreCompact
- Adopt status-line for ML context: showing current competition, OOF score, time elapsed
- Add team pattern: agent-1 builds submission, agent-2 runs `mlebench grade` for validation

## Our Unique Value

1. **Only Kaggle-focused template** with 6 Gold + 2 Silver results
2. **Empirically derived principles** — every principle has experiment evidence
3. **Confirmed dead ends** — saves hours of redundant work
4. **Stacking ceiling formula** — actionable diagnostic
5. **Three-track memory structure** — experiments, competitions, principles, skills
6. **Production-tested hooks** — git_risk and stop_audit actually run and save sessions

## What We Should Adopt (Priority Order)

| Priority | Feature | Source | Effort | Impact | Status |
|----------|---------|--------|--------|--------|--------|
| 🔴 P0 | Auto-activation via `skill-rules.json` | diet103 | Medium | High | ✅ **DONE** |
| 🔴 P0 | `check-dead-ends` runs on every prompt start | us + diet103 | Low | High | ✅ **DONE** |
| 🔴 P0 | SETUP.sh installs hooks to ~/.claude/ | us | Medium | High | ✅ **DONE** |
| 🟡 P1 | 500-line rule for new skills | diet103 | Low | Medium | ✅ **DONE** |
| 🟡 P1 | Dev docs pattern for long-running tasks | diet103 | Low | Medium | ✅ **DONE** |
| 🟡 P1 | Team pattern: builder + grader | disler | Low | Medium | ✅ **DONE** |
| 🟢 P2 | Cross-model review for new experiments | ARIS | Medium | High | ✅ **v0.3.0** |
| 🟢 P2 | High-stakes trigger (review before submit/push) | ARIS | Low | High | ✅ **v0.3.0** |
| 🟢 P2 | `/meta-optimize` to suggest memory updates | ARIS | High | Medium | ⏳ Future |
| 🟢 P2 | TTS feedback / status lines | disler | Low | Low | ❌ Skipped (not aligned with long sessions) |

## What We Should NOT Adopt

- ❌ TypeScript/full-stack rules (we're ML-only)
- ❌ TTS feedback (ML sessions are long, audio would be annoying)
- ❌ 100+ agents (overhead, not value; ARIS's 77 is the right ceiling)
- ❌ Research-paper focus (we're competition-focused)

## Sources

- [hesreallyhim/awesome-claude-code](https://github.com/hesreallyhim/awesome-claude-code) — Curated list (45.4k stars)
- [wanshuiyin/ARIS](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) — ML research automation (11.1k stars)
- [diet103/claude-code-infrastructure-showcase](https://github.com/diet103/claude-code-infrastructure-showcase) — Auto-activation showcase (9.7k stars)
- [ChrisWiles/claude-code-showcase](https://github.com/ChrisWiles/claude-code-showcase) — Full-stack with GH Actions
- [disler/claude-code-hooks-mastery](https://github.com/disler/claude-code-hooks-mastery) — Hook tutorial (3.7k stars)
- [rohitg00/awesome-claude-code-toolkit](https://github.com/rohitg00/awesome-claude-code-toolkit) — 20 hooks + 135 agents
- [Ashfaqbs/software-dev-ai-claude-toolkit](https://github.com/Ashfaqbs/software-dev-ai-claude-toolkit) — Structured full-stack config
- [muhammad-bu/claude-code-unified-agents](https://github.com/muhammad-bu/claude-code-unified-agents) — 54 agents with orchestration