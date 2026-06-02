# Changelog

## 2026-06-01 — v0.4.1: Full cycle test passed + 3 bugs found and fixed

### Test Results (~/ml-agent-test-cycle, aerial-cactus-identification)

All 7 phases passed end-to-end. Found and fixed 3 real bugs:

| Bug | Symptom | Fix |
|-----|---------|-----|
| `meta_optimize.sh` coverage gap false positive | "Dead end" experiments flagged as breakthrough | Tighten regex to `^\s*-?\s*\[x\]\s*\*?\*?breakthrough` |
| `grade_submission.sh` wrong mlebench CLI | Used `mlebench grade` (JSONL) instead of `mlebench grade-sample` (CSV) | Switch to `grade-sample` with positional args |
| Bash `[ "$VAR" ]` evaluates non-empty string as true | `FIRST_PROMPT=false` always passed the test (caught in `skill_activation_hook.sh` earlier) | Use `[ "$VAR" = "true" ]` |

### 7-Phase Test

1. **SessionStart** — loaded dev docs triple (3 done, 5 pending)
2. **First prompt** — auto-activated `check-dead-ends` + `feature-engineering-roi`
3. **Real experiment** — color histogram features got OOF AUC 0.9758 (worse than 0.9978 baseline)
4. **Stuck prompt** — auto-activated `local-optimum-trap` (priority 9)
5. **Submit trigger** — fired cross-review reminder with agy
6. **Grader** — validated format, ran `mlebench grade-sample`, returned score 0.5 (no medal)
7. **Meta-optimize** — reported 1 missing index, 2 false-positive contradictions, 0 coverage gaps

### Verified

- All 9 hooks execute at expected events
- Auto-activation suggests correct skills (priority ordering works)
- Cross-review correctly fires on submit, ignores questions
- Grader correctly handles the actual mlebench CLI signature
- Meta-optimize correctly identifies missing index entries (the new color_hist experiment)
- Local-optimum-trap activates on tuning prompts after failed experiments

## 2026-06-01 — v0.4.0: meta-optimize + Antigravity integration tested

### Added

**P2 — meta-optimize** (ARIS self-evolution pattern)
- `.claude/hooks/meta_optimize.sh` — read-only analyzer, 7 sections
- `.claude/skills/meta-optimize.md` — usage skill
- `.claude/commands/meta-optimize.md` — `/meta-optimize` command
- `.claude/commands/meta-apply.md` — `/meta-apply` command (jury-gated)
- 7 sections: staleness, oversized, missing index, contradictions, coverage gaps, stale dead-ends, broken links

**Cross-review integration test**: agy real invocation verified
- `cross_review.sh` invoked with TPS May 2022 CatBoost question
- agy returned 5-section critique (CLAIMS, RISKS, GAPS, COUNTER-EVIDENCE, VERDICT)
- Cited AmbrosM benchmark, identified f_27 positional gap, gave conditional verdict
- Validates the full critical-reviewer prompt template

### Fixed

- MEMORY.md index missing entry: `skills/500-line-rule.md` added
- Re-ran meta_optimize.sh: 0 missing index entries (was 1)

### Verified

- `meta_optimize.sh` runs cleanly: 0 stale, 0 oversized, 0 missing, 2 false-positive contradictions
- `cross_review.sh` end-to-end with agy (1.0.3) — high-quality critique
- `cross_review_trigger.sh` correctly fires on submit/push/grade prompts, ignores questions

## 2026-05-31 — v0.3.0: P2 Cross-Model Review

Added cross-model review (ARIS-inspired) using Antigravity CLI as default backend.

### Added

**P2 — Cross-model review** (from ARIS)
- `.claude/skills/codex-review.md` — generic skill, supports agy/gemini/codex/ollama
- `.claude/commands/review.md` — `/review <topic>` slash command
- `.claude/hooks/cross_review.sh` — auto-detect backend, invoke with critical-reviewer prompt
- `.claude/hooks/cross_review_trigger.sh` — UserPromptSubmit hook that detects high-stakes operations (submit, mlebench grade, git push, merge to main, etc.) and reminds to use /review first
- Adversarial self-check rubric (5 questions) for when no LLM CLI is available

**Backend priority** (auto-detect):
1. `agy` (Antigravity CLI) — preferred, default
2. `gemini` (Google Gemini CLI)
3. `codex` (OpenAI Codex CLI)
4. `ollama` (local)

### Installed (this session)

- `brew install gemini-cli` (0.44.1) — installed successfully after Node upgrade
- `agy` (1.0.3) — manually installed by user to `~/.local/bin/agy`

### Note on disk space

`brew install gemini-cli` consumed significant disk. The 100MB+ gemini-cli bottle and 331MB cleanup pushed toward 100% disk usage. **Recommend**: `brew cleanup` periodically, or set `HOMEBREW_NO_INSTALL_CLEANUP=1`.

## 2026-05-31 — v0.2.0: Borrowed Features from GitHub

Adopted P0 + P1 features from comparison with top Claude Code templates.

### Added

**P0 — Auto-activation** (from diet103)
- `.claude/skill-rules.json` — declarative pattern→skill mappings (7 rules, priority-ordered)
- `.claude/hooks/skill_activation_hook.sh` — UserPromptSubmit hook that:
  - Auto-injects `check-dead-ends` reminder for any "new direction" prompt
  - Reads skill-rules.json and suggests top-3 matching skills
  - Reminds to read `memory/MEMORY.md` on first prompt of session
  - 24h session-marker for first-prompt detection

**P0 — Installation** 
- `SETUP.sh` updated: now installs hooks to `~/.claude/hooks/`, merges with `~/.claude/settings.json` (with timestamped backup), copies skills to `~/.claude/skills/`

**P1 — Dev docs pattern** (from diet103)
- `dev/active/TEMPLATE-{plan,context,tasks}.md` — three-file pattern for long-running tasks
- `.claude/commands/dev-docs.md` — `/dev-docs <slug>` slash command
- `.claude/commands/dev-docs-update.md` — `/dev-docs-update` slash command
- `.claude/hooks/session_start_hook.sh` — auto-loads active dev docs at session start
- `dev/README.md` — usage docs

**P1 — 500-line rule** (from diet103)
- `.claude/hooks/skill_size_check.sh` — PostToolUse hook:
  - Warns at > 400 lines (soft)
  - Blocks at > 500 lines (hard), suggests `SKILL.md` + `resources/` split
- `memory/skills/500-line-rule.md` — rationale and pattern documentation

**P1 — Team pattern** (from disler)
- `.claude/agents/builder.md` — Builder role: produces `submission.csv`
- `.claude/agents/grader.md` — Grader role: validates format + runs `mlebench grade`
- `.claude/hooks/grade_submission.sh` — full validation: format, sanity, mlebench
- `.claude/commands/grade.md` — `/grade <slug>` slash command

**P1 — Memory index reader** (skill)
- `.claude/skills/memory-index-reader.md` — auto-activate on first prompt of session

### Updated

- `COMPARISON.md` — marked P0/P1 items as DONE
- `README.md` — references new commands and hooks

## 2026-05-31 — v0.1.0: Initial Template

- Base template with hooks, rules, memory, skills
- COMPARISON.md (research)
- ML Agent SOP (5 rules, 4 phases, 8 anti-patterns)
- MLE-Bench STRATEGY.md whitepaper
- 16 cross-domain principles
- 4 skills: feature-engineering-roi, stacking-ceiling, cv-strategy, external-data-fusion, check-dead-ends

---

## Roadmap (Remaining)

| Priority | Feature | Source | Status |
|----------|---------|--------|--------|
| 🟢 P3 | TTS feedback / status lines | disler | ❌ Skipped (not aligned with long ML sessions) |
| 🟢 P3 | Plugin marketplace format | Anthropic | ⏳ Future |
| 🟢 P3 | MCP server templates | various | ⏳ Future |
| ✅ Done | Cross-model review for new experiments | ARIS | ✅ **v0.3.0** |
| ✅ Done | meta-optimize self-evolution (read-only) | ARIS | ✅ **v0.4.0** |
| ✅ Done | meta-apply jury-gated landing | ARIS | ✅ **v0.4.0** |
