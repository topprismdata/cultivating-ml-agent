---
name: agent-session-memory-protocol
description: |
  Use when: (1) Starting a long-running experiment that may span multiple
  sessions/disconnects, (2) Resuming work after a break and needing to recover
  context, (3) Handing off to another agent or session, (4) You realize you've
  lost track of what was tried and why. Provides a structured 4-file session
  memory system that captures goal, timeline, files touched, and handoff
  instructions — lightweight (pure markdown, no infrastructure), works alongside
  AutoMem or standalone. Inspired by NVIDIA's nemo-rl-session-memory, adapted
  for Kaggle competition workflows.
---

# Agent Session Memory Protocol

## Problem
Long-running Kaggle experiments span hours to days. Between sessions:
- Agent forgets what was tried and why
- Submissions and their OOF/LB pairs get lost
- Dead ends get repeated
- Handoff to a new session loses critical context

## The 4-File Session Memory Structure

Create a session directory per experiment session:

```
~/.claude/projects/<project>/sessions/
└── <YYYYMMDD_HHMMSS>/
    ├── session_state.md    # Goal, subtask, status, plan, blockers, next actions
    ├── timeline.md         # Append-only log of actions, commands, results
    ├── experiments.md      # Structured table of all submissions/attempts
    └── handoff.md          # Concise resume instructions for next session
```

### File 1: `session_state.md`

```markdown
# Session State
- Session: 20260708_140000
- Project: pokemon-tcg-ai-battle
- Started: 2026-07-08 14:00
- Updated: 2026-07-08 18:30

## Goal
Optimize PTCG agent to reach top 200 on the leaderboard.

## Current Subtask
Waiting for Nithin A/B convergence after 07-06 re-submit (μ₀=600 reset).

## Status
- Nithin A: LB 869 (converging from 600, +17/day)
- Nithin B: LB 772 (converging from 600, +2.5/day)
- Latest-2 = Nithin A + Nithin B (both in final-2 slots)

## Plan
1. WAIT — do not re-submit (resets convergence)
2. Check LB daily until ~Jul 20 (expected ~950/910)
3. Aug 10-14: submit 2 copies of best agent (high-roll)

## Blockers
- GPU quota exhausted (30h/week) — blocks Biohub, Industrial T1, ARC-AGI-3

## Next Actions
- [ ] Jul 13: Check PTCG LB (expected Nithin A ~920)
- [ ] Jul 20: Check PTCG LB (expected Nithin A ~950)
- [ ] Aug 10: Begin high-roll duplicate submissions
```

### File 2: `timeline.md`

```markdown
# Timeline (append-only)

## 2026-07-08 14:00 — Session Start
- Recalled AutoMem: 109 memories, PTCG status, ROGII Pipeline A

## 14:15 — PTCG Check
- Nithin A: 869.2 (+17 from yesterday)
- Nithin B: 772.3 (+2.5)
- Decision: WAIT, no re-submit

## 14:30 — ROGII Pipeline A
- Pushed rogii-pipea-only kernel (Pipeline B trimmed)
- Kernel COMPLETE, submission.csv written (14152 rows)
- Manual Submit to Competition required

## 15:00 — ROGII Score
- LB: 19545.46 (TERRIBLE — predicted values include -27561 outliers)
- Root cause: Pipeline A predictions have negative TVT values for some rows
- Old v7 (13.82) still better than Pipeline A on LB
```

### File 3: `experiments.md`

```markdown
# Experiment Log

| # | Timestamp | Competition | Change | OOF | LB | Verdict |
|---|-----------|-------------|--------|-----|-----|---------|
| 1 | 07-02 09:30 | PTCG | Nithin A (Archaludon) submit | — | 706.9 | baseline |
| 2 | 07-02 09:35 | PTCG | Nithin B (Alakazam) submit | — | 600.0 | baseline |
| 3 | 07-04 12:21 | PTCG | Pilkwang B (Crustle) submit | — | 707.9 | converged low |
| 4 | 07-06 15:38 | PTCG | Nithin A re-submit (final-2) | — | 600→869 | converging |
| 5 | 07-09 00:51 | ROGII | Pipeline A (ourmatch fork) | 10.38 | 19545 | OUTLIERS |
```

### File 4: `handoff.md`

```markdown
# Handoff — Resume Instructions

## TL;DR
PTCG agents converging (wait). ROGII Pipeline A has outlier bug (negative TVT).
NeuroGolf baseline 7228 submitted. s6e7 at 0.94942 ceiling.

## If Resuming PTCG
- DO NOT re-submit (read trueskill-simulation-competition-strategy)
- Check: kaggle competitions submissions pokemon-tcg-ai-battle
- Next action: wait until Jul 20, then evaluate convergence

## If Resuming ROGII
- Pipeline A produces negative TVT values → need to clamp to [0, ∞) or investigate
- v7 (13.82) is still better than Pipeline A (19545)
- Consider: clip predictions to reasonable TVT range

## If Resuming NeuroGolf
- Baseline 7228 submitted (rank 485)
- Sparse optimization failed (Conv doesn't support sparse_initializer)
- Next: per-task ONNX design for linearly-separable rules
```

## Checkpoint Rhythm

Write a checkpoint (update all 4 files):
- **After gathering enough context to form a plan** (start of session)
- **Before and after each submission** (record OOF/LB pair)
- **Before long-running commands** (kernel push, model training)
- **When the user changes direction** (new competition, new strategy)
- **At least every 30 minutes** during active work
- **Before ending the session** (write handoff.md)

## vs AutoMem

| | Session Memory (this skill) | AutoMem |
|---|---|---|
| Storage | Local markdown files | FalkorDB + Qdrant |
| Setup cost | Zero | Docker stack required |
| Retrieval | Read files directly | Semantic search via API |
| Best for | Current session context | Cross-session knowledge base |
| Relationship | **Complementary** — session memory for "what I'm doing now", AutoMem for "what I've learned" |

**Recommended**: Use BOTH. Session memory for active work tracking, AutoMem for crystallized knowledge.

## Session Lifecycle

```
START:
  1. Check for existing session directories: ls sessions/
  2. If resuming: read latest session_state.md + handoff.md
  3. If new: create timestamped directory, write initial session_state.md
  4. Recall AutoMem for relevant cross-session knowledge

DURING:
  5. Append to timeline.md after each significant action
  6. Update experiments.md after each submission
  7. Update session_state.md when plan/status changes
  8. Checkpoint every 30 min

END:
  9. Write handoff.md with concise resume instructions
  10. Store key learnings to AutoMem (crystallize)
  11. Verify all 4 files are saved
```

## Evidence

Inspired by NVIDIA's `nemo-rl-session-memory` skill, adapted for Kaggle workflows:
- NVIDIA original: 4 files (session_state, timeline, files, handoff) + 30min checkpoints
- Our adaptation: adds `experiments.md` (structured submission log), integrates with AutoMem
- Validated: PTCG (8+ submissions tracked), ROGII (pipeline debugging), NeuroGolf (ONNX analysis)
