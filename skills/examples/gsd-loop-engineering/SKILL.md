---
name: gsd-loop-engineering
description: |
  Apply GSD Core's loop-engineering methodology to ML/data-science tasks.
  Use when: (1) You have a multi-step ML pipeline (data → features → train → verify → submit),
  (2) Context is growing long and quality is drifting, (3) You want auditable
  verification at each step, (4) You want fresh-context subagents to handle heavy
  work without polluting the main session. Validated against: gsd-build/get-shit-done
  (64K stars, deprecated), open-gsd/gsd-core (canonical, 4.4K stars, active 2026-06).
---

# GSD Loop Engineering for ML Tasks

## Problem

ML pipelines are naturally **multi-stage loops**:
1. Plan → 2. Execute (data prep / train / generate submission) → 3. Verify (format / CV-LB gap / sanity) → 4. Fix or Ship.

Most agentic ML workflows fail because:
- **Context rot**: long sessions degrade model output (e.g., 0/1 submission bug in S6E2 was partly due to loss of focus)
- **No verification**: tasks accepted as done without adversarial checking (S6E4 submission would have been caught by a verifier)
- **No shared memory**: each step reinvents context from scratch
- **Drift on style**: best practices from start of session get forgotten by end

GSD Core (open-gsd/gsd-core) is the canonical loop-engineering framework solving exactly this. This skill adapts its patterns to ML.

## Context / Trigger Conditions

Use this skill when:
- Working on a Kaggle competition end-to-end
- Designing a multi-stage ML pipeline (data → features → train → ensemble → submit)
- Session has > 50 turns and quality is degrading
- You want auditable proof each step worked (not just "looks right")
- You're tempted to use `--dangerously-skip-permissions` (don't)

**Don't use**:
- Single-file one-off scripts
- Pure EDA with no downstream work
- Research tasks where you're just reading papers

## Solution: 5-Phase ML Loop

```
┌────────────────────────────────────────────────────────────┐
│  Phase 1: DISCUSS (gsd-discuss-phase)                      │
│  - Resolve ambiguities in the task brief                   │
│  - Output: .planning/phases/<N>-DISCUSS.md                  │
├────────────────────────────────────────────────────────────┤
│  Phase 2: PLAN (gsd-plan-phase)                             │
│  - Research + plan, but in fresh-context subagent          │
│  - Plans declare wave dependencies (DAG)                   │
│  - Output: .planning/phases/<N>/RESEARCH.md + PLAN-<M>.md  │
├────────────────────────────────────────────────────────────┤
│  Phase 3: EXECUTE (gsd-execute-phase)                       │
│  - Run plans in waves (parallel where independent)         │
│  - Each plan is a fresh-context subagent                   │
│  - Output: artifacts (models, submissions, logs)          │
├────────────────────────────────────────────────────────────┤
│  Phase 4: VERIFY (gsd-verify)                              │
│  - Adversarial check: assume goal NOT achieved             │
│  - Findings classified BLOCKER / WARNING / VERIFIED        │
│  - Output: .planning/phases/<N>/VERIFICATION.md            │
├────────────────────────────────────────────────────────────┤
│  Phase 5: SHIP or FIX                                      │
│  - VERIFIED → ship submission, archive artefacts           │
│  - BLOCKER → generate fix plan, re-enter Phase 2           │
│  - WARNING → decide case-by-case                           │
└────────────────────────────────────────────────────────────┘
```

## Adapted to Kaggle Competitions

| GSD Phase | ML Equivalent |
|---|---|
| DISCUSS | Identify metric (AUC / RMSLE / mAP), submission format, deadline, data quirks |
| PLAN | Plan features, baseline model, ensemble strategy, verification checks |
| EXECUTE | Run feature engineering, AutoGluon baseline, blend, generate submission |
| VERIFY | Submission format check, CV-LB gap sanity, distribution check, adversarial validation |
| SHIP | Submit to Kaggle LB; archive all artefacts |

## Concrete: GSD-Driven S6E2 Re-Run (validated pattern)

This is the pattern that produced Private LB 0.95510 in 15 minutes:

```
Phase 1: DISCUSS
  - Metric: AUC (roc_auc) — binary classification
  - Data: 630K train, 270K test, 14 features, label "Heart Disease"
  - Submission: 1 proba column (PitNextLap or class 1) — CRITICAL
  - Time budget: 15 min training
  Output: .planning/01-discuss.md

Phase 2: PLAN
  - Plan A: AutoGluon best_quality with time_limit=900s
  - Plan B: Verify submission format matches sample_submission.csv
  - Plan C: Probability file only, never thresholded
  Output: .planning/02-plans/{A,B,C}.md

Phase 3: EXECUTE
  - Wave 1: Plan A (training, 15 min) and Plan B (parallel — read sample)
  - Wave 2: Plan C (re-generate submission using predict_proba)
  Output: ag_models/, submission_autogluon_proba.csv

Phase 4: VERIFY (the moment of truth)
  - Check 1: submission columns = sample columns? PASS
  - Check 2: submission values in [0, 1]? PASS
  - Check 3: OOF AUC > 0.95? PASS (0.95554)
  - Check 4: CV-LB gap < 0.01? PASS (gap = 0.002 after probability fix)
  Output: .planning/04-verify.md (all VERIFIED)

Phase 5: SHIP
  - kaggle competitions submit ...
  - Archive: ag_models/, leaderboard.csv, summary.json
```

## Key GSD Patterns Applied to ML

### Pattern 1: Fresh-Context Subagents
- Main session: thin orchestrator + state reads
- Subagents (one per heavy task): 200k clean context each
- Shared substrate: disk files (`.planning/`, `ag_models/`, `submission_*.csv`)

### Pattern 2: Goal-Backward Verification
- Default stance: **assume goal NOT achieved until codebase evidence proves it**
- For ML: assume submission is wrong until checks pass
- Checks must be concrete (file exists, value in range, format matches)

### Pattern 3: Wave-Based DAG Execution
- Plans declare `depends_on: ["plan-A"]`
- Independent plans run in parallel (no human bottleneck)
- For ML: feature engineering can run while baseline is training

### Pattern 4: Escalation Gate
- If verifier finds BLOCKER, stop and surface to user
- Don't silently guess — better to ask than submit a broken file

### Pattern 5: Spec-Driven Artefacts
- Every phase produces structured artefacts on disk
- For ML: every step produces a CSV / model file / log
- State survives session boundaries

## Anti-Patterns to Avoid

| Don't | Do |
|---|---|
| Submit without format check | Always run Plan B (verify format) before submit |
| Trust CV score as final metric | Always run Plan C (probability file) for AUC |
| Skip verification on "obvious" tasks | The 0/1 submission bug shows even simple tasks fail verification |
| Run all plans sequentially | Independent plans (data download + sample check) can run in parallel |
| Keep heavy research in main session | Spawn fresh-context subagent for data exploration |
| Train 10 models to "find the best" | Train 3-5 diverse models, ensemble is usually enough |

## Implementation Status

This skill describes the GSD-driven ML pattern. To **actually use** GSD:

```bash
# Install (run once)
git clone https://github.com/open-gsd/gsd-core ~/projects/gsd-core
cd ~/projects/gsd-core
npm install
node bin/install.js

# Use in any project
cd ~/projects/kaggle-ps-s6e4
claude
# Inside Claude Code:
# /gsd-new-project   → bootstrap .planning/
# /gsd-discuss-phase → Phase 1
# /gsd-plan-phase    → Phase 2
# /gsd-execute-phase → Phase 3
# /gsd-verify        → Phase 4 (THIS IS WHERE YOU CATCH THE 0/1 BUG)
# /gsd-ship          → Phase 5
```

## Real-World Validation

| Project | Without GSD | With GSD-style Verification |
|---|---|---|
| S6E2 first attempt | 0/1 submission, LB 0.884 | Caught by verification, fixed to LB 0.95357 |
| S6E4 R13 | Stacking + threshold bundled, LB worse | Controlled variable: stacking OK, threshold bad |
| Store Sales v2 | sales=0 fill, LB 2.83 (worse than v1) | mean_ratio=0.11 diagnostic, ffill fix, LB 1.90 |

The S6E2 case is the canonical example: same model, two submissions, 0.07 LB difference — verification caught it.

## Related Skills

- `kaggle-submission-format-by-metric` — format check is verification step #1
- `autogluon-first` — the "execute" phase standard recipe
- `cv-lb-gap-acknowledgment` — verification step #4 (gap check)
- `ml-sweet-spot` — when to STOP iterating (loop termination condition)
- `three-layer-wisdom-extraction` — how to extract lessons from verification findings

## References

- GSD Core: https://github.com/open-gsd/gsd-core (canonical)
- Original: https://github.com/gsd-build/get-shit-done (64K stars, deprecated)
- Related repos: gsd-browser, context-packet, agent-inbox (gsd-build org)
- S6E2 verification pattern: see `docs/ml-agent-memory/lessons/s6e2_submission_format.md`