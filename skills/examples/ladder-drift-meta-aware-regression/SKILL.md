---
name: ladder-drift-and-meta-aware-regression
description: |
  Use when: (1) Your agent's LB score drops dramatically week-over-week without
  code changes, (2) You're tempted to add "meta-aware" bonuses (priority targets,
  matchup bonuses, deck-type detection) to a strong heuristic, (3) Testing agent
  variants on small samples (60-80 games) and getting noisy/conflicting results,
  (4) You see error rates spike when testing against meta decks. Do NOT trigger
  for standard tabular ML — this is about competition agents where (a) the meta
  shifts daily, and (b) "improvements" that look good in small tests often regress
  in production. Key lesson: same code, 200-point LB drift in 48h. Key anti-pattern:
  +600 target bonus on hypothetical Pokemon → -10% vs that deck.
---

# Ladder Drift + Meta-Aware Regression: The PTCG AI Battle 2026-06 Findings

This skill captures three hard-won lessons from a multi-week **PTCG (Pokémon TCG)
AI Battle** project where a strong 1084.5-LB Lucario agent (v29, public notebook fork)
held the lead for days — then lost 200 LB in 48 hours due to meta shift, and lost
another 150 LB to "meta-aware" patches that LOOKED like improvements in 60-game tests
but REGRESSED in 200-game tests.

## Lesson 1: Ladder Drift is Real (Same Code, -200 LB in 48h)

**v29 baseline (no code changes) — different scores across days**:

| Date | LB Score | Delta |
|------|----------|-------|
| 2026-06-27 13:13 | **970.0** | baseline |
| 2026-06-28 16:29 | **774.6** | **-195** |
| 2026-06-29 16:25 | **770.5** | -4 (stabilizing) |

The agent didn't change. The ladder did. Top 5 decks shifted from Lucario-dominant
to Starmie / Archaludon / Dragapult-dominant. **The same code that scored 970
against a Lucario ladder scored 770 against a Starmie ladder.**

**Implication**: When your agent's LB drops, **check the meta before debugging
the code**. Look at top-20 leaderboard deck names; if they're all different decks
than when you calibrated, it's drift, not bug.

**Detection**: Compare LB score to baseline over time. If drift > 100 points in
a week with no code change, the meta moved. Don't waste quota on resubmits of
the same code.

## Lesson 2: Meta-Aware Heuristics Often REGRESS (The 60→200 Game Lesson)

Three "improvements" tested on small samples (60-80 games) all looked neutral or
marginally positive. **All three regressed when tested at 200 games**:

| Variant | Sample | Mirror | vs Meta Deck | Verdict (60g) | Verdict (200g) |
|---------|--------|--------|--------------|---------------|-----------------|
| v33 (v29 + ALL v31 logic) | 80g mirror + 60g meta | -7.5% mirror | -13.1% vs Arch | "neutral" | **REGRESSION** |
| v34 (v29 + Archaludon-only detection) | 80g mirror + 60g meta | +5.0% mirror | -5.6% vs Arch | "slightly better" | **-10.1% vs Arch** |
| v34 v2 (rerun) | 200g mirror + 200g meta | +10.0% mirror | -25.9% vs Arch | n/a | **-10.1% vs Arch** |

**Why 60 games lies**: With 60 games and a 50% baseline, the 95% CI is roughly
±13 percentage points. A "+5% improvement" is within noise. You need 200+ games
to see real effects.

**Why meta-aware logic backfires**:
- `+500 target bonus` for a specific Pokemon shifts attack decisions to attack
  that target even when not feasible (e.g., Archaludon behind Full Metal Lab's
  -30 damage reduction → 130 dmg Lucario attack only does 100 dmg, takes 3 turns
  to KO → loses tempo).
- `+500 attack bonus` for Mega Brave vs metal deck causes agent to over-prioritize
  high-cost attacks, missing cheap lethal.
- `+6000 Gravity Mountain score` vs metal deck makes agent over-commit stadium
  even when opponent doesn't have a stage 2.

**The hidden assumption violated**: Meta-aware bonuses assume the target IS on
the board AND you can reach it. In practice, the agent commits to a plan
("attack Archaludon") and then crashes when the plan is infeasible (Archaludon
benched, switched, etc.). Error rates spike from ~25% to 45-77% in meta matchups.

**Decision rule**: If a "meta-aware" change doesn't beat baseline by ≥10pp in a
200-game A/B test, **assume it's regression and don't submit**. Meta-aware logic
on a strong heuristic is a coherence conflict — the heuristic's existing priorities
already encode domain knowledge that the bonuses disrupt.

## Lesson 3: Crash-Safety Wrappers Have Subtle Bugs

A "v31b" variant added try/except around the agent with this wrapper:

```python
# v31b's crash safety — DO NOT COPY
k = min(select.maxCount, n)
k = max(k, min(max(1, select.minCount), n))  # BUG: forces k=1
return ordered[:k]
```

**The bug**: When `select.minCount` is `None` or `0`, `max(1, ...)` forces
`k=1`. This breaks multi-option selects like "choose 2 cards" — the agent returns
only 1, the game state is invalid, and the next call crashes.

**Result**: v31b scored 821.3 LB vs v29's 970 LB. **The crash-safety itself
caused crashes**.

**Correct pattern** (preserve minCount=None semantics):

```python
# CORRECT: don't override minCount=None
min_c = select.minCount if select.minCount is not None else 1
max_c = select.maxCount if select.maxCount is not None else max(min_c, len(ranked))
k = min(max_c, n)
k = max(k, min(min_c, n)) if min_c <= n else 1
return ranked[:k]
```

**Lesson**: When adding error handling around existing logic, **trace what
the original code does for every input case** (None, 0, 1, n). A wrapper that
looks defensive can break the underlying contract.

## Methodology: The 200-Game Gate

Before submitting any agent variant to the ladder, run a 200-game A/B test:

```
1. 100 mirror games (same agent vs itself) — establishes noise floor
2. 100 games vs each of top 3 meta decks (Starmie, Archaludon, Dragapult)
3. Compute 95% CI for each matchup
4. Decision:
   - All matchups within ±5pp AND within noise CI: keep, low confidence
   - Any matchup >+5pp better than baseline AND outside CI: submit
   - Any matchup <-5pp worse than baseline: DO NOT SUBMIT (even if others look good)
5. Repeat with 200 games total per matchup for final confirmation
```

**Why 200 games**: With 50% baseline, 200 games gives ~7pp standard error,
so 5pp differences are detectable. 60 games gives 13pp SE, useless for small
effects.

## When NOT to Apply

- **Standard tabular ML** (Kaggle competitions like House Prices, Titanic):
  ladder drift is real but usually within 5-10 LB points over a competition.
  Meta-aware logic doesn't apply.
- **Single-submission competitions**: No "ladder" to drift on.
- **Pure research benchmarks**: No live meta.

## Anti-Patterns (Forbidden)

- **"I'll just add a +500 bonus for [popular target]"** — coherence conflict.
- **"60 games is enough to validate"** — 95% CI is ±13pp, you'll miss regressions.
- **"Resubmit the same code to recover"** — ladder drift only fixes if meta cycles back.
- **"Crash-safety can't make things worse"** — yes it can, see v31b (-150 LB).
- **"Different deck entirely will solve the meta problem"** — new agent, new bugs.
  v30_archaludon had 46% error rate vs Lucario (catastrophic), worse than v29.

## Key Diagnostics (PTCG)

- **LB over time plot**: Detect drift (>50 point drop in 1 week = drift)
- **200-game A/B test**: All agent variants must pass this gate
- **Error rate per matchup**: If errs > 30% vs a meta deck, the agent is broken
  on that matchup — don't submit
- **Top-5 leaderboard deck names**: If different from your test set, retest

## Source

PTCG AI Battle Challenge (Kaggle, 2026-06). Session 8 (2026-06-29):
- v29 ladder drift: 970 → 770 over 48h (Starmie/Arch/Dragapult meta shift)
- v33 regression: 60g test "neutral", 200g test -13% vs Arch
- v34 regression: 60g test "slight improvement", 200g test -10% vs Arch
- v31b crash-safety bug: -150 LB, broken minCount=None semantics
- v30_archaludon: 46% error rate vs Lucario, worse than v29 baseline

Combines with [[learned-value-beats-heuristic-augmentation]] (PTCG v0.8.0 deep RL
breakthrough) — both skills together cover the full PTCG project arc.