---
name: trueskill-simulation-competition-strategy
description: |
  Use when: (1) Competing in Kaggle simulation competitions (PTCG, Orbit Wars, etc.)
  where agents are rated by TrueSkill Bayesian scoring, (2) Your LB score swings
  ±200 points day-to-day with no code changes, (3) You're tempted to re-submit
  frequently to "test" improvements, (4) You see identical agents scoring 150-400
  points differently. Key lessons: re-submitting resets μ to 600 (destroys
  convergence), only latest 2 submissions count for final, the optimal endgame
  strategy is "submit your best agent's duplicate copies near deadline" to
  high-roll the Bayesian lottery. Validated across 8+ PTCG submissions + forum
  consensus from rank 4-9 players.
---

# TrueSkill Simulation Competition Strategy

## Problem
Kaggle simulation competitions (Pokémon TCG AI Battle, Orbit Wars, etc.) use
TrueSkill Bayesian scoring where each submission has N(μ,σ²). This creates
counterintuitive dynamics that trap unwary competitors:

- **LB scores swing wildly** (±200 points) with no code changes
- **Re-submitting to "test"** destroys convergence progress
- **Identical agents score 150-400 points differently** (confirmed by forum)
- **The latest 2 submissions** are the only ones that count for final scoring

## Context / Trigger Conditions

Use this skill when:
- Competing in any Kaggle **Simulation Competition** (not regular ML competitions)
- Your agent's **LB score drops** and you're tempted to debug the code
- You want to **test a new agent variant** by re-submitting
- You're within **2 weeks of the deadline** and deciding final strategy
- You see **rank swings** that seem disproportionate to code changes

## Solution

### Lesson 1: Re-submitting Resets Convergence (Critical)

Each new submission starts at μ₀=600 (the prior). σ is high initially, causing
large μ swings per game. σ decreases over time as more games are played.

**DO NOT re-submit to "test" improvements.** Every re-submit resets weeks of
convergence progress.

**Evidence (PTCG AI Battle, 2026-07)**:
| Submission | Submit-day score | After 3 days |
|-----------|-----------------|--------------|
| v37 (anti-Arch) | 600.0 | 882.9 (+283) |
| v35 (anti-Psychic) | 559.0 | 847.7 (+289) |
| Nithin A (Archaludon) | 600.0 | 967.8 (+368) |

These "failed" agents weren't broken — they just hadn't converged. Re-submitting
to "restore" them reset the convergence clock again.

### Lesson 2: Latest 2 Rule — Final Scoring

Only the **latest 2 submissions** count for final scoring. All other submissions
continue playing games but don't affect the final result.

**Strategy**: Ensure your latest 2 are your BEST 2 agents. Submit early enough
for them to converge before the deadline.

### Lesson 3: LB is a Bayesian Lottery (Not a Measurement)

Forum consensus from rank 4-9 players:
- **djschmit (44th)**: identical agents scored 940.7 vs 790.8 (150-point gap)
- **Shun_PI (4th)**: "optimal strategy = submit repeatedly, stop when high-roll"
- **LagrangianLocomotive (34th)**: dropped from 7th to 1100th

**Practical implication**: A single LB read is unreliable. Trust convergence
trends over 3+ days, not day-to-day fluctuations.

### Lesson 4: Endgame High-Roll Strategy

Near deadline (1-2 weeks before):
1. Select your single best agent (highest converged μ)
2. Submit 2 duplicate copies as your final latest-2
3. Each copy converges independently — keep whichever scores higher
4. This is the forum-validated "cheesy but optimal" strategy

### Lesson 5: Local Eval Harness (Free)

Build a **CPU kernel eval harness** for simulation competitions:
- Load multiple agents as modules in isolated temp directories
- Run round-robin games using the competition's engine (CPU-only, no quota cost)
- Test new agents BEFORE spending submission quota
- PTCG harness: 480 games in 31 seconds, 6000 games in 6 minutes — all free

**Key**: This is separate from both submission quota (5/day) and GPU quota (30h/week).

## Anti-Patterns (Forbidden)

| Anti-Pattern | Why It Fails | Fix |
|-------------|-------------|-----|
| Re-submit to "restore" lost score | Resets μ to 600 | Wait for convergence |
| Judge agent by submit-day score | Always misleading (μ₀=600) | Wait 3+ days |
| Chase LB numbers | They're lottery, not measurement | Trust eval panel + trends |
| Incremental param tweaks | Each re-submit costs convergence time | Only submit genuinely different agents |

## Evidence

- 8+ PTCG submissions tracked over 2 weeks (2026-06-28 to 07-08)
- Eval harness: 3-panel validation (N=400/matchup), caught 2 wrong submissions
- Forum: 6+ rank 4-9 players independently confirming the same dynamics
- Real cost: 5+ wasted submissions from premature "restore" re-submits
