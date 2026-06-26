---
name: learned-value-beats-heuristic-augmentation
description: |
  Use when: (1) You have a strong hand-crafted baseline (heuristic, rule-based agent,
  domain-expert system) and want to improve it with ML, (2) Naive augmentation (search,
  behavioral cloning, type-classification) is making things WORSE, not better, (3) You're
  deciding between "more training data" vs "a learned value function" for a game/RL/problem,
  (4) A competition baseline resists every improvement attempt. Do NOT trigger for
  tabular/standard ML where the baseline is itself a learned model — this is about beating
  a STRONG HEURISTIC with learned components.
---

# Learned Value Function + Search Beats Heuristic Augmentation

When a strong hand-crafted heuristic resists every naive ML augmentation (search,
behavioral cloning, type-classification), the failure isn't data volume or feature
richness — it's the **absence of a learned value function**. The breakthrough comes
from combining (a) a learned value network as a leaf evaluator with (b) sufficient
search depth. Neither alone is enough.

This skill captures the hard-won methodology from a multi-week PTCG (Pokémon TCG)
AI Battle project where a rank-304 rule-based agent (nursrijan_adv) defeated every
naive ML upgrade — until a ReBeL-style value network + 2-ply search finally beat it.

## The Anti-Pattern: Strong Baselines Resist Crude Augmentation

A strong heuristic encodes deep domain knowledge (matchup awareness, sequencing,
timing). Naive ML augmentation disrupts its internal coherence:

- **Search with a hand-crafted eval** overrides good heuristic picks with worse ones
  (the eval is cruder than the heuristic it's trying to improve).
- **Behavioral cloning on winners** picks the wrong move ~45% of the time, and errors
  compound across a game. 10× more data doesn't fix this — 437K rows still lost 98%.
- **Type-guided policy** (predict action TYPE, let heuristic pick within type) breaks
  the heuristic's coherent sequencing — 6% win rate, worse than random.

**Common failure signal:** every augmentation approach lands in the same 5-45% range
regardless of data volume or feature richness. If you see this, you're not fighting a
data problem — you're fighting the coherence problem.

## The Breakthrough: V_net + Search Depth, Together

The combination that finally wins:

1. **Train a value network** V(state) → P(win). This is a *classification* problem
   (who's-ahead), not an imitation problem. Use existing game logs with outcome labels.
   Targets: >70% accuracy, well-calibrated win-probabilities.
2. **Use V_net as the leaf evaluator inside search** (MCTS / minimax / forward-rollout).
   The search provides depth (sees the opponent's response); V_net provides the learned
   judgment at the leaf.
3. **Override the heuristic only when V_net is confident** (margin ≥ 0.03), so the
   heuristic's coherence is preserved on uncertain decisions.

**Empirical result (PTCG, 200-game offline A/B):**
- 2-ply search + hand-crafted eval: 42.5%
- 1-ply search + V_net: 45%
- **2-ply search + V_net: 59.0%** (first statistically-significant win over the heuristic)

Both components are necessary. The learned eval fixes the search-augmentation failure;
the search depth fixes the value-net's myopia.

## Methodology: Phase-Gated with Kill Criteria

Don't commit to the full RL pipeline upfront. Use decision gates:

| Phase | Deliverable | Gate (proceed only if) |
|-------|-------------|------------------------|
| 0 | Simulator/feature infrastructure | can generate ≥10K self-play games/day |
| 1 | Value network from existing logs | **V_net accuracy > 70%, AUC > 0.78** |
| 2 | Search + V_net (1-ply then 2-ply) | beats heuristic offline > 50% |
| 3 | Belief model for hidden info | +2pp over Phase 2 |
| 4 | Policy net + self-play iteration | vs heuristic ≥ 60% |

If Phase 1 fails the gate, enrich features — don't proceed to search. If Phase 2 fails,
the value net is too weak — don't add depth. **Early-stop on gates prevents sunk-cost
on dead approaches** (this project spent days on BC before V_net worked).

## Diagnostics: Which Failure Mode Are You In?

Before choosing an approach, diagnose:

- **V_net calibration plot** (predicted win-prob vs actual win-rate buckets): if
  uncalibrated, features are missing signal. Fix before search.
- **Early-game vs late-game V_net accuracy**: if early-game << late-game, the model
  only learned "who has more prizes" — not positional judgment. Enrich features.
- **Margin sweep** (override threshold 0.02 / 0.03 / 0.05): if looser margin → worse
  win rate, V_net is noisy on near-eval states — keep the conservative margin.

## When NOT to Apply This Skill

- **The baseline is itself a learned model** (standard tabular/CV/NLP ML). The
  "strong heuristic" framing doesn't apply — just improve the model directly.
- **No simulator / no way to forward-rollout.** Search needs a forward model. If you
  only have offline logs, stick with BC + careful evaluation.
- **The baseline is weak** (< median). Cheaper improvements (features, ensembling,
  hyperparameters) dominate — don't jump to RL.

## Anti-Patterns (Forbidden)

- **"More data will fix BC"** — it won't, if the value function is missing. 10× data
  moved PTCG BC from 5.8% → 1.7% (worse, because the larger model overfit winner bias).
- **"Search depth with hand-crafted eval"** — depth amplifies eval errors. Always
  learn the eval first.
- **"Augment the heuristic in-place"** — coherence conflicts kill performance. Either
  wrap (heuristic handles fallback, learned model overrides on confidence) or replace
  entirely; don't interleave their logic.

## Key Files (PTCG reference implementation)

- Value net training: `train_value.py` — HistGradientBoosting, 82 state features.
- Search+V_net agent: 2-ply rollout (heuristic-guided, globals isolated) + V_net leaf.
- Decision-gate metrics: accuracy, AUC, calibration buckets, early/mid/late split.

## Source

PTCG AI Battle Challenge (Kaggle, 2026-06). Sessions 5-6: 8 failed augmentation
approaches → V_net (89.4% acc) + 2-ply search = 59.0% vs rank-304 heuristic.
Full plan in the project's `DEEP_RL_PLAN.md` (ReBeL architecture, 5 phases).
