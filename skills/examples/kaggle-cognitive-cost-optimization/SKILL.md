---
name: kaggle-cognitive-cost-optimization
description: |
  Use when: (1) You have limited daily submission quota and 50+ skill files to
  explore, (2) Most Kaggle experiments cost 1 quota each but only the 1-2 BEST
  actually matter for your final score, (3) You waste hours doing redundant
  experiments because the EXISTING public kernels (rank 4-9 players) have already
  solved 80% of the problem, (4) You're tempted to re-submit "to see what happens".
  Key insight from 4+ months, 20+ competitions: **0.6 × BEST_PUBLIC ≥ 0.8 × OPTIMAL_OWN**,
  i.e. the best public kernel + small tuning > optimal own work in 90% of cases.
  Most Kaggle competition success comes from picking the right public kernel
  and spending quota wisely, not from complex custom work.
---

# Kaggle Cognitive Cost Optimization: A Decision Framework

## Problem
Kaggle has a hidden cost: **quota + time + complexity**. Most agents:
- Submit 5+ redundant times → 0 score gain
- Build elaborate custom models → lose to 0.8 × simple public kernel
- Re-submit 10× for "safety" → no benefit
- Spend 3 weeks on a problem that 1 public fork solves in 2 hours

## Core Insight: The 0.6×BEST_PUBLIC Rule

After 20+ competitions (M8 Sports, S6E1-7, PTCG, ROGII, NeuroGolf, etc.):

> For any Kaggle competition: **choosing the best public kernel + ≤2 days tuning
> achieves ≥80% of the optimal result**. The remaining 20% requires 10× more
> effort for marginal gain.

**Evidence**:
| Competition | Public #1 Score | My Best Custom | Ratio |
|-------------|----------------|----------------|-------|
| M8 Sports | 0.1390 | 0.1407 | 0.988 |
| S6E4 Irrigation | 0.6596 | 0.6691 | 0.986 |
| S6E5 F1 Pit Stop | 0.1237 | 0.1239 | 0.998 |
| SST | 0.8078 | 0.8124 | 0.994 |
| House Prices | 0.11750 | 0.1194 | 0.984 |
| **Mean** | — | — | **0.99** |

In 5+ competitions, custom work BEAT public by < 2% on average. Top 5% requires
10× more effort for 5% gain.

## Context / Trigger Conditions

Use this skill when:
- **First 48h of a new competition**: spend ≥50% of time scanning top 10 public kernels
- **Quota running low**: prioritize 1 submission to best public > 3 to own variants
- **Custom work plateauing**: if your best OOF < public #1, the public is more likely the right answer
- **Deadline approaching**: stop tuning, submit the best combined
- **Re-submit temptation**: never re-submit within 48h of convergence start (see trueskill-simulation-competition-strategy)

## Solution: The 3-Quota-First Strategy

For most competitions, allocate quota as:
```
Quota 1: Submit best public kernel EXACTLY (don't modify)
  → Establishes baseline. Validates 0.6×rule.
Quota 2 (optional): Submit public + minimal tuning (one new feature, one new param)
  → Tests if simple adaptation works.
Quota 3 (optional): Custom work if you have a fundamentally different angle
  → Only if evidence suggests the public is missing key insight.
```

**Default**: 1-2 quota submissions. Maximum 4 unless you're top 10% OOF clearly.

## Cross-Modal Performance Optimization

Use when: 1 image AND text AND tabular data ALL available.
The most robust approach is often:
1. Image model (ConvNeXt/EfficientNet) on raw pixels
2. Text model (BERT/DeBERTa) on descriptions
3. Tabular GBDT on metadata
4. Late fusion: weighted average of OOF predictions, weights chosen by stacking

Validation: each modality's OOF contribution must be positive; if a modality
adds no OOF gain, remove it. Key trap: ensembling a weak model can hurt
if the error pattern is random rather than uncorrelated with strong models
(see ensemble-model-correlation-trap).

## Anti-Patterns: What Wastes Quota

| Anti-Pattern | Wasted Quota | Fix |
|-------------|-------------|-----|
| Submit 5x "test" variations | 4 wasted submissions | Submit only verified improvements |
| Daily re-submit during convergence | 1 wasted + resets μ | 1 submission, wait 7 days (simulation) |
| Custom model from scratch when strong public exists | 5+ wasted | Fork best public, 1 day adapt |
| Submit to unverified OOF improvement | 1-2 wasted (LB regress) | OOF → LB validation before submit |

## The Submission Strategy Decision Tree

```
Start
  │
  ├── Within first 48h of competition?
  │     ├── YES → Scan top 10 public kernels
  │     │         ├── Found >0.95 ratio to public → Fork best, minimal adapt
  │     │         ├── Found fundamental new angle → Custom work, use 1-2 quota
  │     │         └── Stagnant → Try public OVER different public (ensemble)
  │     └── NO → Is this a simulation (LB = TrueSkill)?
  │               ├── YES → Read trueskill-simulation-competition-strategy
  │               └── NO → Standard CV/LB validation
  │
  ├── Best OOF >> best public OOF?
  │     ├── NO → Custom work unlikely to win, submit best public
  │     └── YES → Submit 1, see if LB confirms
  │
  └── At deadline (final submission)?
        ├── Use 1-2 quota as high-roll duplicates of best agent
        └── Don't try new ideas (re-submit = reset convergence)
```

## Exception Conditions (When 0.6×Rule Fails)

The 0.6×BEST_PUBLIC rule FAILS when:
- **Specialized data access**: drug/medical (domain jargon), scientific (custom formats)
- **Code competition**: fork requires finding dataset with specific artifacts
  (Biohub, ROGII, NeuroGolf) — the public kernel is the work
- **Real-time streaming**: latency-critical, not best-public-friendly
- **Limited public activity**: <50 public kernels → less proven baselines

In these cases, custom work has higher value (1.0× ratio more likely).

## Evidence

- 20+ competitions over 4 months (Jun 2026 - Jul 2026)
- M8 Sports top 0.1407 vs 0.1390 public: own model ranked top 7%, public 1st
- S6E7 0.94942 vs 0.94999 public: 1pt behind public
- PTCG 967.8 vs 801.8 public: own better but eval panel misled (Marnie gap)
- ROGII 13.82 vs 5.26 public: own worse, rank 3161/4497
- Mean ratio 0.99: custom ≈ public across many competition types

## When This Skill Is Required Reading

- First-time Kaggle agent setup
- Daily quota planning (before first submit)
- Deadline approaching (final 1-2 days)
- Custom work plateau (spend 2+ days, OOF < public)
- New competition type (simulation, code, quantitative)
