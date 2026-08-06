---
name: feature-engineering-saturation-detection
description: |
  Use when you have run 3+ feature engineering experiments with no improvement, or when distance to theoretical upper bound is <15pp. Triggers when you suspect "I should stop optimizing features and try something else" — the most common failure mode is wasting weeks past the saturation point.
---

# Feature Engineering Saturation Detection

## Context
Feature engineering **always saturates**. Every project has a "ceiling" past which new features stop helping. Continuing to optimize past saturation wastes weeks of engineering time. In a recent retail SKU recommendation project, 13 consecutive failed experiments (v31-v43) occurred before recognizing saturation at 76.5% F1 (theoretical ceiling 90.7%). This skill teaches how to detect saturation early and switch paradigms.

The core lesson: **optimization without a ceiling is wasted effort**. If you don't know your theoretical upper bound, you can't know if you're done.

## Guidance

### The 4 Saturation Signals

```
Signal 1: Consecutive F1 Stagnation
  - 3-4+ experiments without improvement
  - Each new feature adds ≤0.1pp
  - Action: Stop adding features, switch paradigm

Signal 2: High Correlation with Existing Features
  - New feature has Spearman correlation >0.7 with existing ones
  - Captures no new information
  - Action: Skip the feature, document why

Signal 3: Distance to Theoretical Upper Bound < 15pp
  - F1-EM (oracle ceiling) - actual F1 < 15pp
  - Historical coverage ceiling < 10pp above actual
  - Action: Optimization ROI is low, consider external data / new paradigm

Signal 4: Improvements Only From Threshold/Window Tuning
  - Last 3+ improvements came from "tweak N", "expand window", "adjust threshold"
  - No new information captured
  - Action: Architectural change needed (model class, data source, real-time signals)
```

### Diagnostic Script

```python
def detect_saturation(experiment_log):
    """Returns saturation status + recommended action"""

    recent = experiment_log.tail(5)

    # Signal 1: Stagnation (3+ experiments without >0.2pp improvement)
    improvements = recent['f1_diff'].tolist()
    stagnant = len([x for x in improvements[-3:] if x > 0.002]) == 0

    # Signal 3: Distance to ceiling
    upper_bound = compute_f1_em_upper_bound()  # F1-EM = oracle ceiling
    distance_to_ceiling = upper_bound - recent['f1'].iloc[-1]

    # Signal 2: Feature correlation
    new_feature_max_corr = check_feature_correlation(recent['new_features'])
    high_corr = new_feature_max_corr > 0.7

    if stagnant and distance_to_ceiling < 0.15:
        return "SATURATED — switch paradigm (external data, real-time signals, new model class)"
    elif high_corr:
        return "FEATURE REDUNDANT — try different angle or skip"
    elif distance_to_ceiling < 0.05:
        return "NEAR CEILING — declare success and ship"
    else:
        return "ACTIVE — keep optimizing features"
```

### Decision Tree

```
Experiment didn't improve?
├── Yes → Check correlation with existing features
│        ├── max corr > 0.7 → Document as "redundant", skip
│        └── max corr < 0.7 → Check if business meaning makes sense
│                          ├── No → Document as "wrong intuition", skip
│                          └── Yes → Try different aggregation/window
└── No → Compare to theoretical upper bound
         ├── < 5pp gap → Declare success, ship to production
         └── > 15pp gap → Continue feature engineering
```

### Saturation Journal Template

Track WHY each failed experiment failed, not just THAT it failed. Failed experiments are organizational assets.

```markdown
## v31_error_features
- F1: 73.5% (-0.5pp from v28)
- Hypothesis: Add error analysis features (where model was wrong)
- Implementation: [details]
- Failure mode: FEATURE_REDUNDANT
- Diagnosis: New features correlated 0.85+ with `pair_weeks` and `reorder_ratio`
- Action: Skip feature, document pattern

## v34_n_predictor
- F1: 70.0% (-2.5pp from v32)
- Hypothesis: Predict N (recommendation count) instead of using formula
- Failure mode: WRONG_OBJECTIVE
- Diagnosis: N depends on actual purchases that week; using historical N leaks
- Action: Reverted, kept Optuna formula
```

### When to Switch Paradigm (Beyond Feature Engineering)

Once saturated, try:

| Paradigm | Expected Lift | When to Try |
|---|---|---|
| External data (promotions, weather, holidays) | +2-5pp | If you have data access |
| Real-time signals (last 3 days of behavior) | +3-5pp | If you have streaming pipeline |
| Deep sequence model (Transformer, LSTM) | +0-3pp | If you have 100k+ sequences |
| Model architecture change (NN over GBDT) | +0-2pp | If GBDT is saturating |
| RFM segmentation + per-segment models | +0-1pp | Last resort, often saturates fast |
| More hyperparameter tuning | +0-0.3pp | Almost never worth it |

## Why This Matters

In the retail SKU project, 13 consecutive failed experiments (v31-v43) cost ~3 weeks of engineering time. With this skill, you detect saturation at experiment 5 and redirect to "external data" or "real-time signals" — saving **2+ weeks per project**.

| Without Skill | With Skill |
|---|---|
| 13 failed experiments (3 weeks) | 5 failed experiments (1 week) |
| No clear "stop" signal | Explicit saturation criteria (4 signals) |
| Optimization continues past ceiling | Switch to architectural change at ceiling |
| No failure documentation | Every failure has root cause analysis |
| Result: project ends at 76.5% F1 after 6 weeks | Result: project ends at 76.5% F1 after 4 weeks, with 2 weeks for external data exploration |

## When to Apply

### When to Use
- You have run 3+ feature engineering experiments
- You have computed theoretical upper bound (F1-EM, RMSE-min, AUC-max, etc.)
- You suspect diminishing returns
- You want to know "should I keep optimizing or move on?"
- Your team is asking "why isn't this working anymore?"

### When NOT to Use
- You have only run 1-2 experiments (too early)
- You haven't computed theoretical upper bound yet (compute it first)
- Your metric is highly stochastic (need 5+ experiments for confidence)
- You are still in baseline/baseline+1 phase (no saturation possible yet)
- The task is fundamentally novel (no historical patterns to learn from)

## Notes
- **Saturation ≠ Failure**: Saturation means you hit the ceiling of the current approach. Try a new approach (model class, data source, real-time signal), not a better version of the current one
- **Document failures religiously**: Failed experiments become valuable organizational assets — root cause analysis prevents the team from re-trying the same dead ends months later
- **Theoretical ceiling is essential**: Without F1-EM, you can't know you're saturated. Always compute it FIRST before optimizing features
- **"I have an idea!" is not a hypothesis**: Every new feature needs a business reason. "I think this might help" is not a business reason; "Customer X in segment Y behaves like Z, so feature F should capture this" is
- See also: `time-series-walk-forward-validation` (for computing upper bound), `multi-level-aggregation-overfitting`

## References
- Related principle: "Work Smart > Hard Work" — recognize when to stop and pivot
- Pairing skill: `time-series-walk-forward-validation` (provides the upper bound calculation this skill depends on)
- Layer 3 wisdom in cultivating-ml-agent framework: cross-domain principles for ML practitioners