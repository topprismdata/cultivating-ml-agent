# Local Optimum Trap

> When to stop tuning and pivot.

## The Rule

**3 attempts yielding <0.0001 improvement → pivot to a new direction.**

This is one of the most expensive mistakes in ML iteration. The instinct to "try one more parameter" leads to:
- Hours lost on futile tuning
- Misinterpretation of noise as signal
- Missed opportunities for breakthrough changes

## Recognition Signals

You're in a local optimum trap when:
1. The last 3 experiments changed < 0.0001 on the metric
2. Different meta-learner configs converge to identical scores
3. Base model correlation > 0.93 across the board
4. Multi-seed averaging gives identical results
5. Public LB stops moving despite parameter changes

## Decision Protocol

```
After experiment N:
├─ If improvement > 0.001 → continue
├─ If improvement 0.0001 - 0.001 → check if structural change possible
│  ├─ Try new base model
│  ├─ Try different feature engineering
│  └─ Try external data
└─ If improvement < 0.0001 for 3 consecutive → PIVOT

Pivot options:
1. New feature engineering (highest ROI)
2. External data
3. Different problem framing
4. Different metric optimization
```

## Empirical Evidence

**TPS May 2022** (2026-05-17):
- 27 meta-learner variants (9 configs × 3 seeds) → all converged to 0.99745
- Greedy and 20-config comprehensive search → same value
- Diagnosis: stacking ceiling reached
- **Breakthrough**: f_27 character features (not a meta-learner change)
- New AUC: 0.997540

**Lesson**: tuning meta-learners was never going to break through. Only a new base model with correlation < 0.92 could.

## Anti-Patterns to Avoid

- ❌ "Let me try one more seed"
- ❌ "Maybe this regularization will help"
- ❌ "Different optimizer might do it"
- ❌ "More iterations of CV"
- ✅ "Why are we stuck? What's the structural limitation?"

## When You Are Stuck, Ask

1. What is the data telling me that the model isn't capturing?
2. Is the bottleneck information (data), model capacity, or optimization?
3. What did 1st place do differently?
4. Is this even the right problem framing?

If you can't answer in 5 minutes, you've already pivoted in your mind. Make it explicit.