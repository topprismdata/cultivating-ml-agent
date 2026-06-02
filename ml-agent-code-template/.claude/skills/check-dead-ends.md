---
name: check-dead-ends
description: Check memory for known dead ends before starting any new direction or experiment
type: feedback
---

# Check Dead Ends First

> Use this skill before starting any new direction. It saves hours of redundant experimentation.

## When to Use

**Always** before:
- Starting a new experiment
- Trying a new model architecture
- Switching to a different data source
- Adding a new feature engineering approach
- Tuning hyperparameters in a known area

## The Rule

**Read `memory/feedback_no_recheck_confirmed_dead.md` before investing time in any direction.**

If the direction is in the dead ends list, don't re-investigate. Pivot.

## How to Apply

```
1. Open memory/feedback_no_recheck_confirmed_dead.md
2. Search for keywords related to your new direction:
   - The technique (e.g., "CatBoost")
   - The data (e.g., "sentiment")
   - The competition (e.g., "TPS May")
3. If found: stop, read the entry, pivot
4. If not found: proceed with experiment
5. After experiment: if dead end, ADD to the list
```

## Why

The cost of checking is 30 seconds. The cost of re-running a failed experiment is hours or days.

**Evidence**: At least 5+ known dead ends in MLE-Bench agent memory (CatBoost on large tabular, multi-seed meta-learner averaging, etc.). Each one cost 2+ hours to confirm initially. Re-checking wastes those hours again.

## Example

```
Direction: "Try CatBoost with native categorical handling on 800K row tabular data"

Before: Read feedback_no_recheck_confirmed_dead.md
Found entry: "CatBoost on large tabular — fold AUC 0.966 vs LGB 0.994, 10x slower"
Action: Use LightGBM instead. Don't run CatBoost.
```

## Related

- `local-optimum-trap` — recognize when stuck
- `stacking-ceiling` — when to stop stacking
- All other skills benefit from dead-end checks first