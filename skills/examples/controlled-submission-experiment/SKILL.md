---
name: controlled-submission-experiment
description: |
  Use when: (1) a new submission scores worse than baseline and the reason is unclear,
  (2) multiple changes were made simultaneously (new model + new post-processing + new features),
  (3) need to isolate which component caused a regression, (4) CV improves but LB degrades,
  (5) comparing "smart" vs "simple" approaches, (6) verifying that model improvements are real
  vs artifacts of post-processing changes.
  CRITICAL: Never change more than one variable between submissions without controlled comparison.
---

# Controlled Submission Experiment

## Problem

When multiple changes are bundled into a single submission (new features + new model +
new post-processing), it's impossible to tell which change helped or hurt. A submission
that scores worse than baseline might actually contain valuable model improvements hidden
by a post-processing regression.

## Symptoms

- New submission scores worse than baseline despite "better" CV
- Multiple changes made simultaneously (features + model + post-processing)
- Zero ratio in submission changes significantly (>2%) between versions
- Can't explain WHY the score changed in a specific direction

## Solution

### The Controlled Experiment Pattern

Isolate one variable at a time by creating multiple submissions from the same model output:

```
Baseline:  Model_A + Postproc_A  → Score_A
New:       Model_B + Postproc_B  → Score_B
Control:   Model_B + Postproc_A  → Score_C  (KEY!)
```

**Interpretation matrix:**

| Pattern | Model B | Postproc B | Conclusion |
|---------|---------|------------|------------|
| Score_C > Score_A, Score_B ≈ Score_C | Better | Neutral | Model B is better, postproc doesn't matter |
| Score_C > Score_A, Score_C >> Score_B | Better | Worse | Model B is better BUT postproc B regresses |
| Score_C < Score_A, Score_B ≈ Score_C | Worse | Neutral | Model B is genuinely worse |
| Score_C ≈ Score_A, Score_B >> Score_A | Neutral | Better | Postproc B carries the improvement |
| Score_C > Score_A, Score_B > Score_C | Better | Better | Both improve, model B slightly more |

### Implementation

```python
import pandas as pd
import numpy as np

# Step 1: Load both submission predictions (before post-processing)
r10_raw = pd.read_csv("submission_r10_raw.csv")  # baseline raw predictions
r11b_raw = pd.read_csv("submission_r11b_raw.csv")  # new raw predictions

# Step 2: Identify disputed predictions
merged = r10_raw.merge(r11b_raw, on="id", suffixes=("_r10", "_r11b"))
disputed = merged[
    (merged["sales_r11b"] == 0) & (merged["sales_r10"] > 0)
]
print(f"Disputed rows: {len(disputed)}")
print(f"R10 values in disputed: mean={disputed['sales_r10'].mean():.2f}")

# Step 3: Create controlled submission
# Use Model_B predictions but with Postproc_A logic
controlled = r11b_raw.copy()
# Apply only the baseline post-processing
controlled.loc[controlled["sales"] < 0.1, "sales"] = 0
controlled.to_csv("submission_r11c_controlled.csv", index=False)
```

### Diagnostic: Zero Ratio Comparison

```python
# Quick diagnostic — always check this
for name, df in [("R10", r10), ("R11b", r11b), ("R11c", r11c)]:
    zero_pct = (df["sales"] == 0).mean()
    print(f"{name}: zeros={zero_pct:.2%}, mean={df['sales'].mean():.2f}")

# If zero ratio jumps >2% between versions, investigate immediately
```

## Prevention

1. **One variable at a time** — never change model AND post-processing in the same submission
2. **Always compare zero ratios** — a jump >2% signals a post-processing change that may hurt
3. **Save raw predictions** before post-processing, so you can re-apply different post-processing
4. **Name your experiments** (R10, R11b, R11c) to track what changed between each

## Evidence (Real Experiment)

Kaggle Store Sales (Favorita), April 2026:

| Version | Model | Post-processing | Zeros | LB |
|---------|-------|----------------|-------|--------|
| R10 | baseline | `< 0.1 → 0` | 8.36% | 0.39880 |
| R11b | improved | min_sales adaptive | 12.41% | 0.40073 |
| R11c | improved | `< 0.1 → 0` | 8.30% | **0.39824** |

Without the controlled experiment, we would have concluded "the improved model is worse."
With it, we proved the model was actually +0.00056 better, but post-processing regressed -0.00249.

## Notes

- This is the ML equivalent of A/B testing — isolate variables rigorously
- Related to `rmsle-zero-threshold-asymmetry` — the specific post-processing failure mode
- Related to `kaggle-optimal-blending` — when blending, also test components in isolation
- For ensemble experiments, use the same pattern: add one model at a time, measure delta
