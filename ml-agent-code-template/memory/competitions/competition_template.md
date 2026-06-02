# Competition: <name>

## Status

- **Started**: <YYYY-MM-DD>
- **Current**: <in progress / GOLD / SILVER / BRONZE / no medal>
- **Best Score**: <OOF / Public LB / Priv>
- **Target**: <Gold threshold / Silver threshold / etc.>

## Problem

**Goal**: <one sentence>
**Metric**: <AUC / RMSE / LogLoss / Jaccard / etc.>
**Data**: <rows, features, target distribution>
**Constraints**: <compute, time, data type>

## Data Paths

```
Public: <path>
Private: <path>
Submission: <path>
```

## Approach

### Architecture

```
[Input]
   ↓
[Feature Engineering]
   ↓
[Base Models: X, Y, Z]
   ↓
[Stacking / Ensemble]
   ↓
[Post-processing]
   ↓
[Submission]
```

### Base Models Tried

| Model | OOF | Priv | Notes |
|-------|-----|------|-------|
| LGB | 0.994 | - | Baseline |
| CatBoost | 0.966 | - | Too slow, too weak |
| XGBoost | 0.993 | - | Similar to LGB |
| NN | 0.991 | - | Slightly weaker |

### Stacking Strategy

- Method: <greedy / comprehensive / ridge / LGB meta>
- Number of base models in final stack: <N>
- Best meta-learner config: <params>

## Key Insights

1. **Insight 1**: <what you learned>
2. **Insight 2**: <what you learned>
3. **Insight 3**: <what you learned>

## Failures

| Approach | Why it failed | Time wasted |
|----------|---------------|-------------|
| <approach 1> | <reason> | <hours> |
| <approach 2> | <reason> | <hours> |

## Submission Versions

| Version | Date | OOF | Priv | Notes |
|---------|------|-----|------|-------|
| v1 | YYYY-MM-DD | 0.99 | - | Baseline |
| v2 | YYYY-MM-DD | 0.995 | - | + new features |

## Final Approach

**Best version**: v<N>
**Method**: <description>
**Score**: <OOF / Priv>
**Time to develop**: <hours>

## Lessons for Next Time

1. <Lesson 1>
2. <Lesson 2>
3. <Lesson 3>