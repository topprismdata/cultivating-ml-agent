# Experiment: <YYYY-MM-DD> <Short Name>

## Hypothesis

**One sentence**: What are you testing?

**Why this direction**: What evidence led you here?

## Setup

- **Data**: <dataset, train size, test size, target>
- **CV**: <StratifiedKFold(5) / GroupKFold(5) / TimeSeriesSplit(5)>
- **Baseline**: <reference score, e.g. OOF AUC = 0.995>
- **Variables changed**: <list ONE variable changed, list all else held constant>

## Code

```python
# Minimal reproducible example
```

## Results

| Metric | Baseline | This Exp | Delta |
|--------|----------|----------|-------|
| OOF AUC | 0.9950 | 0.9960 | +0.0010 |
| Priv AUC | 0.9951 | 0.9958 | +0.0007 |
| Time | 30 min | 25 min | -5 min |

## Analysis

**What worked**: <if any>
**What didn't work**: <if any>
**Surprises**: <unexpected findings>

## Conclusion

- [ ] Break through (continue iterating in this direction)
- [ ] Marginal (try one more variation)
- [ ] Hit ceiling (pivot to new direction)
- [ ] Dead end (add to `feedback_no_recheck_confirmed_dead.md`)

## Next Steps

1. <Action 1>
2. <Action 2>

## Files

- `submissions/<competition>/<experiment_name>.csv`
- `code/<experiment_name>.py`
- `logs/<experiment_name>.log`