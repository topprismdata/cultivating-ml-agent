# ⛔ Confirmed Dead Ends — DO NOT RECHECK

**Read this file before starting any new direction or experiment.** These are approaches that have been tested and shown to fail repeatedly. Don't waste cycles on them.

## Format

```
| Direction | Failure Mode | Evidence | Date |
|-----------|-------------|----------|------|
| <direction> | <why it fails> | <experiment + result> | <YYYY-MM-DD> |
```

## Entries

| Direction | Failure Mode | Evidence | Date |
|-----------|-------------|----------|------|
| CatBoost on large tabular | fold AUC ~0.966 vs LGB ~0.994; 10x slower | TPS May 2022 800K rows | 2026-05-17 |
| Multi-seed meta-learner averaging | No complementarity between fold splits | 27 variants → same OOF | 2026-05-17 |
| Cross meta-learner blending | LR + LGB blend adds no value | 0 gain across 4 competitions | 2026-05-17 |
| Nelder-Mead on private test | Overfits to private LB | Priv improvement ≠ generalizes | 2026-05-17 |
| Stacking when corr > 0.95 | Adds < 0.001 AUC | TPS May 2022 confirmed | 2026-05-17 |
| Pseudo-labeling iterated | Threshold lowers, noise accumulates | Single round OK, multi round fails | 2026-05-24 |
| Adversarial cleaning when AUC ≈ 0.50 | Train/test already aligned, no gain | Text Normalization | 2026-05-24 |
| Training XLM-R-large on MPS | 10-50x slower than CUDA | Jigsaw / Chaii | 2026-05-16 |

## Adding New Entries

When a direction fails after serious attempt:
1. Document the failure mode precisely
2. Include numerical evidence (OOF/Priv scores)
3. Add to the table above
4. Update this file before the next session starts on the same domain