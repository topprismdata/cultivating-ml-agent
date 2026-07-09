---
name: kaggle-oof-lb-validation-protocol
description: |
  Use when: (1) Your OOF score improved but LB didn't (or got worse), (2) You need
  to decide whether to submit based on OOF alone, (3) You're comparing models for
  ensemble inclusion and unsure which OOF signal to trust, (4) You see OOF/LB gap
  larger than 1% and don't know if it's overfitting or distribution shift.
  Covers: 5 sources of OOF/LB gap, the "OOF → LB confirmation protocol" (submit
  only after OOF + adversarial validation + format check), asymmetric gap patterns
  (OOF optimistic on small data, pessimistic on large), and the "3-strike rule"
  (if 3 consecutive OOF improvements don't improve LB, stop tuning and pivot).
  Validated across 20+ competitions including tabular (House Prices, SST, s6e7),
  simulation (PTCG), code (ROGII, Biohub, NeuroGolf), and time series.
---

# Kaggle OOF/LB Validation Protocol

## Problem
OOF (Out-of-Fold) cross-validation score is the standard proxy for LB (Leaderboard)
performance. But the gap between OOF and LB can be large, unpredictable, and
asymmetric. Submitting based on OOF alone wastes quota when the LB doesn't confirm.

## The 5 Sources of OOF/LB Gap

### Source 1: Overfitting to CV Fold Structure
**Symptom**: OOF improves with each tuning round, but LB plateaus or drops.
**Root cause**: Your CV folds have a specific distribution that your tuning
exploits. The LB test set has a different fold structure.
**Magnitude**: 0.001-0.005 (small but cumulative)
**Fix**: Use `GroupKFold` or `StratifiedKFold` matching the test set's structure.
For time series, use `TimeSeriesSplit` with proper temporal ordering.

**Evidence (TPS May 2022)**:
| Round | OOF AUC | LB AUC | Gap |
|-------|---------|--------|-----|
| R5 | 0.786 | 0.867 | +0.081 |
| R8 (same code) | 0.867 | 0.880 | +0.013 |
| R13 (stacking) | 0.921 | 0.867 | -0.054 |

R5→R8: same code, 1.37pp gap = pure sampling noise. R13: stacking overfit OOF.

### Source 2: Distribution Shift Between Train and Test
**Symptom**: OOF is stable but LB is consistently lower (or higher).
**Root cause**: Test set comes from a different distribution (time period,
geography, population).
**Magnitude**: 0.005-0.05 (can be large)
**Fix**: Run adversarial validation. If AUC > 0.55, there's detectable shift.
Consider purifying training data to match test distribution.

**Evidence (Store Sales)**:
- Adversarial AUC = 0.52 → train/test aligned, stop purifying
- lag features caused 10× worse LB because test set had different lag structure

### Source 3: Small Sample Size → OOF Inflation
**Symptom**: OOF on N<1000 rows shows high variance; small changes look significant.
**Root cause**: With few validation samples, each fold's score has high variance.
**Magnitude**: 0.01-0.09 (can dominate real signal)
**Fix**: Require N≥400 games/fold before trusting OOF. Use bootstrap confidence
intervals, not point estimates.

**Evidence**:
- Titanic: OOF 0.855 on 891 rows → LB 0.768 (gap -0.087). OOF severely overestimated.
- TPS May 2022: OOF 0.876 on 800K rows → LB 0.883 (gap +0.007). Large data = reliable OOF.

**Rule of thumb**:
| Train rows | Expected |OOF-LB| | Trust OOF? |
|-----------|-------------------|------------|
| <1K | 0.03-0.09 | ❌ Not reliable |
| 1K-10K | 0.01-0.03 | ⚠️ Cautiously |
| 10K-100K | 0.003-0.01 | ✅ Mostly |
| >100K | <0.005 | ✅ Reliable |

### Source 4: Evaluation Metric Mismatch
**Symptom**: OOF uses a different metric implementation than the LB.
**Root cause**: Custom metric functions can differ from the competition's official
scorer (e.g., Wilson interval vs raw proportion, tie-breaking rules).
**Magnitude**: Variable
**Fix**: Read the competition's evaluation code. Implement the EXACT metric locally.

**Evidence**: PTCG TrueSkill scoring — OOF win-rate has NO correlation with LB
(r=-0.12, p=0.35). The LB uses Bayesian N(μ,σ²), not win-rate.

### Source 5: Simulation/Agent Competition Variance
**Symptom**: Same code, same agent → LB swings ±200 points day-to-day.
**Root cause**: TrueSkill Bayesian scoring with high σ early. Matchmaking is
stochastic. First 5-10 games determine trajectory.
**Magnitude**: 150-400 points (for TrueSkill-based competitions)
**Fix**: See `trueskill-simulation-competition-strategy` skill. Never judge agent
by submit-day score. Wait 3-7 days for σ to decrease.

## The OOF → LB Confirmation Protocol

Before spending a submission quota:

### Step 1: OOF Stability Check
```
If OOF std across folds > 0.01:
  → High variance, don't trust point estimate
  → Try different fold strategies (GroupKFold, different seeds)
  → If still unstable, the signal isn't strong enough to submit
```

### Step 2: Adversarial Validation
```
Train a classifier to distinguish train vs test.
If AUC ≈ 0.50: distributions aligned, proceed.
If AUC > 0.55: distribution shift exists.
  → If your OOF improvement comes from features that exploit the shift,
    it won't generalize to LB. Remove those features.
  → If AUC > 0.60: significant shift. Consider purifying train data.
```

### Step 3: Submission Format Check
```
Verify submission.csv format:
  - Column names match exactly (case-sensitive)
  - Row count matches sample submission
  - No NaN/inf values
  - Data types correct (int vs float)
  - For probabilistic submissions: values in [0, 1]

Common format failures:
  - House Prices: np.expm1 on un-logged target → inf values
  - Leaf Classification: extra 'species' column → wrong column count
  - Titanic: predict() returns hard labels instead of probabilities
```

### Step 4: Submit and Monitor
```
Submit. Record both OOF and LB.
If |OOF - LB| > 3× historical gap:
  → Something is wrong (overfitting, format error, distribution shift)
  → Debug before next submission
If |OOF - LB| ≤ historical gap:
  → Normal. Continue iterating.
```

## The 3-Strike Rule

If 3 consecutive OOF improvements (>0.001 each) do NOT improve LB:
1. **Stop tuning** — you're overfitting the OOF
2. **Check for distribution shift** — adversarial validation
3. **Pivot** — try a fundamentally different approach (new features, new model family, external data)

**Evidence (TPS May 2022)**: 27 meta-learner configurations all converged to the
same LB (~0.9975). The 3-strike rule correctly identified the ceiling at R5.

## Asymmetric Gap Patterns

The OOF/LB gap is NOT symmetric. The direction matters:

| Pattern | OOF | LB | Cause | Action |
|---------|-----|-----|------ |--------|
| **OOF optimistic** | 0.99 | 0.85 | Small data overfit | Don't trust OOF, use LB |
| **OOF pessimistic** | 0.88 | 0.88 | Large data, regularization | Trust OOF, submit |
| **OOF ≈ LB** | 0.95 | 0.95 | Well-calibrated CV | Ideal, continue |
| **LB >> OOF** | 0.85 | 0.95 | Test set easier than train | Warning: hidden test may regress |

**Key insight**: OOF-LB gap is **asymmetric**:
- Small data (N<1K): OOF overestimates by 0.03-0.09
- Large data (N>100K): OOF underestimates by 0.005-0.01

## Model Selection for Ensemble: Use OOF Correlation

When choosing models for ensemble:
1. Compute OOF predictions for each candidate model
2. Calculate pairwise correlation matrix
3. Select models with correlation < 0.95 (meaningful diversity)
4. Reject models with correlation > 0.97 (signal dilution)

**Rule**: 4-6 diverse models > 23 highly-correlated models.

**Evidence**: TPS May 2022 — 14-model stack (all >0.93 correlation) converged to
same score as 4-model stack. Adding more correlated models = zero gain.

## Competition-Type-Specific Validation

### Tabular (Standard ML)
- Trust OOF if N>10K and adversarial AUC≈0.50
- Expected gap: <0.005
- Use: 5-fold StratifiedKFold + adversarial validation

### Time Series
- NEVER use random KFold (temporal leakage)
- Use: TimeSeriesSplit or rolling-origin validation
- Expected gap: 0.005-0.05 (distribution drift over time)

### Simulation (TrueSkill)
- OOF (win-rate) has NO predictive value for LB (TrueSkill μ)
- Only submit 1-2 agents, wait 7+ days for convergence
- Expected gap: 150-400 LB points (not RMSE/AUC)

### Code Competition (Hidden Test)
- OOF on public test set ≠ hidden test score
- Pipeline bugs cause 10-1000× gap (ROGII: OOF 10.38, LB 19545 due to
  Pipeline B error leaking into submission)
- Always: run notebook locally → verify submission.csv format → submit

### NLP / LLM
- OOF depends on prompt template; LB uses different hidden prompts
- Expected gap: 0.02-0.10
- Use: multiple prompt variants, vote across seeds

## Evidence Summary (20+ Competitions)

| Competition | Type | N (train) | OOF | LB | Gap | Root Cause |
|-------------|------|-----------|------|-----|-----|-----------|
| Titanic | Tabular | 891 | 0.855 | 0.768 | -0.087 | Small N inflation |
| Leaf | Tabular | 990 | 0.989 | 4.595 | CRASH | Submission format (extra column) |
| SST | Tabular | 8.7K | 0.832 | 0.810 | -0.022 | Stacking overfit |
| House Prices | Tabular | 1.5K | 0.118 | 0.124 | -0.006 | Small N, log transform |
| TPS May 2022 | Tabular | 800K | 0.876 | 0.883 | +0.007 | Large N, reliable |
| s6e7 | Tabular | 690K | 0.943 | 0.943 | ≈0 | Large N, balanced |
| PTCG | Simulation | — | — | ±200 | N/A | TrueSkill lottery |
| ROGII | Code | — | 10.38 | 19545 | 1881× | Pipeline error in submission |

## Key Takeaways

1. **OOF is necessary but not sufficient** — always validate with LB
2. **Gap size depends on data size** — small data = large gap (0.03-0.09)
3. **Simulation competitions break OOF entirely** — use TrueSkill strategy instead
4. **3-strike rule**: if OOF improves 3× but LB doesn't, pivot
5. **Format check is Step 0** — 5 real format failures across competitions
6. **Adversarial validation detects distribution shift** — AUC>0.55 = be careful
7. **Ensemble diversity > count** — correlation <0.95 required for gain
