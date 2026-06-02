# 16 Cross-Domain ML Principles

> Adapted from `cultivating-ml-agent/AGENTS.md` and validated through 8 MLE-Bench competitions.

## Core Principles

### 1. work_smart_not_hard
**External data ROI is 7x self-training.** Before tuning any hyperparameters, ask: "What external data could I add?"
- ✅ Jigsaw Toxic: XLM-RoBERTa pretrained on toxic data → +0.003 AUC
- ❌ Anti-pattern: spending 2 hours on hyperparameter search before checking public data sources

### 2. local_optimum_trap
**If 3 attempts yield <0.0001 improvement, pivot.** Don't tune the 4th parameter.
- ✅ Confirmed: TPS May 2022 meta-learner convergence (27 variants → same value)
- Action: after 3 incremental attempts fail, write down the structural limitation, then try a new direction

### 3. simple_diagnostic
**1-line diagnostic > complex debugging.** Examples:
- Check base model correlation matrix before stacking
- Run `get_submission_check` before submitting
- Compute adversarial validation AUC to know if cleaning is needed

### 4. quality_over_quantity
**4-6 high-quality models > 23 weak models.** Signal dilution is real.
- ✅ 4-model stack (corr 0.92) > 23-model stack (corr 0.97+)
- Anti-pattern: adding correlated models hoping for averaging benefits

### 5. adversarial_validation_limitation
**Adversarial validation AUC ≈ 0.50 means stop cleaning.** Train/test already aligned.
- ✅ Text Normalization confirmed: AUC 0.50 → no cleaning needed
- Action: only clean when AUC > 0.55

### 6. signal_dilution
**High-correlation models hurt stacking.** Correlation > 0.97 = clone.
- ✅ Cross meta-learner blend: 0 gain
- Action: cap at 5-7 base models, ensure pairwise corr < 0.95

### 7. bundle_causes_uninterpretable
**Don't change multiple variables in one experiment.** Cannot attribute success/failure.
- ✅ v54/v55 combo: stacking + threshold + pseudo-label all at once → uninterpretable
- Action: A/B test one variable per experiment, log all

### 8. metric_asymmetry
**Asymmetric penalties determine strategy.** Jaccard/RMSLE need different approaches than accuracy.
- ✅ Jaccard trim negative-impact tokens before scoring
- Action: study the metric formula first

### 9. controlled_variable
**Bundle variables → not interpretable.** Each experiment tests ONE hypothesis.
- ✅ Always Start Simple: baseline → add 1 thing → measure
- Anti-pattern: "I changed 5 things and it got better" — can't replicate

### 10. consensus_anchor
**>99% consensus on edge cases = systematic error.** Look for mislabeled examples.
- ✅ S6E4 R15: 1374 disagreements, 6.6% supported, others were systematic error
- Action: investigate the "obvious wrong" cases

### 11. integration_dominance
**How you combine > what you combine.** Ensemble method matters more than components.
- ✅ H&M: mixed ranking -0.0025 vs stratified +0.00035
- Action: spend more time on integration than adding new models

### 12. ground_truth_encoding
**When deterministic rules dominate, encode them explicitly.**
- ✅ Text Normalization: lookup table > statistical model
- Action: check if the target follows deterministic rules before statistical modeling

### 13. distribution_mismatch
**Optimizing for non-deployment conditions → predictable failure.**
- ✅ Store Sales: lag features optimized for training but fail at inference
- Action: simulate inference conditions during training

### 14. diagnosis_first
**Understand why before fixing.** Bad fix is worse than no fix.
- ✅ ffill before model fix: simple diagnostics reveal the issue
- Anti-pattern: trying fixes without understanding root cause

### 15. start_simple
**Baseline first, then complexity.** Most "improvements" are baseline bugs.
- ✅ Aerial Cactus: GBDT (0.9978) > complex CNN on CPU
- Action: 30 minutes on baseline > 3 hours on complex architecture

### 16. extract_knowledge
**After breakthrough: extract and persist.** No knowledge = no future benefit.
- ✅ Each competition has dedicated memory file
- Anti-pattern: breakthrough then forget

## Domain-Specific (BRAIN alpha mining)

| Principle | Core |
|-----------|------|
| signal_purity_over_data_scarcity | Cross-source fusion reduces PROD_CORR |
| structure_match_determines_floor | Per-share/volume/abs need different treatment |
| alpha_is_beta_residual | After neutralization, crash = no alpha |
| complexity_increases_fragility | 8+ operators = OS cliff |
| friction_cost_is_life_death_boundary | TVR > 0.70 = no submission |
| abstraction_level_determines_transferability | Layer 3 principles > specific experience |

## How to Use

1. Before starting a new task, scan this list
2. If a principle applies, follow it explicitly
3. If you discover a new principle, add it here
4. If a principle is wrong, document why and remove