---
name: kaggle-experiment-sop
description: |
  Standard Operating Procedure for ANY Kaggle experiment or competition entry.
  Execute sequentially from Phase 0 to Phase 3. Each phase has explicit
  checklist items with pass/fail criteria. Covers: memory recall, competition
  type identification, public kernel evaluation, baseline establishment,
  controlled iteration, OOF/LB validation, submission format verification,
  and post-experiment knowledge crystallization.
  This SOP synthesizes 4+ months and 20+ competitions of experience into
  a single repeatable workflow.
---

# Kaggle Experiment Standard Operating Procedure

> **Purpose**: Every Kaggle experiment — whether a new competition, a kernel fork,
> or a parameter sweep — follows this SOP. It prevents wasted quota, repeated
> mistakes, and lost knowledge.

---

## Phase 0: Pre-Experiment (10 minutes)

### 0.1 Recall Relevant Experience

```
□ AutoMem recall: query for competition name, task type, metric
□ File memory: grep ~/.claude/projects/*/memory/ for keywords
□ Skills: check skills/examples/ for matching trigger conditions
□ Dead ends: read feedback_no_recheck_confirmed_dead.md (if exists)
```

**Pass criterion**: Can name ≥1 past competition or skill relevant to current task.
**Fail action**: Proceed with caution — first time on this task type.

### 0.2 Identify Competition Type

```
□ Read competition Overview + Evaluation + Rules
□ Classify into one of 6 types:

  Type 1 — Standard Tabular (CSV upload, deterministic score)
  Type 2 — Code Competition (notebook required, hidden test)
  Type 3 — Simulation (TrueSkill Bayesian, agent vs agent)
  Type 4 — Research (custom metric, domain-specific)
  Type 5 — Playground (synthetic data, swag prize)
  Type 6 — LLM Benchmark (open-source model, prompt engineering)
```

**Pass criterion**: Type identified. Strategy selected (see kaggle-competition-type-strategy).
**Why it matters**: Simulation → don't re-submit. Code → find artifacts first.

### 0.3 Evaluate Public Kernel Landscape

```
□ kaggle kernels list --competition <name> --sort-by scoreDescending
□ Read top 5 kernel titles + first cell
□ Estimate: best_public_score × 0.6 = my_expected_baseline
□ Check: does best public use external data? artifacts? special tricks?
□ Record: best_public_score, best_public_approach
```

**Pass criterion**: Know the best public score and approach.
**Decision point**:
- If best public > 0.95 × top LB → fork is the right strategy
- If best public < 0.80 × top LB → significant custom work needed
- If <5 public kernels → less proven, higher custom value

### 0.4 Assess Resource Constraints

```
□ GPU quota remaining? (30h/week, shared across ALL competitions)
□ Submission quota? (typically 5/day)
□ Deadline distance? (<2 weeks = prioritize, >1 month = explore)
□ Local compute available? (Mac/PC specs, Docker, Python env)
```

**Pass criterion**: Know how many submissions and GPU hours you can spend.

---

## Phase 1: Baseline Establishment (30 min — 2 hours)

### 1.1 Fork Best Public Kernel

```
□ kaggle kernels pull <best_public_kernel_ref>
□ Read the notebook structure (cells, dependencies, datasets)
□ For Code Competition: read find_artifacts() function
  → identify ALL required datasets (models, wheels, data)
□ Create kernel-metadata.json with ALL dataset_sources
□ Push to Kaggle: kaggle kernels push
□ Wait for COMPLETE status
```

**Common failures**:
- ERROR: "module not found" → missing library wheel dataset
- ERROR: "file not found" → missing artifact dataset
- ERROR: "expected 400 nets, found 0" → missing model bundle
- Fix: search `kaggle datasets list --search "<missing keyword>"`, add to dataset_sources

**Pass criterion**: Kernel status = COMPLETE.

### 1.2 Submit Baseline

```
□ For Standard/Playground: submit output CSV directly
  → kaggle competitions submit -f submission.csv
□ For Code Competition: kernel must COMPLETE (not ERROR)
  → If kernel COMPLETEs, go to kernel page → "Submit to Competition"
  → If ERROR, fix dependencies first (Phase 1.1)
□ For Simulation: submit agent tar.gz
  → DO NOT re-submit after this (see Phase 2.5)
□ Record: baseline_LB_score, baseline_OOF_score (if available)
```

**Pass criterion**: Submission appears in submissions list with a score.
**Validation**: baseline_LB ≥ 0.6 × best_public_score. If lower → format error.

### 1.3 Build Local Evaluation (If Applicable)

```
□ Simulation competitions: build CPU eval harness
  → Load multiple agents in isolated temp directories
  → Round-robin games using competition engine (free, no quota)
  → N ≥ 400 games per matchup for statistical significance
□ Tabular competitions: set up 5-fold StratifiedKFold
  → Match fold structure to test set distribution
  → Run adversarial validation (train-vs-test classifier)
□ Code competitions: verify submission.csv format locally
  → Check column names, row count, dtypes, no NaN/inf
```

**Pass criterion**: Can evaluate changes locally before spending quota.

---

## Phase 2: Controlled Iteration (repeated per experiment)

### 2.1 State Hypothesis

```
□ Write 1-sentence hypothesis: "X will improve metric by Y because Z"
□ Check: does this contradict any dead-end memory?
□ Check: is this fundamentally different from last experiment?
□ If same approach as last 3 attempts → STOP, pivot
```

**Rule**: ONE variable per experiment. Bundling changes = uninterpretable.

### 2.2 Implement Change

```
□ Modify exactly ONE thing:
  - One new feature (not five)
  - One hyperparameter (not a grid)
  - One model addition (not an ensemble redesign)
  - One preprocessing change (not a pipeline rewrite)
□ All other variables held constant
```

### 2.3 Validate Locally (Before Submitting)

```
Step A — OOF Stability:
  □ OOF score improved vs baseline?
  □ OOF std across folds < 0.01? (if not, unreliable)
  □ If OOF worse → abort, don't submit

Step B — Adversarial Validation (if tabular):
  □ Train-vs-test AUC ≈ 0.50? → distributions aligned, proceed
  □ AUC > 0.55? → distribution shift exists, be cautious
  □ AUC > 0.60? → significant shift, consider purification

Step C — Submission Format Check:
  □ Column names match sample submission EXACTLY (case-sensitive)
  □ Row count matches sample submission
  □ No NaN, inf, or null values
  □ Data types correct (int vs float, string vs numeric)
  □ For probabilistic: values in [0, 1]
  □ For regression: target scale matches (log vs raw)

Step D — OOF/LB Gap Assessment:
  □ Historical gap for this competition type?
    - Tabular N>10K: expect <0.005 gap
    - Tabular N<1K: expect 0.03-0.09 gap (OOF overestimates)
    - Simulation: OOF has NO predictive value for LB
  □ If this submission's expected gap > 3× historical → investigate
```

**Pass criterion**: ALL of A, B, C, D pass.
**Fail action**: Fix issue or abort. Do NOT submit failed validation.

### 2.4 Submit and Record

```
□ Submit
□ Wait for score
□ Record in experiment log:
  | Experiment | Change | OOF | LB | Gap | Verdict |
  |-----------|--------|-----|-----|-----|---------|
  | v2 | +feature_x | 0.951 | 0.948 | -0.003 | improvement |

□ Verdict:
  - breakthrough: LB improved > baseline + 3× historical gap
  - marginal: LB improved but within noise
  - hit ceiling: LB unchanged despite OOF improvement
  - dead end: LB worse or no change after 3+ attempts
```

### 2.5 The 3-Strike Rule

```
If 3 consecutive OOF improvements (>0.001 each) do NOT improve LB:
  → STOP tuning this approach
  → You are overfitting the OOF
  → Pivot to fundamentally different approach:
    - New feature family (not incremental additions)
    - New Model Family (LGB → NN, or GBDT → Transformer)
    - External Data (highest ROI lever, ~7× self-training)
    - Different Public Kernel (if current is suboptimal)
```

### 2.6 Simulation-Specific Rules (CRITICAL)

```
If competition type = Simulation (TrueSkill):
  □ DO NOT re-submit to "test" — each re-submit resets μ to 600
  □ DO NOT judge agent by submit-day score (always misleading)
  □ DO submit only genuinely different agents (different deck/architecture)
  □ Wait 3-7 days between submissions for σ to decrease
  □ Latest 2 submissions count for final — choose wisely
  □ Near deadline: submit 2 copies of best agent (high-roll strategy)
```

### 2.7 Code Competition-Specific Rules

```
If competition type = Code Competition:
  □ Kernel must COMPLETE (not ERROR) for submission to score
  □ If kernel crashes mid-pipeline:
    → submission.csv may exist but won't be scored
    → Fix: wrap risky cells in try/except, or delete them
    → Verify: kernel status = COMPLETE before celebrating
  □ Pipeline trimming is valid: delete Pipeline B if Pipeline A works
  □ Test locally impossible (Linux .so on Mac) — rely on Kaggle kernel runs
  □ Each kernel push costs ~5-10 min compute — plan accordingly
```

---

## Phase 3: Knowledge Crystallization (10 minutes, post-experiment)

### 3.1 Identify Outcome

```
□ Was this a success, failure, marginal, or dead end?
□ What was the SINGLE most important factor? (not 5 factors — ONE)
□ Can this factor be expressed as a 1-sentence rule?
```

### 3.2 Extract Pattern

```
□ Rule: "When X, do Y, because Z"
□ Evidence: specific numbers from this experiment
□ Scope: when does this apply? when does it NOT?
□ Counter-examples: any cases where this rule fails?
```

### 3.3 Classify and Store

```
□ feedback (anti-pattern): "Don't do X" → feedback_*.md
  Example: "Re-submitting converged agents resets TrueSkill μ to 600"

□ learned (success pattern): "Do X for result Y" → learned in SKILL.md
  Example: "CatBoost-heavy blend (0.7 CAT weight) beats equal blend"

□ reference (location pointer): "X is at Y" → reference_*.md
  Example: "ravaghi/wellbore-geology-prediction-artifacts has data/train.csv"

□ L3 principle (cross-domain): abstract rule → principles in SKILL.md
  Example: "0.6 × BEST_PUBLIC ≥ 0.8 × OPTIMAL_OWN"
```

### 3.4 Store to Memory

```
□ File-based: write to ~/.claude/projects/*/memory/
  - Frontmatter: name, description, type
  - Body: rule + evidence + when to apply

□ AutoMem (if running):
  curl -X POST http://localhost:8001/memory \
    -H "Authorization: Bearer $TOKEN" \
    -d '{"content":"...", "tags":["..."]}'

□ Update MEMORY.md index with one-line pointer
```

### 3.5 Update Skills (If Reusable)

```
□ Does this pattern apply to FUTURE competitions?
  - YES → create or update skills/examples/<name>/SKILL.md
  - NO (competition-specific) → just memory file

□ Skill quality checklist:
  - Description has specific trigger conditions ("Use when: (1)...")
  - Evidence section cites real numbers
  - Anti-patterns section lists what NOT to do
  - Cross-references related skills via [[name]]
```

### 3.6 Forget Stale Knowledge

```
□ Is this memory superseded by a newer finding?
□ Does the referenced file/path still exist?
□ Is the competition still active?
□ If any answer is NO → UPDATE or DELETE the memory
```

---

## Quick Reference Card

### Daily Decision: Should I Submit?

```
Is this a simulation competition?
  YES → Have 3+ days passed since last submit?
         YES → Is this a genuinely different agent? → Submit
         NO  → Wait. Don't reset convergence.
  NO  → Did OOF improve AND pass format check AND pass adversarial validation?
         YES → Submit (1 quota)
         NO  → Don't submit. Fix the issue first.
```

### Emergency: LB Dropped Significantly

```
1. Is this a simulation? → Check meta shift (not a bug)
2. Same code as before? → TrueSkill variance (wait 3 days)
3. Code changed? → Check submission format (NaN, column names)
4. OOF was good but LB bad? → Distribution shift (adversarial validation)
5. Pipeline crashed? → Kernel ERROR = no score (fix pipeline)
```

### Quota Budget Rule

```
Total available = 5 × days_until_deadline
Reserve 20% for final week.
Spend remaining 80%:
  40% on baseline + public fork
  30% on verified improvements (passed all Phase 2.3 checks)
  10% on ensemble experiments
  20% reserved (emergencies, final blend, high-roll)
```

---

## Validation Checklist Summary

| Check | When | Pass Criterion | Fail Action |
|-------|------|---------------|-------------|
| AutoMem recall | Phase 0.1 | ≥1 relevant memory found | Proceed cautiously |
| Competition type | Phase 0.2 | Type identified (1-6) | Read more rules |
| Public kernel scan | Phase 0.3 | Best public score known | Enter competition first |
| Baseline submit | Phase 1.2 | LB ≥ 0.6×best_public | Fix format/dependencies |
| OOF stability | Phase 2.3A | std < 0.01 across folds | More data / different folds |
| Adversarial validation | Phase 2.3B | AUC ≈ 0.50 | Purify or remove shift features |
| Format check | Phase 2.3C | All 5 format items pass | Fix before submitting |
| OOF/LB gap | Phase 2.3D | Gap ≤ 3× historical | Investigate root cause |
| 3-strike check | Phase 2.5 | <3 OOF improvements without LB gain | Pivot to new approach |
| Crystallization | Phase 3 | Memory written + indexed | Knowledge lost between sessions |
