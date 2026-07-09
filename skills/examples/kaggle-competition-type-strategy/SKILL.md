---
name: kaggle-competition-type-strategy
description: |
  Use when: (1) Entering a new Kaggle competition and unsure what type it is,
  (2) Need to choose between forking public kernels vs building custom,
  (3) Deciding how to allocate submission quota across the competition lifecycle,
  (4) Determining which evaluation/validation strategy to use.
  Covers 6 competition types (Standard Tabular, Code Competition, Simulation,
  Research, Prediction, Playground), each with distinct: submission mechanism,
  scoring system, public kernel value, GPU requirements, and optimal strategy.
  Includes a decision tree for first-48h actions and a quota allocation framework.
  Validated across 20+ competitions Jun-Sep 2026.
---

# Kaggle Competition Type Strategy

## The 6 Competition Types

Kaggle competitions are NOT all the same. The type determines your entire strategy.

### Type 1: Standard Prediction (Tabular)
**Examples**: House Prices, TPS series, Spaceship Titanic, s6e7
**Mechanism**: Upload CSV file. Scored immediately on fixed test set.
**Scoring**: Deterministic (RMSE, AUC, logloss — same formula every time)
**Public kernel value**: ★★★★★ (best public kernel ≈ 95-99% of winning score)
**GPU needed**: Usually no (CPU GBDT sufficient)
**Quota**: 5/day, no convergence delay

**Strategy**:
1. Day 1: Fork best public kernel → submit → establish baseline
2. Day 2-3: AutoGluon best_quality → compare with public → blend if complementary
3. Day 4-7: Feature engineering (highest ROI lever for tabular)
4. Day 8+: Model diversity (LGB+XGB+CAT blend), pseudo-labeling (cautious)
5. Final: Submit best blend, stop tuning 24h before deadline

**Validation**: 5-fold StratifiedKFold + adversarial validation. Trust OOF if N>10K.

### Type 2: Code Competition (Notebook Required)
**Examples**: ROGII Wellbore, Biohub Cell Tracking, NeuroGolf
**Mechanism**: Submit Kaggle Notebook. Runs on hidden test set. No internet.
**Scoring**: Deterministic on hidden test (may differ significantly from public)
**Public kernel value**: ★★★★☆ (fork is main strategy, but artifacts/dependencies)
**GPU needed**: Often yes (UNet, LLM inference, model training)
**Quota**: 5/day or 10/day, kernel run time limit (9-12h)

**Strategy**:
1. Day 1: Read top 10 public kernels. Identify required datasets (artifacts).
2. Day 2: Fork best kernel. Add ALL required datasets as inputs. Push → verify COMPLETE.
3. Day 3: If kernel ERRORs → read log → fix dependencies (see code-competition-artifact-pipeline)
4. Day 4-7: Modify pipeline (trim crashing components, tune parameters)
5. Final: Ensure kernel COMPLETES (ERROR → no score, even if submission.csv exists)

**Critical**: Code competitions do NOT score ERROR-status kernels. A kernel that
crashes mid-pipeline produces NO submission, even if submission.csv was written
before the crash. Always trim or try/except around risky cells.

### Type 3: Simulation Competition (TrueSkill)
**Examples**: PTCG AI Battle, Orbit Wars, Connect-X
**Mechanism**: Submit agent (tar.gz or notebook). Agent plays games against
other agents on the ladder. TrueSkill Bayesian rating N(μ,σ²).
**Scoring**: Probabilistic (μ changes with each game, σ decreases over time)
**Public kernel value**: ★★★☆☆ (fork gets you started, but meta shifts weekly)
**GPU needed**: Usually no (game agents are rule-based or RL)
**Quota**: 5/day, but DON'T use them all — see below

**Strategy** (CRITICAL — read trueskill-simulation-competition-strategy):
1. Day 1-3: Fork best public agent. Build local eval harness (free CPU kernel).
2. Day 4-7: Submit 1-2 agents. DO NOT re-submit to "test" (resets μ to 600).
3. Week 2-4: Let agents converge (σ decreases). Monitor LB but don't act on noise.
4. Week 5+: Near deadline, submit 2 duplicate copies of best agent (high-roll).
5. Final 2 weeks: Kaggle continues running games until convergence.

**Key rule**: Latest 2 submissions count for final. ALL agents keep playing.
LB shows best-ever score, but FINAL uses latest-2 only.

### Type 4: Research Competition
**Examples**: Biohub Cell Tracking, ARC Prize
**Mechanism**: Notebook submission + sometimes written report
**Scoring**: Custom metric (often domain-specific)
**Public kernel value**: ★★☆☆☆ (less community, harder problems)
**GPU needed**: Usually yes (heavy compute for research-grade tasks)
**Quota**: 5/day, long timeline (3+ months)

**Strategy**:
1. Read the domain literature (papers, GitHub repos)
2. Fork best public kernel (often the only option — custom work is months of effort)
3. Focus on pipeline correctness (format, dependencies) before optimization
4. Allocate GPU wisely (30h/week shared across ALL competitions)

### Type 5: Playground Series
**Examples**: s6e1-s6e7 (monthly)
**Mechanism**: Standard prediction, synthetic data, swag prizes (not medals)
**Scoring**: Deterministic (same as Type 1)
**Public kernel value**: ★★★★★ (highly collaborative, best public ≈ winning)
**GPU needed**: No (synthetic data is small/medium)
**Quota**: 5/day, low stakes

**Strategy**:
1. Day 1: AutoGluon best_quality (5-15 min baseline)
2. Day 2: Fork best public → blend with AutoGluon
3. Day 3-5: Feature engineering, model diversity
4. Day 6+: Stop when ceiling reached (3-strike rule)
5. No stress — these are practice competitions

### Type 6: LLM/Agent Benchmark
**Examples**: Industrial Automation Track 1, ARC-AGI-3, ai-agent-security
**Mechanism**: Submit notebook with model/prompt. Evaluated on hidden prompts.
**Scoring**: Accuracy or custom metric
**Public kernel value**: ★★☆☆☆ (LLM space moves fast, public kernels expire)
**GPU needed**: Yes (running open-source LLMs)
**Quota**: 5-10/day, GPU-intensive

**Strategy**:
1. Choose the right open-source model (Qwen, DeepSeek, LLaMA — competition rules)
2. Prompt engineering is 80% of the work
3. Few-shot examples from training data
4. Closed-book setting: no RAG, no tools — rely on model's internal knowledge
5. Test locally with vLLM before submitting

## The First-48-Hours Decision Tree

```
New Competition Found
│
├── Read Overview + Evaluation + Rules (30 min)
│
├── Identify Type
│   ├── Standard Tabular → AutoGluon first, then fork public
│   ├── Code Competition → Find artifacts, fork best public
│   ├── Simulation → Read TrueSkill strategy, build eval harness
│   ├── Research → Read literature, fork best public
│   ├── Playground → AutoGluon, blend, practice
│   └── LLM Benchmark → Choose model, prompt engineer
│
├── Scan Top 10 Public Kernels
│   ├── Sort by score descending
│   ├── Read their approach (title + first cell)
│   ├── Check: does best public use external data? special tricks?
│   └── Estimate: 0.6×BEST_PUBLIC baseline
│
├── Submit Best Public Kernel (1 quota)
│   ├── Validates format + baseline
│   └── If LB < 0.6×BEST_PUBLIC → something is wrong (check format)
│
└── Plan Quota Allocation
    ├── Week 1: 2-3 submissions (baseline + first improvement)
    ├── Week 2-3: 1-2 submissions/week (only verified improvements)
    └── Final week: 1-2 submissions (high-roll for simulation, final blend for tabular)
```

## Quota Allocation Framework

### For Standard/Playground (5/day)
```
Total submissions available: 5 × (days_until_deadline)
Recommended usage:
  - 30% baseline + public fork exploration
  - 40% verified improvements (OOF + format check before submit)
  - 20% ensemble/blend experiments
  - 10% reserved for emergencies/final day
```

### For Simulation (5/day, but DON'T use)
```
Total submissions available: 5 × (days_until_deadline)
Recommended usage:
  - Week 1: 1-2 (establish agent, start convergence)
  - Week 2-N: 0 (let convergence happen, don't re-submit)
  - Final 2 weeks: 2 (duplicate best agent for high-roll)
  - NEVER: re-submit same agent to "test" (resets convergence)
```

### For Code Competition (5/day)
```
Total submissions available: 5 × (days_until_deadline)
Recommended usage:
  - Day 1-3: 2-3 (fork + fix dependencies + first COMPLETE run)
  - Day 4-7: 2-3 (pipeline optimization, parameter tuning)
  - Week 2+: 1/week (only significant improvements)
  - Final: 1-2 (submission format verification + final model)
```

## GPU Quota Management (30h/week shared)

GPU time is the scarcest resource. It's shared across ALL your competitions.

```
Priority allocation:
1. Active competition with deadline < 2 weeks → 50% GPU
2. Active competition with deadline < 1 month → 30% GPU
3. Exploration/baseline runs → 15% GPU
4. Reserved for emergencies → 5% GPU
```

**Anti-pattern**: running AutoGluon best_quality (8h) on a Playground competition
while a Code competition deadline is 3 days away.

## Evidence

Validated across 20+ competitions:
- Standard: House Prices (0.11750), SST (0.80780), s6e7 (0.94942), TPS May (0.99754)
- Code: ROGII (Pipeline A 10.38), Biohub (v1 fork), NeuroGolf (7228)
- Simulation: PTCG (LB 967.8, rank 219/4164), Orbit Wars (observed)
- Research: ARC-AGI-3 (Duck harness studied)
- Playground: s6e1-s6e7 (6 competitions, mean top 5%)
- LLM: Industrial Automation Track 1 (FMEA knowledge base)
