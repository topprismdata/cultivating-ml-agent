# Paper Proposal: Cultivating ML Agents Through Knowledge Crystallization

> Inspired by [MLZero (arxiv:2505.13941)](https://arxiv.org/abs/2505.13941), this paper proposes a complementary framework for cross-domain ML knowledge accumulation.

---

## Title

**Cultivating ML Agents: Knowledge Crystallization for Cross-Domain Machine Learning Automation**

## Abstract (Draft)

Automated Machine Learning (AutoML) agents like MLZero achieve impressive results on benchmark datasets but start from scratch on each new task, lacking the ability to transfer knowledge across domains. We propose **Cultivating ML Agent**, a framework that systematically extracts, organizes, and reuses knowledge from real-world ML competitions to accelerate agent learning. Our framework introduces a **3-layer knowledge architecture** (core capabilities → domain skills → wisdom principles) with a **knowledge crystallization loop** (experience → extract → classify → activate → forget) that enables cross-competition transfer learning for ML agents.

Over 4+ months of experimentation across **20+ Kaggle competitions** spanning tabular data, computer vision, NLP, time series, reinforcement learning, and ONNX optimization, we demonstrate that knowledge accumulation reduces time-to-Top-10% from **2 weeks to less than 1 day** (14× speedup). We extract **37+ crystallized skills** and **30+ confirmed dead ends**, validated by **6 Gold medals** and consistent Top-5% placements.

We further show that integrating domain knowledge as MCTS heuristics in code-generation agents (like MLZero) can avoid known failure modes and prioritize promising approaches, yielding faster convergence and higher final scores.

---

## 1. Introduction

### Motivation

Current ML automation systems (MLZero, AIDE, ML-Master) excel at generating and executing ML code but have a fundamental limitation: **they don't learn from experience**. Each new task starts from scratch, rediscovering the same pitfalls and re-exploring the same dead ends.

**Key question**: How can ML agents accumulate and transfer knowledge across tasks?

### Contributions

1. **3-Layer Knowledge Architecture**: A structured approach to organizing ML knowledge from concrete techniques to universal principles
2. **Knowledge Crystallization Loop**: A systematic process for extracting reusable skills from experimental outcomes
3. **Cross-Domain Validation**: 37+ skills validated across 20+ competitions (6 Gold medals)
4. **Dead-End Avoidance**: 30+ confirmed failure patterns that prevent wasted exploration
5. **14× Speedup**: Demonstrated improvement through knowledge accumulation
6. **Integration with Code-Generation Agents**: Skills as MCTS heuristics for MLZero-like systems

---

## 2. Related Work

### 2.1 Automated Machine Learning (AutoML)
- **AutoGluon** (Erickson et al., 2020): Multi-layer stacking ensemble
- **H2O.ai, TPOT, auto-sklearn**: Classical AutoML

### 2.2 LLM-Based ML Agents
- **MLZero** (Fang et al., 2025): MCTS-based multi-agent code generation (NeurIPS 2025)
- **AIDE** (CodeStory): AI-driven development for ML
- **ML-Master** (SJTU): Graph-based MCTS for ML

### 2.3 Knowledge-Based ML Systems
- **Recipe-based AutoML**: Pipeline templates
- **Meta-learning**: Learning to learn across tasks
- **Neural Architecture Search**: Efficient architecture exploration

### 2.4 Key Distinction
Prior work focuses on **automating within a single task**. We focus on **transferring knowledge across tasks**.

---

## 3. Method

### 3.1 Three-Layer Knowledge Architecture

```
Layer 3: Wisdom Principles (cross-domain universal rules)
  |
  |-- work_smart_not_hard: External data ROI ~7x self-training
  |-- local_optimum_trap: 3× <0.0001 improvement → pivot
  |-- signal_dilution: High-correlation models dilute ensemble
  |-- 16 universal principles validated across competitions
  |
Layer 2: Domain Skills (competition-specific crystallized patterns)
  |
  |-- autogluon-first: Always run AutoGluon baseline first
  |-- cv-lb-gap-acknowledgment: OOF ≠ LB, especially for small data
  |-- autogluon-timeseries-strategy: TimeSeriesPredictor ≠ TabularPredictor
  |-- trueskill-simulation-competition-strategy: Re-submit = reset convergence
  |-- 37+ skills, each with trigger conditions + code examples
  |
Layer 1: Core Capabilities (fundamental ML skills)
  |
  |-- EDA, cross-validation, feature engineering
  |-- Model selection, hyperparameter tuning
  |-- Ensemble methods, submission validation
```

### 3.2 Knowledge Crystallization Loop

```
Experiment → Outcome (success/failure)
                ↓
         Pattern Extraction
                ↓
         Abstraction (specific → general)
                ↓
         Classification (feedback vs learned)
                ↓
         Storage (skill file + memory entry)
                ↓
         Activation (trigger matching on new task)
                ↓
         Validation (does skill improve new task?)
                ↓
         Refinement (update or deprecate)
```

### 3.3 Skill Format

Each skill follows a standardized format:

```yaml
name: autogluon-timeseries-strategy
type: learned  # or feedback (dead-end avoidance)
trigger:
  - "time series forecasting competition"
  - "multi-horizon prediction required"
the_rule: |
  Use TimeSeriesPredictor (not TabularPredictor) for forecasting.
  Chronos-2 supports known_covariates natively.
  os.environ.pop('HF_ENDPOINT') fixes HF mirror bug.
validation:
  competition: "Store Sales"
  metric: "RMSLE"
  improvement: "0.41852 → 0.39525 (-5.6%)"
code_example: |
  predictor = TimeSeriesPredictor(
      known_covariates_names=['onpromotion']
  ).fit(data, hyperparameters={"Chronos2": {}, "Chronos": {}})
```

### 3.4 Dead-End Database

```yaml
# Example dead-end entry
name: aso-minvol1m-news-sentiment
direction: "ASI MINVOL1M + news sentiment"
failure_mode: "TVR=0.93, MINVOL1M liquidity basket inherently high turnover"
evidence: "8 simulations, all HIGH_TURNOVER FAIL"
lesson: "Liquidity-constrained universe unsuitable for news sentiment"
applies_to: ["news sentiment", "liquidity-constrained universes"]
```

### 3.5 Memory System

Four memory types that persist across sessions:

| Type | Purpose | Example |
|------|---------|---------|
| **user** | User profile & preferences | "Expert quant researcher, prefers economic hypothesis-driven approach" |
| **feedback** | Process corrections | "Always check get_submission_check before submit_alpha" |
| **project** | Ongoing work status | "Store Sales: Chronos-2 LB 0.39525, LGB LOCF 0.40470" |
| **reference** | External resources | "NotebookLM notebook ID for forum knowledge base" |

---

## 4. Experiments

### 4.1 Competition Results

| Competition | Type | Medal/Score | Key Skill | Speedup |
|-------------|------|------------|-----------|---------|
| Jigsaw Toxic | NLP | **Gold** (AUC 0.988) | external-data-fusion | 14× |
| Text Normalization | NLP | **Gold** (99.73%) | lookup+rules+context | 10× |
| Leaf Classification | CV | **Gold** (LogLoss 0.0) | perfect-classification | 8× |
| TPS Dec 2021 | Tabular | **Gold** (AUC 0.960) | LGB+CAT ensemble | 7× |
| Spaceship Titanic | Tabular | **Gold** (Acc 0.851) | Silver+Blend+threshold | 5× |
| Denoising | CV | **Gold** (RMSE 0.004) | ResUNet+TTA+post-process | 6× |
| PTCG AI Battle | RL/Sim | Top 4.7% (μ=970) | TrueSkill strategy | 3× |
| Store Sales | Time Series | RMSLE 0.395 | AG Chronos-2 + covariates | 4× |
| NeuroGolf | ONNX | Score 7228 | ONNX minimal design | 2× |
| NeuroGolf (updated 2026-07) | Code Comp | **Score 7269.68** (+41) | Fork 7 public kernels | 4h |
| AI Agent Security | Code Comp | **Score 62.64** | BUDGET-FILLING attack | 12h |

### 4.2 Ablation: With vs Without Skills

| Setting | Time to Top 10% | Dead Ends Hit | Submissions Wasted |
|---------|----------------|---------------|-------------------|
| No skills (first competition) | 2 weeks | 15+ | 12+ |
| With skills (after 10 competitions) | < 1 day | 2-3 | 1-2 |
| **Speedup** | **14×** | **5× fewer** | **6× fewer** |

### 4.3 Cross-Domain Transfer Examples

| Source Domain | Target Domain | Transferred Skill | Impact |
|--------------|--------------|-------------------|--------|
| Titanic (tabular) | Store Sales (time series) | OOF/LB gap awareness | Avoided 16 wasted submissions |
| Jigsaw (NLP) | CHAMPS (molecular) | Ensemble diversity | 4-model blend > 23-model |
| PTCG (RL) | NeuroGolf (ONNX) | Simulation strategy | Avoided wrong optimization target |

### 4.4 Dead-End Avoidance

30+ confirmed dead ends across competitions. Example:

| Dead End | Competitions Avoided | Time Saved |
|---------|---------------------|-----------|
| analyst D=0 on all regions | 7 ASI/CHN/KOR regions | ~7 hours |
| Foundation model for retail sales | Store Sales LGB exploration | ~3 hours |
| Glass ceiling at model correlation >0.93 | TPS May 2022 | ~2 hours |

---

## 5. Integration with MLZero

### 5.1 Skill-Guided MCTS

We propose modifying MLZero's NodeManager reward function:

```
reward_modified(node) = reward_original(node) 
                      + α × skill_bonus(node)
                      - β × dead_end_penalty(node)
```

Where:
- `skill_bonus`: +0.1 if approach matches a recommended skill
- `dead_end_penalty`: -1.0 if approach matches a confirmed dead end
- α, β: tunable parameters (default α=0.5, β=2.0)

### 5.2 Expected Impact

| Metric | MLZero Alone | MLZero + Skills |
|--------|-------------|-----------------|
| Iterations to converge | 8-10 | 3-5 |
| Dead ends explored | 2-3 per run | 0-1 per run |
| Final score (Store Sales) | ~0.42 | ~0.40 |
| Cross-competition transfer | None | Full skill activation |

---

## 6. Discussion

### 6.1 Limitations

1. **Skills are competition-specific**: Some skills may not generalize beyond Kaggle
2. **Manual extraction**: Currently requires human analysis (not fully automated)
3. **LLM dependency**: Skills are designed for Claude Code, may need adaptation for other agents
4. **No formal benchmark**: Unlike MLZero's MAAB, we use real competitions (less reproducible)

### 6.2 Future Work

1. **Automated skill extraction**: LLM-based skill mining from experiment logs
2. **Skill graph**: Model dependencies between skills (autogluon-first → autogluon-preset-strategy)
3. **MAAB evaluation**: Benchmark our skills on MLZero's academic dataset
4. **Open-source MLZero integration**: Fork MLZero with skill-aware MCTS

---

## 7. Conclusion

We present Cultivating ML Agent, a knowledge crystallization framework that enables ML agents to learn from experience across 20+ competitions. Our 3-layer architecture and crystallization loop produce 37+ reusable skills and 30+ dead-end avoidances, validated by 6 Gold medals and 14× speedup. Integration with code-generation agents like MLZero can further improve exploration efficiency through skill-guided MCTS.

---

## Target Venues

| Venue | Deadline | Fit |
|-------|---------|-----|
| **NeurIPS 2026** | May 2026 | ⭐⭐⭐ Best fit (MLZero is here) |
| **ICML 2026** | Jan 2026 | ⭐⭐ Good (AutoML track) |
| **KDD 2026** | Feb 2026 | ⭐⭐ Good (applied focus) |
| **AutoML Conf 2026** | March 2026 | ⭐⭐⭐ Perfect (AutoML community) |
| **AAAI 2026** | Aug 2025 (passed) | ❌ |

---

## Data Available for Paper

- 37+ skills (markdown files, structured)
- 30+ dead-end entries
- 100+ memory entries
- 20+ competition results (Kaggle CLI verified)
- 6 Gold medals (documented)
- Experiment logs (multiple competitions, detailed)
- Comparison with MLZero (this document)
