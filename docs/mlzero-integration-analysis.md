# MLZero vs Cultivating ML Agent — Framework Comparison & Integration Plan

> Comparison between [MLZero (AutoGluon Assistant)](https://github.com/autogluon/autogluon-assistant) (NeurIPS 2025) and [Cultivating ML Agent](https://github.com/topprismdata/cultivating-ml-agent), with concrete integration strategies.

---

## 1. What is MLZero?

**MLZero (AutoGluon Assistant)** is Amazon's official multi-agent LLM system for end-to-end ML automation. Published at NeurIPS 2025 ([arxiv:2505.13941](https://arxiv.org/abs/2505.13941)).

**Core innovation**: Node-Based Manager with **Monte Carlo Tree Search (MCTS)** for exploring the ML solution space. LLM generates code, AutoGluon executes, MCTS decides which branches to explore next.

**Architecture** (10 specialized agents):

```
DataPerceptionAgent → automatic EDA (understands data structure)
ToolSelectorAgent   → selects AutoGluon tool (tabular/multimodal/timeseries)
TaskDescriptorAgent → understands task type (classification/regression/forecasting)
CoderAgent          → generates Python/bash code from LLM
ExecuterAgent       → executes generated code
ErrorAnalyzerAgent  → analyzes errors → generates debug code
RerankerAgent       → ranks solutions by validation score
RetrieverAgent      → retrieves similar examples from tool tutorials
MetaPromptingAgent  → optimizes prompts across iterations
NodeManager         → MCTS coordinator (exploration vs exploitation)
```

**Tool Registry**: AutoGluon tabular, multimodal, timeseries + FlagEmbedding + Qwen3 + generic ML tools

**Benchmark**: MAAB (Multimodal AutoML Agent Benchmark, 140GB, academic datasets)

**Interfaces**: CLI (`mlzero -i data_folder`), Web UI, MCP server

---

## 2. What is Cultivating ML Agent?

**Cultivating ML Agent** is a knowledge crystallization framework for training AI agents from ML novice to competition Top 10%. Built from 4+ months of real Kaggle competition experience across 20+ competitions.

**Core innovation**: **3-layer knowledge architecture** with explicit skill extraction from real competition outcomes.

```
L3: Wisdom Principles (cross-domain universal rules)
    - work_smart_not_hard, local_optimum_trap, signal_dilution
    - 16 universal principles validated across competitions

L2: Domain Skills (37+ crystallized patterns)
    - autogluon-first, autogluon-preset-strategy, catboost-first
    - cv-lb-gap-acknowledgment, kaggle-experiment-sop
    - autogluon-timeseries-strategy, trueskill-simulation-strategy
    - Each skill: trigger conditions + the rule + how to apply + code example

L1: Core Capabilities (EDA, CV, feature engineering, model training)
```

**Memory System**: Structured vault with 4 types (user, feedback, project, reference) + 100+ memory entries

**Execution Model**: Human + AI agent collaborate via Claude Code, skills auto-activate based on problem type

**Competition Coverage**: Tabular, CV, NLP, time series, RL (PTCG), ONNX (NeuroGolf), simulation, LLM benchmark

---

## 3. Side-by-Side Comparison

| Dimension | MLZero | Cultivating ML Agent |
|-----------|--------|---------------------|
| **Philosophy** | Zero human intervention | Knowledge accumulation + human-AI collaboration |
| **Search Strategy** | MCTS (blind exploration of solution space) | Skill-guided (domain knowledge narrows search) |
| **Knowledge** | Implicit (in LLM weights + tool registry) | **Explicit** (37+ skills, 3-layer architecture, 100+ memories) |
| **Feedback Loop** | Execution results → MCTS reward signal | OOF/LB → skill extraction → memory vault |
| **LLM Role** | Code generation + agent coordination | Skill matching + experiment planning + error analysis |
| **AutoML Backend** | AutoGluon only (built-in) | Any (AutoGluon, LightGBM, CatBoost, manual) |
| **Agent Architecture** | 10+ specialized agents (perception, coding, execution) | 3-layer knowledge + 5-stage learning loop |
| **Benchmark** | MAAB (academic, 140GB controlled datasets) | Real Kaggle competitions (20+ practical cases) |
| **Memory** | File-based iteration logs per run | Persistent structured memory (survives across sessions) |
| **Cross-competition Transfer** | ❌ Starts from scratch each time | ✅ Skills auto-activate from prior experience |
| **Known Dead Ends** | ❌ MCTS must rediscover | ✅ Explicit dead-end list (30+ confirmed failures) |
| **GPU Support** | Native (Docker + GPU) | External (remote GPU server) |
| **1st Place Methodology** | ❌ Not encoded | ✅ autogluon-preset-strategy, LightGBM+lag=0.37 |
| **OOF/LB Gap Awareness** | ❌ Not modeled | ✅ cv-lb-gap-acknowledgment skill |
| **Error Decomposition** | ❌ Binary success/fail | ✅ 4-level decomposition (family × day × bias × variance) |
| **Time Series Multi-horizon** | ❌ Would make same recursive bug | ✅ Learned: recursive error accumulation kills LB |
| **Reproducibility** | Config + random seed | SOP + experiment log + naming convention |

---

## 4. Key Insight: Complementary, Not Competing

**MLZero is an execution engine. We are a knowledge system.**

```
MLZero answers:  "How to automatically generate and execute ML code?"
We answer:       "What strategies work, why they work, and when they fail."
```

### What MLZero Does Better

1. **Fully autonomous**: No human needed, give data → get results
2. **Code generation**: LLM writes Python code, not just recommendations
3. **MCTS exploration**: Systematic search of solution space
4. **Multi-modal**: Handles text, image, tabular in one pipeline
5. **Docker isolation**: Safe execution of LLM-generated code
6. **Academic validation**: NeurIPS 2025 paper, MAAB benchmark

### What We Do Better

1. **Cross-competition learning**: Skills transfer across competitions
2. **Known dead ends**: Don't waste time on confirmed failures
3. **Domain-specific strategies**: Chronos-2 covariates, TrueSkill strategy, ONNX optimization
4. **Error analysis methodology**: Per-family × per-day decomposition
5. **OOF/LB gap awareness**: Critical for Kaggle (MLZero doesn't model this)
6. **Competition-specific tactics**: Submission format validation, quota management, latest-2 rule
7. **Persistent memory**: 100+ structured memories across sessions
8. **Real Kaggle results**: 6 Gold medals, practical competition coverage

---

## 5. Integration Strategies

### Strategy A (Lightweight): Skills as MLZero Tool Prompts

Register our skills in MLZero's tool registry:

```python
from autogluon.assistant.tools_registry import register_tool

register_tool(
    name="kaggle-sop",
    tutorials=[
        "skills/examples/kaggle-experiment-sop/SKILL.md",
        "skills/examples/autogluon-preset-strategy/SKILL.md",
        "skills/examples/cv-lb-gap-acknowledgment/SKILL.md",
    ],
    prompt_template=(
        "Follow the Kaggle Experiment SOP: "
        "1. Run AutoGluon baseline first (autogluon-first skill) "
        "2. One variable per experiment "
        "3. Always validate OOF vs LB "
        "4. 3 strikes < 0.0001 improvement → pivot"
    )
)
```

**Effort**: Low (add prompt files to MLZero's tool registry)
**Impact**: Medium (MLZero gets our domain knowledge as context)

### Strategy B (Deep): Skill-Aware MCTS

Modify MLZero's NodeManager to use our skills as MCTS heuristics:

```python
class SkillAwareNodeManager(NodeManager):
    def __init__(self, *args, skills_registry=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.skills = skills_registry or load_skills()

    def calculate_reward(self, node):
        """Override MCTS reward with skill-guided bonus."""
        base_reward = node.validation_score

        # Bonus: approach matches recommended skill
        for skill in self.skills:
            if skill.matches(node.method, node.features):
                base_reward += skill.confidence * 0.1

        # Penalty: approach is in known dead-ends
        for dead_end in self.skills.dead_ends:
            if dead_end.matches(node.method):
                base_reward -= 1.0

        return base_reward

    def select_promising_approaches(self, candidates):
        """Filter MCTS candidates using skill knowledge."""
        filtered = []
        for c in candidates:
            if not self.skills.is_dead_end(c.method):
                filtered.append(c)
        return filtered
```

**Effort**: Medium (modify NodeManager, create skills→MCTS adapter)
**Impact**: High (MLZero avoids dead ends, explores promising directions)

### Strategy C (Practical): MLZero Baseline + Skills Optimization

```
Phase 1: MLZero auto-generates baseline (5 min)
    → mlzero -i competition_data --provider anthropic --max_iterations 5
    → AutoGluon TabularPredictor / TimeSeriesPredictor baseline

Phase 2: Skills identify optimization direction (1 min)
    → "This is time series → activate autogluon-timeseries-strategy"
    → "OOF 0.85 but LB 0.77 → activate cv-lb-gap-acknowledgment"
    → "Single model → activate kaggle-optimal-blending"

Phase 3: MLZero executes skill-guided improvements (10 min)
    → Add covariates (Chronos-2 native support)
    → Try different presets (good_quality → high_quality → best_quality)
    → Ensemble models

Phase 4: Human/AI does final optimization (30 min)
    → Error decomposition (per-family × per-day)
    → Per-family bias correction
    → Feature engineering guided by domain knowledge
```

**Effort**: Low (use both tools sequentially, no code changes)
**Impact**: Highest (leverages both frameworks' strengths)

---

## 6. Concrete Integration Example: Store Sales

| Step | Tool | Action | Result |
|------|------|--------|--------|
| 1 | MLZero | `mlzero -i store_sales_data` | AutoGluon TabularPredictor baseline (LB ~0.42) |
| 2 | Our skill | `autogluon-timeseries-strategy` activates | "Use TimeSeriesPredictor, not TabularPredictor" |
| 3 | MLZero | Re-runs with TimeSeriesPredictor | LB ~0.41 (Chronos-2 zero-shot) |
| 4 | Our skill | `cv-lb-gap-acknowledgment` warns | "OOF may not match LB for multi-horizon" |
| 5 | Our skill | `feedback_foundation_model_vs_gbdt` | "For retail with lags, LightGBM+lag > Chronos" |
| 6 | MLZero | MCTS explores LightGBM with lags | OOF 0.37 |
| 7 | Our skill | Dead-end warning | "Direct lag≥17 too weak; recursive has error accumulation; LOCF best compromise" |
| 8 | Final | LOCF LightGBM + Chronos-2 ensemble | LB ~0.40 |

**Without integration**: MLZero alone would get ~0.42 (tabular AutoGluon, no time series awareness)
**With integration**: ~0.40 (time series + Chronos-2 + LOCF LightGBM, guided by our skills)

---

## 7. Implementation Roadmap

### Phase 1: Skill Export Format (1 week)

Create a standard format for exporting skills as MLZero tool prompts:

```yaml
# skill_export.yaml
name: autogluon-timeseries-strategy
trigger: "time series forecasting competition"
recommendation: "Use TimeSeriesPredictor with Chronos-2"
dead_ends:
  - "TabularPredictor on time series (no multi-horizon awareness)"
  - "Recursive forecasting with error accumulation"
best_practices:
  - "known_covariates_names for Chronos-2"
  - "make_future_data_frame() for test covariates"
  - "os.environ.pop('HF_ENDPOINT') for HF mirror bug"
code_template: |
  from autogluon.timeseries import TimeSeriesPredictor
  predictor = TimeSeriesPredictor(
      target='target', prediction_length=16, freq='D',
      known_covariates_names=['onpromotion'],
  ).fit(data, hyperparameters={"Chronos2": {}, "Chronos": {}})
```

### Phase 2: MLZero Tool Registry Integration (2 weeks)

1. Fork MLZero
2. Add our skills as tool tutorials in `tools_registry/`
3. Modify `ToolSelectorAgent` to consider skill triggers
4. Modify `NodeManager` reward function with skill bonuses/penalties

### Phase 3: Benchmark on Real Kaggle (2 weeks)

1. Run MLZero alone vs MLZero+Skills on 5 competitions
2. Measure: LB score, iterations to converge, dead-end avoidance
3. Target competitions: Store Sales, House Prices, Spaceship Titanic, TPS Dec, s6e7

### Phase 4: MCP Integration (1 week)

1. Package skills as MCP server
2. MLZero can query skills via MCP
3. Claude Code can invoke MLZero via MCP

---

## 8. Academic Positioning

| Aspect | MLZero | Cultivating ML Agent | Integrated |
|--------|--------|---------------------|------------|
| **Contribution** | MCTS-based code generation | Knowledge crystallization framework | Skill-guided MCTS |
| **Novelty** | Pure MCTS exploration | Explicit skill extraction | Domain knowledge as MCTS heuristic |
| **Weakness** | No cross-task learning | No autonomous code generation | Best of both |
| **Benchmark** | MAAB | Kaggle competitions | Both MAAB + Kaggle |
| **Publication** | NeurIPS 2025 | Open-source (GitHub) | Potential joint paper |

---

## 9. Summary

```
MLZero: "I can automatically write and run ML code"
Us:     "I know what code to write and why"

Together: "I know what to write, why, and I can write it automatically"
```

The integration creates a system that:
1. **Learns from experience** (our skills) → avoids known dead ends
2. **Generates code automatically** (MLZero) → no manual coding
3. **Systematically explores** (MCTS) → doesn't miss good approaches
4. **Transfers knowledge** (our memory) → gets better over time

This is strictly better than either framework alone.
