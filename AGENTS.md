# AGENTS.md — Autonomous ML Learning Agent Instructions

This file guides Claude Code (or any AI coding agent) to autonomously execute ML competition workflows using the Cultivating ML Agent framework.

## Overview

This project provides a **knowledge-driven MLOps framework** for training ML agents from novice to competition-level performance. It combines:

1. **Framework code** (`framework/`) — Reusable Python modules for config, logging, validation, MLflow tracking
2. **Skills library** (`skills/examples/`) — 19 crystallized patterns from real competitions
3. **Main guide** (`docs/cultivating-ml-agent-expert.md`) — Methodology (1088 lines)
4. **Templates** (`templates/`) — Skill creation templates

## When Agent Is Activated

Upon installation, the agent should:

1. Read this file for operational instructions
2. Browse `skills/examples/` for available domain knowledge
3. Use `framework/` modules for pipeline structure
4. Follow the five-stage learning loop for continuous improvement

## Agent Workflow

### Starting a New Competition

Follow this exact sequence:

```
1. Read competition overview (metric, data format, submission format)
2. Copy framework/config_template.yaml → config.yaml, fill in competition details
3. Copy framework/script_template.py → scripts/run_r01_baseline.py
4. Run EDA: data shapes, class balance, missing values, feature types
5. Build baseline (single model, no feature engineering)
6. Submit to LB — this is your floor score
7. Iteratively improve: R02, R03, ... with clear naming
```

### Per-Experiment Workflow (5-Stage Pipeline)

Every experiment script (`run_r{NN}_{name}.py`) MUST follow this structure:

```
Stage 0: Configuration
  - load_config("config.yaml", overrides={...})
  - get_logger("R{NN}_{name}")
  - argparse: --smoke, --no-mlflow, --override

Stage 1: Data Loading
  - Load train/test/sample_submission
  - Log shapes: log.data_shape("train", df)
  - Validate: validate_pipeline(train, test, cfg)

Stage 2: Feature Engineering
  - Create features
  - Validate: validate_features(df, FEATURE_COLS)
  - Log feature count and names

Stage 3: Model Training
  - Train with CV (5-fold default)
  - Log CV score: log.metric("CV", score)
  - Save model artifact
  - Save OOF predictions for stacking

Stage 4: Prediction + Submission
  - Generate predictions
  - get_submission_filename() for naming
  - validate_and_save() for format checking
  - evaluation_gate() to block regressions
  - Submit to Kaggle LB
```

### Framework Module Map

| Module | Import | Purpose |
|--------|--------|---------|
| `src/config.py` | `from config import CompetitionConfig, load_config` | Type-safe YAML config with CLI overrides |
| `src/utils/logging_utils.py` | `from utils.logging_utils import get_logger` | Timestamped experiment logging |
| `src/utils/metrics.py` | `from utils.metrics import get_metric` | Competition metric functions |
| `src/utils/submission.py` | `from utils.submission import validate_and_save, get_submission_filename` | Submission format validation |
| `src/utils/paths.py` | `from utils.paths import get_competition_dirs` | Standard directory resolution |
| `src/pipeline/validate.py` | `from pipeline.validate import validate_pipeline, validate_features, evaluation_gate` | Data quality + regression gates |
| `src/pipeline/mlflow_utils.py` | `from pipeline.mlflow_utils import start_experiment` | MLflow experiment tracking |

### Integration Pattern (Per-Project)

```
your-competition/
├── config.yaml                  # Competition-specific config
├── scripts/
│   ├── run_r01_baseline.py      # Each experiment is self-contained
│   ├── run_r02_feature_eng.py
│   └── ...
├── src/                         # Local modules (optional)
├── outputs/
│   ├── submissions/             # Kaggle submission files
│   ├── models/                  # Saved model artifacts
│   └── oof/                     # Out-of-fold predictions
├── data/                        # Competition data (gitignored)
└── README.md                    # Experiment log
```

## Autonomous Learning Rules

### Rule 1: Always Start Simple

```
Baseline (R01) → Single model, zero feature engineering
Feature Engineering (R02-R04) → Add ONE technique per experiment
Multi-Model (R05-R07) → Add model diversity
Ensemble (R08+) → Combine best models
```

### Rule 2: One Hypothesis Per Experiment

Every `run_r{NN}` must have:
- A clear hypothesis: "Adding pairwise TE will improve CV by X"
- A single variable change from previous experiment
- If it doesn't work, document WHY and move on

### Rule 3: CV Before LB

- Always compute CV score before submitting
- If CV doesn't improve, don't submit (wastes daily quota)
- Monitor CV-LB gap: if gap grows, you're overfitting

### Rule 4: Extract + Persist Knowledge

After any non-trivial finding, run `/claudeception` to extract a skill AND persist to the Obsidian Memory Vault:

```python
# After experiment completes, immediately persist:
# 1. Read the vault to know current state:
with open("~/obsidian/ml-agent-memory/dashboard.md") as f:
    print(f.read())

# 2. Update competition record:
#    → Append experiment result to ~/obsidian/ml-agent-memory/competitions/{name}.md

# 3. Update principles if cross-domain insight found:
#    → Edit ~/obsidian/ml-agent-memory/principles/16-principles.md

# 4. Update skills if anti-pattern discovered:
#    → Edit ~/obsidian/ml-agent-memory/skills/{skill-name}.md
```

**Anti-patterns to record immediately** (these save hours of wasted effort):
- "净化后 CV 没改善" → adversarial-validation.md
- "多模型 ensemble 相关性 >0.97" → ensemble_method.md
- "lag feature 全 NaN 导致低预测" → time_series.md

### Rule 5: Consult Memory Before Acting

At these decision points, read the vault FIRST before writing any code:

| Trigger | File to Read | What to Look For |
|---------|-------------|-----------------|
| Starting a new competition | `dashboard.md` | Cross-competition patterns, highest ROI techniques |
| CV-LB gap > 0.01 | `skills/adversarial-validation.md` | AUC≈0.50 → stop purification, try GroupKFold |
| Feature engineering | `dashboard.md` | External data fusion ROI ~7x over self-training |
| Ensemble plateau | `principles/16-principles.md` | Quality over quantity: 4-6 sources > 23 sources |
| Time series lag features | `skills/time_series.md` | `preds.mean()/train.mean()` diagnostic first |

### Rule 6: One Change Per Experiment — Document the Outcome

After experiment, record in `~/obsidian/ml-agent-memory/experiments/experiments.md`:

```markdown
### v{NN}_{name}
- **CV/LB**: 0.xxxx / 0.xxxx
- **做法**: [ONE sentence, what changed]
- **诊断**: [why it worked or didn't]
- **关键洞察**: `short_slug_name`
- **标签**: [comma-separated]
```

Format follows `experiment_log.jsonl` conventions so both machine-readable (JSONL) and human-readable (MD).

## Competition-Specific Guidance

### Tabular Classification/Regression

Key skills to activate:
- `tabular-feature-engineering-patterns` — 7 FE patterns
- `pairwise-target-encoding-strategy` — Interaction encoding
- `sigmoid-smoothing-target-encoding` — High-cardinality encoding
- `kaggle-optimal-blending` — 80/20 blending rule
- `ensemble-source-quality-over-quantity` — Ensemble source selection

Typical progression:
1. Baseline (LightGBM, default params)
2. Target encoding for categoricals
3. Pairwise interaction features
4. Multi-model ensemble (XGB + LGB + CB)
5. Pseudo-labeling (if test labels are predictable)
6. Stacking + threshold optimization
7. External data integration (if available)

### Time Series Forecasting

Key skills:
- `ts-forecasting-stale-lag-methodology` — Complete lag feature guide
- `ts-lag-out-of-sample-trap` — Lag leakage prevention
- `ts-lag-nan-cascade-bug` — NaN handling
- `dense-grid-cv-lb-divergence` — CV strategy for dense grids
- `unified-vs-day-specific-forecasting` — Model architecture choice

Typical progression:
1. Baseline (global model, basic lags)
2. Time features (day-of-week, month, holidays)
3. Lag features with out-of-sample prevention
4. Per-store or per-category models (carefully!)
5. Ensemble of unified + day-specific models
6. Post-processing (clipping, rounding)

### Recommendation Systems

Key skills:
- `kaggle-recommendation-recall-bottleneck` — Recall > ranking insight
- `kaggle-submission-id-reset-index-bug` — Submission format traps

Typical progression:
1. Co-occurrence / popular baseline
2. Repurchase-based recall
3. Item-to-item collaborative filtering
4. GBDT ranking on recall candidates
5. Layered prediction (high-confidence first, fill empty slots)

## Framework Gotchas

These issues were discovered during real competition usage:

1. **`log.metric()` only accepts floats** — Use `log.info()` for strings
2. **`get_data_path()` may resolve to wrong directory** — Use `cfg.project_root / "data_raw"` as fallback
3. **`evaluation_gate()` requires `cv_std`** — Pass `0.0` for single-fold validation
4. **Config `config.py` name collision** — Load local config before adding shared `src/` to `sys.path`
5. **`log_feature_importance()` needs sklearn API** — Raw `lgb.train()` Booster won't work; use `LGBMClassifier`
6. **MLflow SQLite backend** — Add `mlflow.db` to `.gitignore`

## Skill Activation Priority

When encountering a problem, check skills in this order:

1. **Problem-specific** — If the error/situation matches a skill's trigger, use it immediately
2. **Domain-specific** — Activate domain skills (time series, tabular, etc.) at the start of a new competition
3. **Process skills** — Use `claudeception` after breakthroughs, `ml-pipeline-unit-testing` before long runs

## Experiment Naming Convention

```
run_r{NN}_{descriptive_name}.py
```

- `NN`: Zero-padded 2-digit version (R01, R02, ..., R15)
- `name`: kebab-case technique description (e.g., `target_encoding`, `gbdt_framework`)
- Submission: `submission_r{NN}_{name}.csv`

## Quality Gates

Before any submission:
1. `validate_pipeline()` — No target leakage, shape checks
2. `validate_features()` — No NaN/Inf in features
3. `validate_and_save()` — Correct submission format
4. `evaluation_gate()` — CV score meets threshold

After any submission:
1. Log LB score to MLflow: `log_lb_score(run_id, lb_score)`
2. Compare CV vs LB gap
3. If gap > 5%, investigate overfitting
4. Update README with results

## Obsidian Memory Vault Pattern

A filesystem-based knowledge base that persists across agent sessions. Any agent can read it to acquire accumulated experience from previous sessions.

### Structure

```
~/obsidian/ml-agent-memory/              # home-relative, works on all Macs
# OR for iCloud sync (same vault, different absolute path):
#   ~/Library/Mobile Documents/iCloud~md~obsidian/Documents/06_ML_Agent_Memory/
# Agents: use whichever path resolves on the running system.
├── competitions/         # Per-competition structured records
│   ├── s6e5.md           # F1 Pit Stop — best LB 0.9526
│   └── s6e4.md           # Irrigation — best LB 0.98150
├── experiments/          # Experiment timeline with results
│   └── experiments.md    # 28 experiments, cross-competition patterns
├── principles/          # Layer 3 universal principles (cross-domain)
│   └── 16-principles.md # e.g. "Work Smart > Hard Work", "Local Optimum Trap"
├── skills/              # Core skill decision frameworks
│   └── adversarial-validation.md  # Decision tree + AUC interpretation
└── memory/              # (optional, agent-only dir, gitignored)
    ├── experiment_log.jsonl
    └── principles_index.json
```

### Agent Query Pattern

At any decision point, an agent can read relevant files:

```python
# When starting a new competition:
with open("~/obsidian/ml-agent-memory/dashboard.md") as f:
    insights = f.read()

# When diagnosing CV-LB gap:
with open("~/obsidian/ml-agent-memory/competitions/s6e5.md") as f:
    competition_record = f.read()

# When deciding strategy:
with open("~/obsidian/ml-agent-memory/principles/16-principles.md") as f:
    principles = f.read()
```

### When to Update Memory

After completing an experiment or finding a breakthrough:

1. **Experiments**: Append to `experiments/experiments.md` with CV/LB, key changes, key insight
2. **Principles**: If finding transfers across domains, add to `principles/16-principles.md`
3. **Competition record**: Update `competitions/{name}.md` with latest version and findings
4. **Skills**: If anti-pattern discovered, add/update `skills/{skill-name}.md`

### Quick Reference: Decision → File Mapping

| Decision Point | File | Key Content |
|---------------|------|-------------|
| Adversarial validation | `skills/adversarial-validation.md` | AUC 0.50 = stop purification |
| CV strategy | `principles/16-principles.md` P16 | GroupKFold when AUC≈0.50 |
| Feature engineering | `dashboard.md` | External data fusion highest ROI |
| Ensemble failure | `skills/ensemble_method.md` | Model correlation trap |
| Time series lag | `skills/time_series.md` | `preds.mean()/train.mean()` diagnostic |

### Cross-Competition Patterns (from experiments.md)

| Technique | Effective For | Effect | Condition |
|-----------|-------------|--------|-----------|
| Pairwise target encoding | s6e4, s6e5 | +0.001~+0.002 | categorical × numerical |
| External data fusion | s6e4, s6e5 | +0.001~+0.007 | Same target definition |
| Pseudo-labeling | s6e4 | +0.001 single round, -0.001 iterative | Threshold ≥0.90 |
| Polynomial features | House Prices | +0.016 (small dataset) | Rows < 10K |
| Adversarial purification | s6e5 | No benefit if AUC≈0.50 | Check first before filtering |

## Resources

- **Main guide**: `docs/cultivating-ml-agent-expert.md` (1088 lines)
- **Framework config spec**: `docs/framework/config-spec.md`
- **Framework pipeline**: `docs/framework/mlops-pipeline.md`
- **Experiment workflow**: `docs/framework/experiment-workflow.md`
- **Project template**: `docs/framework/project-template.md`
- **Skill templates**: `templates/bug-fix-skill.md`, `templates/knowledge-skill.md`
- **Memory vault**: `~/obsidian/ml-agent-memory/` (see Obsidian Memory Vault Pattern above)
