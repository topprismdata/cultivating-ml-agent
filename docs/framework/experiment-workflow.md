# Experiment Workflow SOP

## Experiment Naming & Documentation

### Script Header (mandatory)

Every script starts with:

```python
"""R{NN}: {Brief description}

Changes from R{NN-1}:
  1. {Change 1}
  2. {Change 2}

Usage: python scripts/run_r{NN}_{name}.py
"""
```

The "Changes from R{NN-1}" section is critical for traceability. Every experiment builds on a previous one.

### Experiment Log

Track experiments in `docs/experiment_log.md`:

```markdown
| Run | CV Score | LB Score | Key Changes | Date |
|-----|----------|----------|-------------|------|
| R01 | 0.842    | 0.835    | Baseline LightGBM | 2024-01-15 |
| R02 | 0.856    | 0.849    | + Rolling features | 2024-01-16 |
```

## Experiment Lifecycle

### 1. Create Script

Copy `framework/script_template.py` to `scripts/run_r{NN}_{name}.py` and implement:

```python
# Stage 1: Data Loading
# Stage 2: Feature Engineering
# Stage 3: Model Training
# Stage 4: Prediction & Submission
```

### 2. Run & Validate

```bash
python scripts/run_r01_baseline.py
```

The script automatically:
- Loads config from `config.yaml`
- Runs `validate_pipeline()` after feature engineering
- Logs experiment to MLflow via `log_experiment()`
- Saves validated submission

### 3. Check Results

```bash
# View MLflow experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Or check run details
mlflow runs list --experiment-id 0
```

### 4. Submit to Leaderboard

```python
from utils.submission import submit_to_kaggle

submit_to_kaggle(
    "outputs/submissions/submission_r01_baseline.csv",
    cfg.slug,
    message="R01 baseline LightGBM"
)
```

### 5. Record LB Score

```python
from pipeline.mlflow_utils import log_lb_score

log_lb_score(run_id="abc123", lb_score=0.835)
```

### 6. Update Experiment Log

Add row to `docs/experiment_log.md`.

### 7. Extract Skills

After a significant finding, run `claudeception` to extract reusable patterns.

## Decision Framework

### When to start a new experiment (new R{NN})

- Changing the model architecture
- Adding a significant new feature category
- Changing the validation strategy
- Changing the data preprocessing approach

### When to iterate within an experiment

- Tuning hyperparameters
- Adjusting feature thresholds
- Small bug fixes
- Post-processing tweaks

### Red Lines (from AUTOSOTA_ML_PROCESS)

1. **No target leakage**: Always run `validate_pipeline()`
2. **No test data in training**: No fitting encoders on test
3. **Reproducible**: Same script + same data = same result
4. **One metric**: Use the competition metric, not proxies
5. **CV before LB**: Never optimize directly on LB
6. **Document everything**: Every experiment has a docstring and log entry

## Multi-Agent Coordination

When using the mentor + specialist agent pattern:

1. **Mentor** assigns tasks to specialists
2. **Feature Engineer** creates features, validates with `validate_pipeline()`
3. **Model Trainer** trains models, logs with `log_experiment()`
4. **Ensemble Specialist** combines predictions, validates submission
5. **Code Reviewer** checks final script against plan

All specialists use the same `CompetitionConfig` and shared modules.
