# Configuration Specification

## Overview

`config.yaml` is the single source of truth for each competition. Loaded into a **nested dataclass** hierarchy: `CompetitionConfig` contains `DataConfig`, `ValidationConfig`, `ModelConfig`, `ExperimentConfig`, `MLflowConfig`, `SubmissionConfig`.

## Sections

### `competition`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Full competition name |
| `slug` | string | yes | URL slug for Kaggle API |
| `url` | string | no | Competition URL |
| `metric` | string | yes | Primary metric name |
| `metric_direction` | string | yes | `"maximize"` or `"minimize"` |
| `task_type` | string | yes | `tabular`, `timeseries`, `recommendation`, `cv`, `nlp` |

### `data` (DataConfig)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `train_file` | string | yes | Training data filename |
| `test_file` | string | yes | Test data filename |
| `target_col` | string/null | yes | Target column name. `null` for recommendation tasks |
| `id_col` | string/null | no | ID column for submission |
| `submission_format` | string | yes | `csv`, `space_separated_ids`, etc. |
| `exclude_cols` | list | no | Columns to exclude from features |
| `sample_submission` | string | no | Sample submission filename |

### `training` (ValidationConfig)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `strategy` | string | `"stratified"` | `stratified`, `kfold`, `time_based`, `group` |
| `n_folds` | int | 5 | Number of CV folds |
| `time_col` | string | "" | Time column for time-based splits |
| `val_size_weeks` | int | 20 | Validation window size (time-based) |
| `group_col` | string | "" | Group column for group K-Fold |

### `experiment` (ExperimentConfig)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `random_state` | int | 42 | Random seed |
| `n_seeds` | int | 1 | Number of seeds for multi-seed averaging |
| `seeds` | list | [42] | Seed values |
| `smoke_test` | bool | false | Quick test mode (1 fold, 100 estimators) |

### `mlflow` (MLflowConfig)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `experiment_name` | string | "" | Falls back to slug |
| `tracking_uri` | string | `"sqlite:///mlflow.db"` | MLflow tracking server URI |

### `submission` (SubmissionConfig)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_per_day` | int | 5 | Kaggle=5, Playground=10 |
| `validate_before_submit` | bool | true | Run validation before kaggle submit |

### `models` (optional, partial override)

```yaml
models:
  default: "lightgbm"     # which model to use by default
  lightgbm:
    learning_rate: 0.01   # overrides default 0.05
    num_leaves: 127       # overrides default 63
    # all other params keep their defaults
  xgboost:
    max_depth: 8
  catboost:
    iterations: 2000
```

## Usage

### Basic loading
```python
from config import CompetitionConfig
cfg = CompetitionConfig.from_yaml("config.yaml")
print(cfg.metric)              # "MAP@12"
print(cfg.data.target_col)     # "label"
print(cfg.validation.n_folds)  # 5
print(cfg.model.lgb_params)    # dict with merged params
```

### With per-experiment overrides
```python
from config import load_config

OVERRIDES = {
    "model.lgb_params.learning_rate": 0.01,
    "validation.n_folds": 3,
}
cfg = load_config("config.yaml", overrides=OVERRIDES)
```

### With CLI args
```python
# python script.py --override validation.n_folds=3 --override experiment.random_state=123
cfg = load_config("config.yaml", cli_args=["validation.n_folds=3"])
```

### Path resolution
```python
from utils.paths import get_competition_dirs
dirs = get_competition_dirs(cfg)
dirs.data_raw.mkdir(parents=True, exist_ok=True)
dirs.submissions  # outputs/submissions/
dirs.oof          # outputs/oof/
```

## project_root Resolution

`project_root` is resolved from the config file location:
- If loaded via `from_yaml("config.yaml")`, `project_root = parent of config.yaml`
- Falls back to `Path.cwd()` if no config path stored
