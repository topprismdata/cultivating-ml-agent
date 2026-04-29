"""Competition configuration loader.

Nested dataclass structure for type-safe config management.
Supports YAML loading, per-experiment overrides, and CLI parameter override.

Usage:
    cfg = CompetitionConfig.from_yaml("config.yaml")
    cfg = apply_overrides(cfg, {"model.lgb_params.learning_rate": 0.01})
"""
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Nested config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Data source configuration."""
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    target_col: Optional[str] = None
    id_col: Optional[str] = None
    submission_format: str = "csv"  # csv|space_separated_ids|...
    exclude_cols: list[str] = field(default_factory=list)
    sample_submission: str = "sample_submission.csv"


@dataclass
class ValidationConfig:
    """Cross-validation strategy configuration."""
    strategy: str = "stratified"  # stratified|kfold|time_based|group
    n_folds: int = 5
    time_col: str = ""
    val_size_weeks: int = 20  # for time_based strategy
    group_col: str = ""       # for group strategy


@dataclass
class ModelConfig:
    """Model parameters with merge support for partial overrides."""
    default: str = "lightgbm"
    lgb_params: dict = field(default_factory=lambda: {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
    })
    xgb_params: dict = field(default_factory=lambda: {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bybytree": 0.8,
        "verbosity": 0,
    })
    cb_params: dict = field(default_factory=lambda: {
        "iterations": 1000,
        "learning_rate": 0.1,
        "depth": 8,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "early_stopping_rounds": 50,
        "task_type": "CPU",
        "thread_count": -1,
        "random_seed": 42,
        "verbose": 100,
    })


@dataclass
class ExperimentConfig:
    """Experiment-level settings."""
    random_state: int = 42
    n_seeds: int = 1
    seeds: list[int] = field(default_factory=lambda: [42])
    smoke_test: bool = False


@dataclass
class MLflowConfig:
    """MLflow tracking configuration."""
    experiment_name: str = ""
    tracking_uri: str = "sqlite:///mlflow.db"


@dataclass
class SubmissionConfig:
    """Submission limits and validation."""
    max_per_day: int = 5
    validate_before_submit: bool = True


@dataclass
class CompetitionConfig:
    """Full per-competition configuration.

    Top-level identity fields + nested sub-configs.
    Loaded from config.yaml, supports per-experiment overrides.
    """
    # Internal: config file path for resolving project_root
    _config_path: Optional[Path] = field(default=None, repr=False, compare=False)

    # Identity
    name: str = ""
    slug: str = ""
    url: str = ""
    task_type: str = "tabular"  # tabular|timeseries|recommendation|cv|nlp
    metric: str = ""
    metric_direction: str = "maximize"

    # Nested configs
    data: DataConfig = field(default_factory=DataConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    submission: SubmissionConfig = field(default_factory=SubmissionConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "CompetitionConfig":
        """Load configuration from a YAML file."""
        yaml_path = Path(yaml_path)
        with open(yaml_path) as f:
            raw = yaml.safe_load(f) or {}

        # Top-level competition fields
        comp = raw.get("competition", {})

        # Nested sub-configs
        data_cfg = DataConfig(**_filter_dataclass_kwargs(raw.get("data", {}), DataConfig))
        val_cfg = ValidationConfig(**_filter_dataclass_kwargs(raw.get("training", {}), ValidationConfig))
        exp_cfg = ExperimentConfig(**_filter_dataclass_kwargs(raw.get("experiment", {}), ExperimentConfig))
        sub_cfg = SubmissionConfig(**_filter_dataclass_kwargs(raw.get("submission", {}), SubmissionConfig))

        # Model config with partial override support
        model_cfg = _build_model_config(raw.get("models", {}))

        # MLflow config
        mlflow_raw = raw.get("mlflow", {})
        mlflow_cfg = MLflowConfig(
            experiment_name=mlflow_raw.get("experiment_name", ""),
            tracking_uri=mlflow_raw.get("tracking_uri", "sqlite:///mlflow.db"),
        )

        return cls(
            _config_path=yaml_path,
            name=comp.get("name", ""),
            slug=comp.get("slug", ""),
            url=comp.get("url", ""),
            task_type=comp.get("task_type", "tabular"),
            metric=comp.get("metric", ""),
            metric_direction=comp.get("metric_direction", "maximize"),
            data=data_cfg,
            validation=val_cfg,
            model=model_cfg,
            experiment=exp_cfg,
            mlflow=mlflow_cfg,
            submission=sub_cfg,
        )

    @property
    def project_root(self) -> Path:
        """Resolve project root from config file location."""
        if self._config_path is not None:
            return self._config_path.parent
        return Path.cwd()

    def get_data_path(self, filename: str, subdir: str = "raw") -> Path:
        return self.project_root / "data" / subdir / filename

    def get_output_path(self, filename: str, subdir: str = "submissions") -> Path:
        return self.project_root / "outputs" / subdir / filename

    def get_oof_path(self, run_name: str) -> Path:
        return self.project_root / "outputs" / "oof" / f"oof_{run_name}.csv"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _filter_dataclass_kwargs(raw: dict, cls) -> dict:
    """Filter dict keys to only those accepted by a dataclass."""
    from dataclasses import fields as dc_fields
    valid = {f.name for f in dc_fields(cls)}
    return {k: v for k, v in raw.items() if k in valid}


def _build_model_config(models_cfg: dict) -> ModelConfig:
    """Build ModelConfig with partial override merge support."""
    cfg = ModelConfig()
    cfg.default = models_cfg.get("default", "lightgbm")

    for param_field, model_key in [
        ("lgb_params", "lightgbm"),
        ("xgb_params", "xgboost"),
        ("cb_params", "catboost"),
    ]:
        default_val = ModelConfig.__dataclass_fields__[param_field].default_factory()
        overrides = models_cfg.get(model_key, {})
        if isinstance(overrides, dict):
            setattr(cfg, param_field, {**default_val, **overrides})
        else:
            setattr(cfg, param_field, default_val)
    return cfg


def apply_overrides(cfg: CompetitionConfig, overrides: dict) -> CompetitionConfig:
    """Apply per-experiment overrides to a CompetitionConfig.

    Supports dot-notation for nested access:
        {"model.lgb_params.learning_rate": 0.01}
        {"validation.n_folds": 3}
        {"experiment.smoke_test": True}
    """
    for key_path, value in overrides.items():
        parts = key_path.split(".")
        obj = cfg
        for part in parts[:-1]:
            if isinstance(obj, dict):
                obj = obj.get(part, {})
            else:
                obj = getattr(obj, part)

        final_key = parts[-1]
        if isinstance(obj, dict):
            obj[final_key] = value
        else:
            setattr(obj, final_key, value)
    return cfg


def apply_cli_overrides(cfg: CompetitionConfig, cli_args: list[str]) -> CompetitionConfig:
    """Apply CLI --override key=value pairs.

    Usage: --override validation.n_folds=3 --override experiment.random_state=123
    """
    overrides = {}
    for arg in cli_args:
        if "=" not in arg:
            continue
        key, value = arg.split("=", 1)
        # Try to cast value
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass  # keep as string
        overrides[key] = value
    return apply_overrides(cfg, overrides)


def load_config(
    config_path: str | Path | None = None,
    overrides: dict | None = None,
    cli_args: list[str] | None = None,
) -> CompetitionConfig:
    """Load config with optional overrides.

    Priority (highest to lowest):
    1. CLI args (--override key=value)
    2. Per-experiment overrides dict
    3. config.yaml values
    4. Dataclass defaults
    """
    if config_path is None:
        env_path = os.environ.get("PROJECT_ROOT")
        if env_path:
            config_path = Path(env_path) / "config.yaml"
        else:
            config_path = Path("config.yaml")

    cfg = CompetitionConfig.from_yaml(config_path)

    if overrides:
        cfg = apply_overrides(cfg, overrides)
    if cli_args:
        cfg = apply_cli_overrides(cfg, cli_args)

    return cfg
