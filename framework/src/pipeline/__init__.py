"""Pipeline utilities: validation, MLflow tracking, evaluation gates."""

from .validate import (
    validate_pipeline,
    validate_features,
    validate_submission,
    evaluation_gate,
    classify_failure,
    PipelineValidationError,
    EvaluationGateError,
    FAILURE_CATEGORIES,
)
try:
    from .mlflow_utils import (
        start_experiment,
        log_experiment,
        log_lb_score,
        setup_mlflow,
        ExperimentContext,
    )
except ImportError:  # mlflow not installed; tracking utilities unavailable
    pass

from .oof import build_oof_frame
from .splits import (
    make_folds,
    stratified_folds,
    kfold_folds,
    time_based_folds,
    group_folds,
)

__all__ = [
    "validate_pipeline",
    "validate_features",
    "validate_submission",
    "evaluation_gate",
    "classify_failure",
    "PipelineValidationError",
    "EvaluationGateError",
    "FAILURE_CATEGORIES",
    "build_oof_frame",
    "make_folds",
    "stratified_folds",
    "kfold_folds",
    "time_based_folds",
    "group_folds",
    "start_experiment",
    "log_experiment",
    "log_lb_score",
    "setup_mlflow",
    "ExperimentContext",
]
