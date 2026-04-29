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
from .mlflow_utils import (
    start_experiment,
    log_experiment,
    log_lb_score,
    setup_mlflow,
    ExperimentContext,
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
    "start_experiment",
    "log_experiment",
    "log_lb_score",
    "setup_mlflow",
    "ExperimentContext",
]
