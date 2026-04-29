"""Competition evaluation metrics.

Common metrics used across Kaggle competitions, collected in one place
for consistent evaluation. Each metric matches the Kaggle evaluation
formula exactly.

Supported metrics:
    - balanced_accuracy: Multiclass balanced accuracy (PS S6E4)
    - map_at_k: Mean Average Precision at K (H&M Recommendations)
    - wmae: Weighted Mean Absolute Error (Walmart Sales)
    - rmsle: Root Mean Squared Log Error
    - rmse: Root Mean Squared Error
    - mae: Mean Absolute Error
"""
import numpy as np
import pandas as pd
from typing import Optional


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute balanced accuracy (average recall per class).

    Used in Playground Series S6E4 (Bedrock Race Classification).

    Args:
        y_true: True labels (integer array).
        y_pred: Predicted labels (integer array).

    Returns:
        Balanced accuracy score in [0, 1].
    """
    classes = np.unique(y_true)
    recalls = []
    for cls in classes:
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        recalls.append((y_pred[mask] == cls).mean())
    return np.mean(recalls) if recalls else 0.0


def map_at_k(
    y_true: list[list[str]],
    y_pred: list[list[str]],
    k: int = 12,
) -> float:
    """Compute Mean Average Precision at K.

    Used in H&M Personalized Fashion Recommendations.

    Args:
        y_true: List of lists, each containing ground-truth article IDs.
        y_pred: List of lists, each containing predicted article IDs (ordered).
        k: Number of top predictions to consider.

    Returns:
        MAP@K score.
    """
    scores = []
    for true_items, pred_items in zip(y_true, y_pred):
        ap = 0.0
        hits = 0
        for i, pred in enumerate(pred_items[:k]):
            if pred in true_items:
                hits += 1
                ap += hits / (i + 1)
        n_true = min(len(true_items), k)
        scores.append(ap / n_true if n_true > 0 else 0.0)
    return np.mean(scores)


def wmae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Compute Weighted Mean Absolute Error.

    Used in Walmart Store Sales Forecasting.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        weights: Per-sample weights. If None, uses uniform weights.

    Returns:
        WMAE score.
    """
    if weights is None:
        weights = np.ones_like(y_true, dtype=float)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Log Error.

    Uses np.log1p (log(1+x)) which handles x=0 correctly.
    Clips negative inputs to 0 as Kaggle formula requires non-negative values.
    Matches the standard Kaggle RMSLE formula exactly.

    Args:
        y_true: True values (must be non-negative).
        y_pred: Predicted values (will be clipped to >= 0).

    Returns:
        RMSLE score.
    """
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        RMSE score.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        MAE score.
    """
    return np.mean(np.abs(y_true - y_pred))


# Metric registry for config-driven evaluation
METRIC_REGISTRY = {
    "balanced_accuracy": balanced_accuracy,
    "map@12": lambda y_true, y_pred: map_at_k(y_true, y_pred, k=12),
    "map@k": map_at_k,
    "wmae": wmae,
    "rmsle": rmsle,
    "rmse": rmse,
    "mae": mae,
}


def get_metric(name: str):
    """Get metric function by name.

    Args:
        name: Metric name (case-insensitive).

    Returns:
        Metric function.

    Raises:
        KeyError: If metric name is not recognized.
    """
    key = name.lower().replace(" ", "_")
    if key not in METRIC_REGISTRY:
        raise KeyError(
            f"Unknown metric '{name}'. Available: {list(METRIC_REGISTRY.keys())}"
        )
    return METRIC_REGISTRY[key]
