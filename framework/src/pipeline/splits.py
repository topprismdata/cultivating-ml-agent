"""Cross-validation splitters aligned with ValidationConfig.

Pure functions (no MLflow, no I/O) returning lists of (train_idx, val_idx)
numpy arrays, so they are usable by any training script.

Usage:
    from pipeline.splits import make_folds

    folds = make_folds(y, cfg.validation, time=df["date"].values,
                       groups=df["user_id"].values)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold


def _check_y(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        y = y.ravel()
    return y


def _looks_like_regression(y: np.ndarray) -> bool:
    """Heuristic: float dtype with many distinct values is regression, not
    a discrete label set that StratifiedKFold can handle."""
    if not np.issubdtype(y.dtype, np.floating):
        return False
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return True
    return np.unique(finite).size > max(50, finite.size // 10)


def stratified_folds(
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """StratifiedKFold on labels; falls back to KFold when the task is
    regression (labels not discrete)."""
    y = _check_y(y)
    if _looks_like_regression(y):
        return kfold_folds(y.size, n_folds=n_folds, random_state=random_state)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros(len(y)), y))


def kfold_folds(
    n_samples: int,
    n_folds: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Plain shuffled KFold."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return list(kf.split(np.zeros(n_samples)))


def time_based_folds(
    time: np.ndarray,
    val_size_weeks: int = 20,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Single forward split: train = rows with time < cutoff, val = rows with
    time in [cutoff, cutoff + val_size_weeks). Invariant: max(train time) <
    min(val time). Time is interpreted as float weeks (e.g. ordinal/7)."""
    time = np.asarray(time, dtype=float)
    order = np.argsort(time, kind="stable")
    t_sorted = time[order]
    max_t = t_sorted[-1]
    cutoff = max_t - val_size_weeks
    train_mask = time < cutoff
    val_mask = ~train_mask
    if not train_mask.any() or not val_mask.any():
        raise ValueError(
            f"time_based split degenerate: no rows on one side of cutoff "
            f"{cutoff:.3f} (max={max_t:.3f}, val_size_weeks={val_size_weeks})"
        )
    return [(np.where(train_mask)[0], np.where(val_mask)[0])]


def group_folds(
    groups: np.ndarray,
    n_folds: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold guaranteeing no group appears in both train and val."""
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    if n_groups < n_folds:
        raise ValueError(
            f"group strategy needs >= {n_folds} groups, got {n_groups}"
        )
    gkf = GroupKFold(n_splits=n_folds)
    return list(gkf.split(np.zeros(len(groups)), groups=groups))


def make_folds(
    y: np.ndarray,
    validation: object,
    time: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Dispatch on a ValidationConfig-like object (strategy/n_folds/
    val_size_weeks/group_col fields)."""
    strategy = getattr(validation, "strategy", "stratified")
    n_folds = int(getattr(validation, "n_folds", 5))
    if strategy == "stratified":
        return stratified_folds(y, n_folds=n_folds, random_state=random_state)
    if strategy == "kfold":
        return kfold_folds(len(y), n_folds=n_folds, random_state=random_state)
    if strategy == "time_based":
        if time is None:
            raise ValueError("time_based strategy requires time array")
        return time_based_folds(time, val_size_weeks=int(
            getattr(validation, "val_size_weeks", 20)))
    if strategy == "group":
        if groups is None:
            col = getattr(validation, "group_col", "")
            raise ValueError(
                f"group strategy requires groups array (group_col={col!r})"
            )
        return group_folds(groups, n_folds=n_folds)
    raise ValueError(f"Unknown validation strategy: {strategy!r}")
