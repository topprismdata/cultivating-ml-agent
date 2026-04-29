"""Feature encoding utilities.

Common encoding strategies for tabular ML competitions.
Extracted from Playground Series S6E4 and H&M experiments.

Supported encodings:
    - Target Encoding (with smoothing)
    - Frequency Encoding
    - Label Encoding (with NaN handling)
    - WOEEncoding (for binary targets)

Usage:
    from features.encoding import target_encode, frequency_encode

    train_enc, test_enc = target_encode(
        train, test, col="category", target="label", smoothing=10
    )
"""
import numpy as np
import pandas as pd
from typing import Optional


def target_encode(
    train: pd.DataFrame,
    test: pd.DataFrame,
    col: str,
    target: str,
    smoothing: float = 10.0,
    prefix: str = "te",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Target encoding with Bayesian smoothing.

    Computes mean target per category, smoothed toward the global mean
    based on category frequency. Uses train-only statistics to prevent
    target leakage.

    Args:
        train: Training DataFrame.
        test: Test DataFrame.
        col: Column to encode.
        target: Target column name.
        smoothing: Smoothing factor (higher = more regularization).
        prefix: Column name prefix.

    Returns:
        (train, test) with new encoded column added.
    """
    train = train.copy()
    test = test.copy()

    global_mean = train[target].mean()
    stats = train.groupby(col)[target].agg(["mean", "count"])

    # Bayesian smoothing: weighted average between category mean and global mean
    stats["smoothed"] = (
        (stats["count"] * stats["mean"] + smoothing * global_mean)
        / (stats["count"] + smoothing)
    )

    encoded_col = f"{prefix}_{col}"
    mapping = stats["smoothed"].to_dict()

    train[encoded_col] = train[col].map(mapping).fillna(global_mean)
    test[encoded_col] = test[col].map(mapping).fillna(global_mean)

    return train, test


def frequency_encode(
    train: pd.DataFrame,
    test: pd.DataFrame,
    col: str,
    prefix: str = "freq",
    normalize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frequency encoding (count / total count).

    Encodes categories by their occurrence frequency in training data.

    Args:
        train: Training DataFrame.
        test: Test DataFrame.
        col: Column to encode.
        prefix: Column name prefix.
        normalize: If True, normalize to [0, 1]. If False, use raw counts.

    Returns:
        (train, test) with new encoded column added.
    """
    train = train.copy()
    test = test.copy()

    counts = train[col].value_counts(normalize=normalize)
    encoded_col = f"{prefix}_{col}"

    train[encoded_col] = train[col].map(counts).fillna(0)
    test[encoded_col] = test[col].map(counts).fillna(0)

    return train, test


def label_encode_with_nan(
    train: pd.DataFrame,
    test: pd.DataFrame,
    col: str,
    prefix: str = "le",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Label encoding that preserves NaN as a special category.

    Args:
        train: Training DataFrame.
        test: Test DataFrame.
        col: Column to encode.
        prefix: Column name prefix.

    Returns:
        (train, test, mapping_dict) with new encoded column added.
    """
    train = train.copy()
    test = test.copy()

    # Get unique values from train only
    unique_vals = train[col].dropna().unique()
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    mapping[np.nan] = -1  # Special NaN category

    encoded_col = f"{prefix}_{col}"

    train[encoded_col] = train[col].map(mapping).fillna(-1).astype(int)
    test[encoded_col] = test[col].map(mapping).fillna(-1).astype(int)

    return train, test, mapping


def woe_encode(
    train: pd.DataFrame,
    test: pd.DataFrame,
    col: str,
    target: str,
    prefix: str = "woe",
    min_samples: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weight of Evidence encoding for binary targets.

    WOE = ln(% of events / % of non-events)

    Args:
        train: Training DataFrame.
        test: Test DataFrame.
        col: Column to encode.
        target: Binary target column (0/1).
        prefix: Column name prefix.
        min_samples: Minimum samples per category to compute WOE.

    Returns:
        (train, test) with new encoded column added.
    """
    train = train.copy()
    test = test.copy()

    total_events = train[target].sum()
    total_non_events = len(train) - total_events

    # Compute WOE per category
    stats = train.groupby(col)[target].agg(["sum", "count"])
    stats.columns = ["events", "total"]
    stats["non_events"] = stats["total"] - stats["events"]

    # Avoid division by zero
    stats["pct_events"] = (stats["events"] + 0.5) / (total_events + 0.5)
    stats["pct_non_events"] = (stats["non_events"] + 0.5) / (total_non_events + 0.5)
    stats["woe"] = np.log(stats["pct_events"] / stats["pct_non_events"])

    # Regularize small categories
    stats.loc[stats["total"] < min_samples, "woe"] = 0.0

    encoded_col = f"{prefix}_{col}"
    mapping = stats["woe"].to_dict()

    train[encoded_col] = train[col].map(mapping).fillna(0.0)
    test[encoded_col] = test[col].map(mapping).fillna(0.0)

    return train, test
