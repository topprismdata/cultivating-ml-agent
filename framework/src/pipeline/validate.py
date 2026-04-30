"""Pipeline data validation.

Runs sanity checks at each pipeline stage to catch data issues early.
Inspired by the ml-pipeline-unit-testing skill.

Usage:
    from pipeline.validate import validate_pipeline
    validate_pipeline(train, test, cfg, stage="after_feature_engineering")
"""
import pandas as pd
import numpy as np
from typing import Optional, Any


class PipelineValidationError(Exception):
    """Raised when pipeline validation detects a critical data issue."""


def validate_pipeline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: Any = None,
    stage: str = "unknown",
    target_col: Optional[str] = None,
    allow_train_test_mismatch: bool = False,
) -> list[str]:
    """Validate data integrity at each pipeline stage.

    Args:
        train: Training DataFrame.
        test: Test DataFrame.
        cfg: CompetitionConfig (optional, used for target_col/id_col fallback).
        stage: Pipeline stage name for logging.
        target_col: Name of target column. If None, tries cfg.target_col.
        allow_train_test_mismatch: Set True for recommendation tasks where
            train/test column sets may differ naturally.

    Returns:
        List of warning messages (non-critical issues).
        Raises PipelineValidationError on critical errors.
    """
    if cfg is not None and target_col is None:
        target_col = getattr(cfg, "target_col", None)

    errors: list[str] = []
    warnings: list[str] = []

    # 1. Target column checks
    if target_col:
        if target_col not in train.columns:
            errors.append(f"Target '{target_col}' missing from train columns")
        if target_col in test.columns:
            errors.append(f"TARGET LEAKAGE: '{target_col}' found in test columns!")

    # 2. Column alignment
    train_cols = set(train.columns) - {target_col} if target_col else set(train.columns)
    test_cols = set(test.columns)
    missing_in_test = train_cols - test_cols

    if missing_in_test and not allow_train_test_mismatch:
        warnings.append(f"Columns in train missing from test ({len(missing_in_test)}): "
                        f"{sorted(missing_in_test)[:10]}")

    # 3. Duplicate columns
    for name, df in [("train", train), ("test", test)]:
        if len(df.columns) != len(set(df.columns)):
            dupes = df.columns[df.columns.duplicated()].tolist()
            errors.append(f"DUPLICATE columns in {name}: {dupes}")

    # 4. NaN summary
    for name, df in [("train", train), ("test", test)]:
        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0]
        if len(nulls) > 0:
            warnings.append(f"NaN in {name}: {nulls.to_dict()}")

    # 5. Shape info
    print(f"[validate:{stage}] train={train.shape}, test={test.shape}")

    # 6. ID column uniqueness (if configured)
    if cfg is not None:
        id_col = getattr(cfg, "id_col", None)
        if id_col and id_col in test.columns:
            n_unique = test[id_col].nunique()
            if n_unique != len(test):
                warnings.append(
                    f"Test {id_col} has {n_unique} unique values "
                    f"but {len(test)} rows (duplicates exist)"
                )

    # Report and raise on critical errors
    if errors:
        print(f"[validate:{stage}] ERRORS:\n" +
              "\n".join(f"  - {e}" for e in errors))
        raise PipelineValidationError(
            f"[{stage}] Critical validation failures:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )

    if warnings:
        print(f"[validate:{stage}] Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    else:
        print(f"[validate:{stage}] All checks passed")

    return warnings


def validate_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    stage: str = "features",
) -> list[str]:
    """Validate that feature columns exist and have no infinite values.

    Args:
        df: DataFrame to validate.
        feature_cols: Expected feature column names.
        stage: Stage name for logging.

    Returns:
        List of issues found.
    """
    issues = []

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        issues.append(f"Missing feature columns: {missing}")

    existing = [c for c in feature_cols if c in df.columns]
    for col in existing:
        if pd.api.types.is_numeric_dtype(df[col]):
            n_inf = np.isinf(df[col]).sum()
            if n_inf > 0:
                issues.append(f"Column '{col}' has {n_inf} infinite values")

    if issues:
        print(f"[validate:{stage}] Feature issues: {issues}")
    else:
        print(f"[validate:{stage}] {len(feature_cols)} features validated OK")

    return issues


def validate_submission(
    submission: pd.DataFrame,
    sample_submission: pd.DataFrame,
    stage: str = "submission",
) -> list[str]:
    """Validate submission format against sample.

    Args:
        submission: Generated submission DataFrame.
        sample_submission: Reference sample submission.
        stage: Stage name for logging.

    Returns:
        List of issues found.
    """
    issues = []

    if len(submission) != len(sample_submission):
        issues.append(
            f"Row count mismatch: {len(submission)} vs {len(sample_submission)}"
        )

    if list(submission.columns) != list(sample_submission.columns):
        issues.append(
            f"Column mismatch: {submission.columns.tolist()} vs "
            f"{sample_submission.columns.tolist()}"
        )

    # Check for NaN in submission values
    value_cols = [c for c in submission.columns if c != sample_submission.columns[0]]
    for col in value_cols:
        n_nan = submission[col].isna().sum()
        if n_nan > 0:
            issues.append(f"NaN in submission column '{col}': {n_nan}")

    if issues:
        print(f"[validate:{stage}] Submission issues: {issues}")
    else:
        print(f"[validate:{stage}] Submission validated: {len(submission)} rows")

    return issues


# ---------------------------------------------------------------------------
# Evaluation Gate (NotebookLM-validated)
# ---------------------------------------------------------------------------

class EvaluationGateError(Exception):
    """Raised when CV score fails evaluation gate threshold."""


def evaluation_gate(
    cv_score: float,
    cv_std: Optional[float] = None,
    baseline_score: Optional[float] = None,
    metric_direction: str = "maximize",
    min_improvement: float = 0.0,
    max_cv_std_ratio: float = 0.1,
) -> bool:
    """Evaluation gate: block submission if CV doesn't meet threshold.

    Implements the AgentOps "Evaluation Gates" pattern: only allow
    submission when the experiment passes a quality threshold.

    Args:
        cv_score: Mean CV score from the experiment.
        cv_std: Standard deviation of CV scores. None or 0.0 for single-fold.
        baseline_score: Previous best score to beat (optional).
        metric_direction: "maximize" or "minimize".
        min_improvement: Minimum improvement over baseline (absolute).
        max_cv_std_ratio: Maximum CV std / |cv_score| ratio (stability check).

    Returns:
        True if gate passes.

    Raises:
        EvaluationGateError if gate fails.
    """
    reasons = []

    # Stability check: CV std should be reasonable
    if cv_std is None:
        cv_std = 0.0
    if abs(cv_score) > 0 and cv_std > 0:
        std_ratio = cv_std / abs(cv_score)
        if std_ratio > max_cv_std_ratio:
            reasons.append(
                f"CV unstable: std/score ratio = {std_ratio:.4f} > {max_cv_std_ratio}"
            )

    # Improvement check: must beat baseline
    if baseline_score is not None:
        if metric_direction == "maximize":
            improved = cv_score > baseline_score + min_improvement
            if not improved:
                reasons.append(
                    f"No improvement: {cv_score:.4f} <= baseline {baseline_score:.4f} "
                    f"+ {min_improvement}"
                )
        else:
            improved = cv_score < baseline_score - min_improvement
            if not improved:
                reasons.append(
                    f"No improvement: {cv_score:.4f} >= baseline {baseline_score:.4f} "
                    f"- {min_improvement}"
                )

    if reasons:
        msg = "Evaluation gate FAILED:\n" + "\n".join(f"  - {r}" for r in reasons)
        print(f"[evaluation_gate] {msg}")
        raise EvaluationGateError(msg)

    print(f"[evaluation_gate] PASSED: cv={cv_score:.4f} +/- {cv_std:.4f}")
    return True


# ---------------------------------------------------------------------------
# Failure Classification (NotebookLM-validated)
# ---------------------------------------------------------------------------

FAILURE_CATEGORIES = {
    "data_leakage": "Target or future information leaked into features",
    "distribution_shift": "Train/test distribution mismatch detected",
    "feature_error": "Feature engineering produced invalid values (NaN, Inf)",
    "model_overfit": "CV score much better than LB score",
    "model_underfit": "CV score poor, model capacity insufficient",
    "submission_format": "Submission format mismatch (rows, columns, types)",
    "metric_mismatch": "Evaluation metric doesn't match competition metric",
    "random_seed": "Non-reproducible results due to unseeded randomness",
    "tool_failure": "External tool or library error",
    "unknown": "Unclassified failure",
}


def classify_failure(
    error_message: str,
    cv_score: Optional[float] = None,
    lb_score: Optional[float] = None,
) -> dict:
    """Classify a pipeline failure into structured categories.

    Implements the AgentOps "Failure Classification" pattern for
    structured error attribution and targeted fixes.

    Args:
        error_message: The error or issue description.
        cv_score: CV score (if available).
        lb_score: LB score (if available).

    Returns:
        Dict with category, severity, and suggested fix.
    """
    msg = error_message.lower()
    category = "unknown"

    # Pattern matching for failure classification
    if any(k in msg for k in ["leakage", "target in test", "future data"]):
        category = "data_leakage"
    elif any(k in msg for k in ["distribution", "drift", "adversarial"]):
        category = "distribution_shift"
    elif any(k in msg for k in ["nan", "inf", "infinite", "missing"]):
        category = "feature_error"
    elif any(k in msg for k in ["duplicate", "column mismatch", "row count"]):
        category = "submission_format"
    elif any(k in msg for k in ["metric", "score type"]):
        category = "metric_mismatch"
    elif any(k in msg for k in ["seed", "random", "reproducib"]):
        category = "random_seed"

    # CV vs LB divergence detection
    if cv_score is not None and lb_score is not None:
        gap = abs(cv_score - lb_score)
        if gap > 0.05 * abs(cv_score) if cv_score != 0 else gap > 0.01:
            if category == "unknown":
                category = "model_overfit"

    # Severity assessment
    severity = "critical" if category in [
        "data_leakage", "distribution_shift"
    ] else "warning"

    result = {
        "category": category,
        "description": FAILURE_CATEGORIES.get(category, "Unknown"),
        "severity": severity,
        "original_error": error_message,
    }

    if cv_score is not None:
        result["cv_score"] = cv_score
    if lb_score is not None:
        result["lb_score"] = lb_score

    print(f"[classify_failure] {category} ({severity}): {error_message[:80]}")
    return result
