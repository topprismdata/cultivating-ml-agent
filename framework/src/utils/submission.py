"""Submission validation and Kaggle API submission utilities.

Handles:
    - Submission format validation against sample
    - Kaggle API submission
    - Submission file naming conventions

Usage:
    from utils.submission import validate_and_save, submit_to_kaggle

    validate_and_save(submission, sample_sub, output_path)
    submit_to_kaggle(output_path, competition_slug, message="R06 crossjoin")
"""
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd

from pipeline.validate import validate_submission


def validate_and_save(
    submission: pd.DataFrame,
    sample_submission: pd.DataFrame,
    output_path: str | Path,
    check_nan: bool = True,
    check_range: Optional[tuple[float, float]] = None,
) -> Path:
    """Validate submission format and save to file.

    Uses validate_submission from pipeline.validate for consistent checks.

    Args:
        submission: Generated submission DataFrame.
        sample_submission: Reference sample submission.
        output_path: Where to save the submission CSV.
        check_nan: Check for NaN values in prediction columns.
        check_range: Optional (min, max) range to clip predictions.

    Returns:
        Path to saved submission file.

    Raises:
        ValueError: On validation issues (row count, columns, NaN).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Delegate to shared validation
    issues = validate_submission(submission, sample_submission)

    # Additional NaN check (stricter than validate_submission which only warns)
    if check_nan:
        pred_cols = [c for c in submission.columns
                     if c != sample_submission.columns[0]]
        for col in pred_cols:
            n_nan = submission[col].isna().sum()
            if n_nan > 0:
                issues.append(f"NaN in submission column '{col}': {n_nan}")

    if issues:
        raise ValueError(f"Submission validation failed:\n" +
                         "\n".join(f"  - {i}" for i in issues))

    # Range clipping
    if check_range:
        pred_cols = [c for c in submission.columns
                     if c != sample_submission.columns[0]]
        for col in pred_cols:
            submission[col] = submission[col].clip(*check_range)

    submission.to_csv(output_path, index=False)
    print(f"[submission] Saved to {output_path} ({len(submission)} rows)")

    return output_path


def submit_to_kaggle(
    submission_path: str | Path,
    competition_slug: str,
    message: str = "",
) -> str:
    """Submit to Kaggle via the kaggle API.

    Args:
        submission_path: Path to submission CSV.
        competition_slug: Kaggle competition slug.
        message: Submission description.

    Returns:
        Kaggle submission output string.

    Raises:
        FileNotFoundError: If submission file doesn't exist.
        RuntimeError: If kaggle CLI is not available.
    """
    submission_path = Path(submission_path)
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission not found: {submission_path}")

    cmd = [
        "kaggle", "competitions", "submit",
        "-c", competition_slug,
        "-f", str(submission_path),
        "-m", message or f"Submit {submission_path.name}",
    ]

    print(f"[submission] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[submission] stderr: {result.stderr}")
        raise RuntimeError(f"Kaggle submission failed: {result.stderr}")

    print(f"[submission] Success: {result.stdout}")
    return result.stdout


def get_submission_filename(
    run_name: str,
    output_dir: str | Path = "outputs/submissions",
) -> Path:
    """Generate standardized submission filename from run name.

    Args:
        run_name: Run name like "R06_crossjoin".
        output_dir: Output directory.

    Returns:
        Full path: output_dir/submission_r06_crossjoin.csv
    """
    output_dir = Path(output_dir)
    filename = f"submission_{run_name.lower().replace(' ', '_')}.csv"
    return output_dir / filename
