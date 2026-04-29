"""MLflow experiment tracking utilities.

Provides one-call experiment logging, OOF prediction tracking,
feature importance logging, and LB score tracking.
Designed to be embedded into each training script with minimal boilerplate.

Usage:
    from pipeline.mlflow_utils import start_experiment, log_lb_score

    # Context manager (recommended):
    with start_experiment("R06_crossjoin", cfg) as ctx:
        ctx.log_params(params)
        ctx.log_metrics({"cv_rmse": 21.55})
        ctx.log_features(feature_cols)
        ctx.log_oof(oof_preds, train_ids)
        ctx.log_feature_importance(model, feature_cols)
        ctx.log_submission(submission_path)

    # After LB submission:
    log_lb_score(run_id="abc123", lb_score=21.3)
"""
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Context manager for experiment lifecycle
# ---------------------------------------------------------------------------

class ExperimentContext:
    """MLflow experiment context, used inside start_experiment().

    Provides typed methods for logging all experiment artifacts.
    """

    def __init__(self, run_name: str, run):
        self.run_name = run_name
        self._run = run
        self.run_id = run.info.run_id

    def log_params(self, params: dict) -> None:
        """Log parameters (auto-converts non-scalar types to strings)."""
        safe_params = {
            k: str(v) if not isinstance(v, (str, int, float)) else v
            for k, v in params.items()
        }
        mlflow.log_params(safe_params)

    def log_metrics(self, metrics: dict) -> None:
        """Log CV metrics."""
        mlflow.log_metrics(metrics)

    def log_features(self, feature_list: list[str]) -> None:
        """Log feature list as both param count and text artifact."""
        mlflow.log_param("n_features", len(feature_list))
        mlflow.log_text("\n".join(feature_list), "features.txt")

    def log_oof(self, oof_predictions: np.ndarray, ids: Optional[np.ndarray] = None,
                target_col: str = "target") -> None:
        """Log out-of-fold predictions as CSV artifact.

        OOF predictions are essential for:
        - Stacking/ensemble without data leakage
        - Error analysis on training data
        """
        df = pd.DataFrame({"oof_pred": oof_predictions})
        if ids is not None:
            df.insert(0, "id", ids)
        df[target_col] = target_col  # placeholder, caller should merge actual target

        tmp_path = Path(f"/tmp/oof_{self.run_name}.csv")
        df.to_csv(tmp_path, index=False)
        mlflow.log_artifact(str(tmp_path), artifact_path="oof")
        mlflow.log_param("oof_n_rows", len(df))
        tmp_path.unlink(missing_ok=True)

    def log_feature_importance(
        self,
        model,
        feature_names: list[str],
        top_n: int = 30,
    ) -> None:
        """Log feature importance from tree-based models.

        Supports LightGBM, XGBoost, CatBoost.
        Saves as both text file and JSON for programmatic access.
        """
        try:
            if hasattr(model, "feature_importances_"):
                # sklearn-compatible (LGBM, XGB)
                importances = model.feature_importances_
            elif hasattr(model, "get_feature_importance"):
                # CatBoost
                importances = model.get_feature_importance()
            else:
                self.log_note("Model has no feature_importances method")
                return

            sorted_idx = np.argsort(importances)[::-1]
            top_features = [
                {"feature": feature_names[i], "importance": float(importances[i])}
                for i in sorted_idx[:top_n]
            ]

            # Save as JSON
            mlflow.log_text(
                json.dumps(top_features, indent=2),
                "feature_importance.json",
            )

            # Save as readable text
            lines = [f"{f['feature']:40s} {f['importance']:10.2f}" for f in top_features]
            mlflow.log_text("\n".join(lines), "feature_importance.txt")

        except Exception as e:
            self.log_note(f"Feature importance logging failed: {e}")

    def log_submission(self, submission_path: str) -> None:
        """Log submission file as artifact."""
        mlflow.log_artifact(submission_path, artifact_path="submissions")

    def log_note(self, note: str) -> None:
        """Add a text note/tag to the run."""
        mlflow.set_tag("notes", note)

    def log_tag(self, key: str, value: str) -> None:
        """Set a tag on the run."""
        mlflow.set_tag(key, value)


@contextmanager
def start_experiment(
    run_name: str,
    cfg=None,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
):
    """Context manager for MLflow experiment lifecycle.

    Args:
        run_name: Human-readable run name (e.g., "R06_crossjoin").
        cfg: CompetitionConfig (auto-extracts tracking_uri and experiment_name).
        tracking_uri: Override MLflow tracking URI.
        experiment_name: Override experiment name.

    Yields:
        ExperimentContext with typed logging methods.

    Usage:
        with start_experiment("R06_crossjoin", cfg) as ctx:
            ctx.log_params({"model": "catboost"})
            ctx.log_metrics({"cv_rmse": 21.55})
    """
    # Resolve tracking config
    if cfg is not None:
        uri = tracking_uri or getattr(cfg.mlflow, "tracking_uri", "sqlite:///mlflow.db")
        exp_name = experiment_name or getattr(cfg.mlflow, "experiment_name", "") or getattr(cfg, "slug", "")
    else:
        uri = tracking_uri or "sqlite:///mlflow.db"
        exp_name = experiment_name or ""

    mlflow.set_tracking_uri(uri)
    if exp_name:
        mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name=run_name) as run:
        yield ExperimentContext(run_name, run)


# ---------------------------------------------------------------------------
# Standalone functions (for simple use cases)
# ---------------------------------------------------------------------------

def setup_mlflow(
    experiment_name: str,
    tracking_uri: str = "sqlite:///mlflow.db",
) -> str:
    """Initialize MLflow tracking (standalone, no context manager)."""
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    return experiment.experiment_id


def log_experiment(
    run_name: str,
    params: dict,
    metrics: dict,
    feature_list: list[str],
    submission_path: Optional[str] = None,
    notes: str = "",
    tags: Optional[dict] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> str:
    """Log a complete experiment run (standalone, no context manager).

    Returns:
        MLflow run ID for later reference.
    """
    with start_experiment(run_name, tracking_uri=tracking_uri,
                          experiment_name=experiment_name) as ctx:
        ctx.log_params(params)
        ctx.log_metrics(metrics)
        ctx.log_features(feature_list)
        if notes:
            ctx.log_note(notes)
        if tags:
            for key, value in tags.items():
                ctx.log_tag(key, value)
        if submission_path:
            ctx.log_submission(submission_path)
        return ctx.run_id


def log_lb_score(
    run_id: str,
    lb_score: float,
    lb_metric: str = "lb_score",
    tracking_uri: Optional[str] = None,
) -> None:
    """Append LB score to an existing run after submission."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric(lb_metric, lb_score)
        mlflow.set_tag("lb_submitted", "true")
