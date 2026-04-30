#!/usr/bin/env python3
"""R{NN}: {Brief description}

Changes from R{NN-1}:
  1. {Change 1}
  2. {Change 2}

Validation: {validation_strategy}
Expected improvement: {rationale}

Usage:
    python scripts/run_r{NN}_{name}.py
    python scripts/run_r{NN}_{name}.py --smoke        # 1-fold smoke test
    python scripts/run_r{NN}_{name}.py --no-mlflow     # skip MLflow logging
    python scripts/run_r{NN}_{name}.py --override validation.n_folds=3
"""
import argparse
import gc
import sys
import time
from pathlib import Path

# ---- Shared modules ----
# IMPORTANT: sys.path ordering matters when local and shared src have same-named
# files (e.g., config.py). Always load local config FIRST.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Option A: Framework src/ is inside this project (monorepo layout)
SHARED_SRC = PROJECT_ROOT / "src"

# Option B: Framework is in a sibling project (multi-project layout)
# Uncomment and set the absolute path:
# SHARED_SRC = Path("/absolute/path/to/shared/framework/src")

sys.path.insert(0, str(SHARED_SRC))

from config import CompetitionConfig, load_config
from pipeline.validate import validate_pipeline, validate_features, evaluation_gate
from pipeline.mlflow_utils import start_experiment, log_lb_score
from utils.metrics import get_metric
from utils.submission import validate_and_save, get_submission_filename
from utils.logging_utils import get_logger

# ---- Per-experiment overrides ----
OVERRIDES = {
    # "model.lgb_params.learning_rate": 0.01,
    # "model.lgb_params.num_leaves": 127,
    # "experiment.n_seeds": 1,
}

RUN_NAME = "R{NN}_{name}"
RANDOM_STATE = 42
start_time = time.time()


def parse_args():
    parser = argparse.ArgumentParser(description=f"{RUN_NAME}")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 1 fold, 100 estimators")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Skip MLflow logging")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Override config: key=value pairs")
    return parser.parse_args()


# ===========================================================================
# Stage 0: Configuration
# ===========================================================================
def setup(args):
    """Load config, apply overrides, setup logging."""
    overrides = {**OVERRIDES}
    if args.smoke:
        overrides["validation.n_folds"] = 1
        overrides["experiment.smoke_test"] = True

    cfg = load_config(
        config_path=PROJECT_ROOT / "config.yaml",
        overrides=overrides,
        cli_args=args.override,
    )
    log = get_logger(RUN_NAME)
    return cfg, log


# ===========================================================================
# Stage 1: Data Loading
# ===========================================================================
def load_data(cfg, log):
    """Load and return train, test, sample_submission."""
    log.section("Stage 1: Data Loading")
    import numpy as np
    import pandas as pd

    np.random.seed(RANDOM_STATE)

    # Use cfg.get_data_path() or override for non-standard layouts:
    # data_dir = cfg.project_root / "data_raw"  # If data is in data_raw/ not data/raw/
    data_dir = cfg.project_root / "data" / "raw"

    train = pd.read_csv(data_dir / cfg.data.train_file)
    test = pd.read_csv(data_dir / cfg.data.test_file)
    sample_sub = pd.read_csv(data_dir / cfg.data.sample_submission)

    log.data_shape("train", train)
    log.data_shape("test", test)
    log.data_shape("sample_sub", sample_sub)

    validate_pipeline(
        train, test, cfg,
        stage="after_data_loading",
        target_col=cfg.data.target_col,
    )

    return train, test, sample_sub


# ===========================================================================
# Stage 2: Feature Engineering
# ===========================================================================
def engineer_features(train, test, cfg, log):
    """Build features and return (train, test, feature_cols)."""
    log.section("Stage 2: Feature Engineering")
    feature_cols = []

    # --- Your features here ---
    # Example:
    # from features.encoding import target_encode
    # train, test = target_encode(train, test, col="category", target=cfg.data.target_col)
    # feature_cols.append("te_category")

    # --- Pipeline validation ---
    validate_pipeline(
        train, test, cfg,
        stage="after_feature_engineering",
        target_col=cfg.data.target_col,
    )
    validate_features(train, feature_cols, stage="train_features")

    log.info(f"  Feature count: {len(feature_cols)}")
    return train, test, feature_cols


# ===========================================================================
# Stage 3: Model Training
# ===========================================================================
def train_model(train, test, feature_cols, cfg, log):
    """Train model with CV and return (model, oof_preds, test_preds, cv_metrics)."""
    log.section("Stage 3: Model Training")

    # --- Your model training here ---
    # Example (LightGBM with 5-fold CV):
    #
    # import lightgbm as lgb
    # from sklearn.model_selection import cross_val_score
    #
    # model = lgb.LGBMRegressor(**cfg.model.lgb_params)
    # ...

    raise NotImplementedError("Implement your model training")


# ===========================================================================
# Stage 4: Prediction & Submission
# ===========================================================================
def generate_submission(test_preds, test, sample_sub, cfg, log):
    """Generate predictions and save submission."""
    log.section("Stage 4: Prediction & Submission")

    # --- Build submission ---
    # submission = pd.DataFrame({
    #     sample_sub.columns[0]: test[cfg.data.id_col].values,
    #     sample_sub.columns[1]: test_preds,
    # })

    # --- Validate & Save ---
    # output_path = get_submission_filename(RUN_NAME, PROJECT_ROOT / "outputs" / "submissions")
    # validate_and_save(submission, sample_sub, output_path)
    # return output_path

    raise NotImplementedError("Implement your submission logic")


# ===========================================================================
# Main
# ===========================================================================
def main():
    args = parse_args()
    cfg, log = setup(args)

    log.separator(f"{RUN_NAME}: {cfg.name}")
    log.info(f"  Metric: {cfg.metric} ({cfg.metric_direction})")
    log.info(f"  CV: {cfg.validation.strategy}, folds: {cfg.validation.n_folds}")

    # Stage 1: Load
    train, test, sample_sub = load_data(cfg, log)

    # Stage 2: Features
    train, test, feature_cols = engineer_features(train, test, cfg, log)

    # Stage 3: Train
    # model, oof, test_preds, cv_metrics = train_model(train, test, feature_cols, cfg, log)

    # Stage 4: Submit
    # output_path = generate_submission(test_preds, test, sample_sub, cfg, log)

    # Stage 5: Evaluation gate
    # evaluation_gate(
    #     cv_score=cv_metrics["cv_mean"],
    #     cv_std=cv_metrics.get("cv_std", 0.0),
    #     baseline_score=0.0,  # Set from previous best
    #     metric_direction=cfg.metric_direction,
    # )

    # MLflow tracking
    if not args.no_mlflow:
        with start_experiment(RUN_NAME, cfg) as ctx:
            ctx.log_params({"model": cfg.model.default})
            # ctx.log_metrics(cv_metrics)
            # ctx.log_features(feature_cols)
            # ctx.log_submission(str(output_path))
            log.info(f"  MLflow run_id: {ctx.run_id}")

    elapsed = time.time() - start_time
    log.separator(f"{RUN_NAME} Complete")
    log.info(f"  Elapsed: {elapsed:.0f}s")
    # log.metric("CV", cv_metrics["cv_mean"])
    gc.collect()


if __name__ == "__main__":
    main()
