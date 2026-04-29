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
import sys
import time
from pathlib import Path

# ---- Shared modules ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent.parent / "src"))  # shared src/
sys.path.insert(0, str(PROJECT_ROOT))                          # competition local

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
    import pandas as pd

    data_dir = cfg.project_root / "data" / "raw"
    train = pd.read_csv(data_dir / cfg.data.train_file)
    test = pd.read_csv(data_dir / cfg.data.test_file)
    sample_sub = pd.read_csv(data_dir / cfg.data.sample_submission)

    log.data_shape("train", train)
    log.data_shape("test", test)
    log.data_shape("sample_sub", sample_sub)

    # Pipeline validation
    validate_pipeline(
        train, test, cfg,
        stage="after_loading",
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
    # from features.encoding import target_encode, frequency_encode
    # train, test = target_encode(train, test, col="category", target=cfg.data.target_col)
    # feature_cols.append("te_category")

    # --- Pipeline validation ---
    validate_pipeline(
        train, test, cfg,
        stage="after_feature_engineering",
        target_col=cfg.data.target_col,
    )

    # --- Feature validation ---
    validate_features(train, feature_cols, stage="train_features")
    validate_features(test, feature_cols, stage="test_features")

    log.info(f"Feature count: {len(feature_cols)}")
    return train, test, feature_cols


# ===========================================================================
# Stage 3: Model Training
# ===========================================================================
def train_model(train, test, feature_cols, cfg, log):
    """Train model with cross-validation, return (models, oof_preds, cv_metrics)."""
    log.section("Stage 3: Model Training")
    import numpy as np

    # --- Your model training here ---
    # Example with LightGBM + K-Fold:
    # import lightgbm as lgb
    # from sklearn.model_selection import KFold
    #
    # kf = KFold(n_splits=cfg.validation.n_folds, shuffle=True, random_state=cfg.experiment.random_state)
    # oof_preds = np.zeros(len(train))
    # test_preds = np.zeros(len(test))
    # models = []
    # fold_scores = []
    #
    # metric_fn = get_metric(cfg.metric)
    #
    # for fold, (trn_idx, val_idx) in enumerate(kf.split(train)):
    #     X_trn = train.iloc[trn_idx][feature_cols]
    #     y_trn = train.iloc[trn_idx][cfg.data.target_col]
    #     X_val = train.iloc[val_idx][feature_cols]
    #     y_val = train.iloc[val_idx][cfg.data.target_col]
    #
    #     model = lgb.LGBMRegressor(**cfg.model.lgb_params)
    #     model.fit(X_trn, y_trn, eval_set=[(X_val, y_val)],
    #               callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    #
    #     oof_preds[val_idx] = model.predict(X_val)
    #     test_preds += model.predict(test[feature_cols]) / cfg.validation.n_folds
    #     models.append(model)
    #
    #     fold_score = metric_fn(y_val, oof_preds[val_idx])
    #     fold_scores.append(fold_score)
    #     log.metric(f"fold_{fold}", fold_score)
    #
    # cv_metrics = {
    #     "cv_mean": np.mean(fold_scores),
    #     "cv_std": np.std(fold_scores),
    # }
    # return models, oof_preds, test_preds, cv_metrics

    raise NotImplementedError("Implement your model training")


# ===========================================================================
# Stage 4: Prediction & Submission
# ===========================================================================
def generate_submission(test_preds, sample_sub, cfg, log):
    """Generate submission with validation."""
    log.section("Stage 4: Submission")

    import pandas as pd

    # submission = pd.DataFrame({
    #     sample_sub.columns[0]: test[cfg.data.id_col].values,
    #     sample_sub.columns[1]: test_preds,
    # })
    #
    # output_path = get_submission_filename(RUN_NAME, cfg.project_root / "outputs" / "submissions")
    # validate_and_save(submission, sample_sub, output_path)
    # return output_path

    raise NotImplementedError("Implement your submission logic")


# ===========================================================================
# Main Pipeline
# ===========================================================================
def main():
    args = parse_args()
    t0 = time.time()

    # Stage 0: Config
    cfg, log = setup(args)
    log.separator(f"{RUN_NAME}: {cfg.name}")
    log.info(f"Metric: {cfg.metric} ({cfg.metric_direction})")
    log.info(f"CV: {cfg.validation.strategy}, Folds: {cfg.validation.n_folds}")
    if args.smoke:
        log.info("*** SMOKE TEST MODE ***")

    # Stage 1: Load
    train, test, sample_sub = load_data(cfg, log)

    # Stage 2: Features
    train, test, feature_cols = engineer_features(train, test, cfg, log)

    # Stage 3-4: Train + Submit (wrapped in MLflow context)
    if not args.no_mlflow:
        with start_experiment(RUN_NAME, cfg) as ctx:
            # Train
            # models, oof_preds, test_preds, cv_metrics = train_model(
            #     train, test, feature_cols, cfg, log
            # )

            # Log to MLflow
            # ctx.log_params({"model": cfg.model.default, **cfg.model.lgb_params})
            # ctx.log_metrics(cv_metrics)
            # ctx.log_features(feature_cols)
            # ctx.log_oof(oof_preds, train[cfg.data.id_col].values, cfg.data.target_col)
            # ctx.log_feature_importance(models[0], feature_cols)

            # Evaluation gate
            # evaluation_gate(
            #     cv_metrics["cv_mean"], cv_metrics["cv_std"],
            #     metric_direction=cfg.metric_direction,
            # )

            # Submit
            # output_path = generate_submission(test_preds, sample_sub, cfg, log)
            # ctx.log_submission(str(output_path))

            log.info(f"MLflow run_id: {ctx.run_id}")
    else:
        log.info("MLflow logging disabled (--no-mlflow)")
        # models, oof_preds, test_preds, cv_metrics = train_model(...)
        # output_path = generate_submission(test_preds, sample_sub, cfg, log)

    # Summary
    elapsed = time.time() - t0
    log.separator(f"{RUN_NAME} Complete ({elapsed:.0f}s)")
    # log.metric("cv_mean", cv_metrics["cv_mean"])


if __name__ == "__main__":
    main()
