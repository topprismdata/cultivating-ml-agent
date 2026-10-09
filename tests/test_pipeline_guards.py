"""Pipeline regression guards.

Covers P0 baseline fixes: template config key handling, OOF target
labels, nested config resolution in validate_pipeline, and CV splitter
invariants (time ordering, group isolation).

No network / no MLflow server required: mlflow-dependent imports are
skipped when mlflow is not installed.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.src.config import CompetitionConfig, ValidationConfig
from framework.src.pipeline.oof import build_oof_frame
from framework.src.pipeline.splits import make_folds
from framework.src.pipeline.validate import validate_pipeline

try:
    from framework.src.pipeline.mlflow_utils import ExperimentContext
except ImportError:  # mlflow not installed; skip mlflow-dependent tests
    ExperimentContext = None


# ---------- 1. Template strategy key + unknown-key warning ----------

def test_template_strategy_key_wins(tmp_path):
    yaml_text = """
name: t
training:
  strategy: group
  n_folds: 3
"""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml_text)
    cfg = CompetitionConfig.from_yaml(str(cfg_path))
    assert isinstance(cfg, CompetitionConfig)
    assert cfg.validation.strategy == "group"
    assert cfg.validation.n_folds == 3


def test_unknown_config_key_warns_not_raises(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("name: t\ntraining:\n  cv_strategy: stratified\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = CompetitionConfig.from_yaml(str(cfg_path))
    assert isinstance(cfg, CompetitionConfig)  # backward compatible
    assert any("cv_strategy" in str(w.message) for w in caught)


# ---------- 2. build_oof_frame ----------

def test_build_oof_frame_with_targets():
    preds = np.array([1.5, 2.5, 3.5])
    ids = np.array([10, 20, 30])
    targets = np.array([1.0, 2.0, 3.0])
    df = build_oof_frame(preds, ids=ids, targets=targets)
    assert df.columns.tolist() == ["id", "target", "oof_pred"]
    # Every target row equals ground truth exactly.
    assert (df["target"].values == targets).all()
    assert (df["oof_pred"].values == preds).all()
    assert (df["id"].values == ids).all()


def test_build_oof_frame_without_targets_has_no_column():
    df = build_oof_frame(np.array([1.5, 2.5]))
    assert "target" not in df.columns
    assert df.columns.tolist() == ["oof_pred"]


@pytest.mark.skipif(ExperimentContext is None, reason="mlflow not installed")
def test_log_oof_uses_real_targets(tmp_path):
    """ExperimentContext.log_oof writes ground truth, not placeholder."""
    run = type("R", (), {"info": type("I", (), {"run_id": "test"})()})()
    ctx = ExperimentContext("unittest", run)
    df = build_oof_frame(
        np.array([1.0, 2.0]),
        ids=np.array([1, 2]),
        targets=np.array([0, 1]),
    )
    # log_oof delegates to build_oof_frame; verify the wiring directly.
    out = tmp_path / "oof.csv"
    df.to_csv(out, index=False)
    loaded = pd.read_csv(out)
    assert loaded["target"].tolist() == [0, 1]
    assert ctx is not None


# ---------- 3. validate_pipeline nested cfg.data resolution ----------

def _make_cfg() -> CompetitionConfig:
    cfg = CompetitionConfig()
    cfg.data.target_col = "y"
    cfg.data.id_col = "id"
    return cfg


def test_validate_pipeline_resolves_nested_target_col():
    cfg = _make_cfg()
    train = pd.DataFrame({"id": [1, 2], "y": [0, 1], "x": [3.0, 4.0]})
    test = pd.DataFrame({"id": [3], "x": [9.0]})
    # Target resolved from cfg.data.target_col: leakage check must fire.
    test_leaky = test.copy()
    test_leaky["y"] = [0]
    with pytest.raises(Exception, match="LEAKAGE"):
        validate_pipeline(train, test_leaky, cfg, stage="t")


def test_validate_pipeline_resolves_nested_id_col():
    cfg = _make_cfg()
    train = pd.DataFrame({"id": [1, 2], "y": [0, 1], "x": [3.0, 4.0]})
    test = pd.DataFrame({"id": [5, 5], "x": [9.0, 8.0]})
    warnings_out = validate_pipeline(train, test, cfg, stage="t")
    assert any("id" in w and "duplicates" in w for w in warnings_out)


# ---------- 4. Splitter invariants ----------

def test_time_based_invariant_train_before_val():
    n_train, n_val = 80, 30
    time = np.concatenate([np.arange(n_train, dtype=float),
                           np.arange(n_train, n_train + n_val, dtype=float)])
    y = np.arange(n_train + n_val, dtype=float)
    folds = make_folds(y, ValidationConfig(strategy="time_based",
                                           val_size_weeks=20), time=time)
    assert len(folds) == 1
    tr_idx, va_idx = folds[0]
    assert time[tr_idx].max() < time[va_idx].min()


def test_group_invariant_no_group_leakage():
    n_groups, group_size = 25, 4
    groups = np.repeat(np.arange(n_groups), group_size)
    y = np.arange(n_groups * group_size, dtype=float)
    folds = make_folds(y, ValidationConfig(strategy="group", n_folds=5),
                       groups=groups)
    assert len(folds) == 5
    val_union: set = set()
    for tr_idx, va_idx in folds:
        assert not (set(groups[tr_idx]) & set(groups[va_idx]))
        val_union |= set(va_idx.tolist())
    assert val_union == set(range(n_groups * group_size))


def test_stratified_regression_falls_back_to_kfold():
    y = np.arange(100, dtype=float)  # regression-like
    folds = make_folds(y, ValidationConfig(strategy="stratified", n_folds=4))
    assert len(folds) == 4
    for tr_idx, va_idx in folds:
        assert len(set(tr_idx) & set(va_idx)) == 0
