"""Out-of-fold prediction frame construction.

Pure helpers (no MLflow import) so they can be unit-tested and reused
by training scripts that do not track experiments.

Usage:
    from pipeline.oof import build_oof_frame

    df = build_oof_frame(oof_preds, ids=train_ids, targets=y_true)
    df.to_csv("oof.csv", index=False)   # columns: id, target, oof_pred
"""
import numpy as np
import pandas as pd
from typing import Optional


def build_oof_frame(
    oof_preds: np.ndarray,
    ids: Optional[np.ndarray] = None,
    targets: Optional[np.ndarray] = None,
    target_col: str = "target",
) -> pd.DataFrame:
    """Build an OOF predictions DataFrame.

    Args:
        oof_preds: Out-of-fold predictions, one per training row.
        ids: Optional row ids inserted as the first column.
        targets: Optional ground-truth labels. When omitted the target
            column is not written at all — never fabricate a placeholder.
        target_col: Name of the target column (default "target").

    Returns:
        DataFrame with column order [id,] target_col, oof_pred.
    """
    data: dict = {}
    if ids is not None:
        data["id"] = np.asarray(ids)
    if targets is not None:
        data[target_col] = np.asarray(targets)
    data["oof_pred"] = np.asarray(oof_preds)
    return pd.DataFrame(data)
