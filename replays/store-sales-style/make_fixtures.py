#!/usr/bin/env python3
"""Generate the deterministic store-sales-style (multi-series) replay fixture.

Writes ``fixtures/train.csv`` next to this script: 3 stores x 180 daily
rows with pre-built features (promo flag, calendar weekday, stock level)
and weekly-seasonality labels — no sequence model required.

Byte stability by construction:
    - numpy PCG64 stream with fixed seed 42 (stream compatibility policy);
    - fixed column order, fixed store ids, calendar-derived weekday;
    - floats rendered as "%.6f", LF line endings, UTF-8, no index column.

Rerunning reproduces the committed file byte for byte (asserted in
tests/test_replay_reproducibility.py). Prints the file's sha256 on stdout.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_STORES = 3
N_DAYS = 180
START_DATE = "2026-04-01"
#: Mon..Sun multiplicative-free weekly profile added to every store.
WEEKDAY_EFFECT = np.array([2.0, -1.0, -1.5, 0.0, 1.5, 4.0, 6.0])


def build_frame() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
    weekday = dates.weekday.to_numpy()

    frames = []
    for store_idx in range(N_STORES):
        promo = (rng.random(N_DAYS) < 0.25).astype(np.int64)
        stock = rng.normal(200.0, 40.0, N_DAYS).clip(50.0, None)
        target = (
            18.0 + 4.0 * store_idx  # per-store level
            + WEEKDAY_EFFECT[weekday]  # weekly seasonality
            + 6.5 * promo  # promo lift
            + 0.02 * (stock - 200.0)
            + rng.normal(0.0, 1.2, N_DAYS)
        )
        frames.append(
            pd.DataFrame(
                {
                    "id": store_idx * N_DAYS + np.arange(N_DAYS, dtype=np.int64),
                    "date": dates.strftime("%Y-%m-%d"),
                    "store_id": np.full(N_DAYS, f"S{store_idx + 1}"),
                    "promo": promo,
                    "weekday": weekday.astype(np.int64),
                    "stock": stock,
                    "target": np.clip(target, 0.5, None),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    out_path = Path(__file__).resolve().parent / "fixtures" / "train.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_frame().to_csv(
        index=False, float_format="%.6f", lineterminator="\n"
    ).encode("utf-8")
    out_path.write_bytes(payload)
    print(hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    main()
