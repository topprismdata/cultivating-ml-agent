#!/usr/bin/env python3
"""Generate the deterministic S6E5-style (borehole rig) replay fixture.

Writes ``fixtures/train.csv`` next to this script: ~2000 daily rows, 7
features (5 numeric + 2 categorical), a label with a mild temporal trend
and ~1% missing labels (the missing-label regime of Playground S6E5 rig
usage).

Byte stability by construction:
    - numpy PCG64 stream with fixed seed 42 (stream compatibility policy);
    - fixed column order and fixed category values;
    - floats rendered as "%.6f", LF line endings, UTF-8, no index column;
    - missing labels serialized as empty fields (pandas default na_rep).

Rerunning reproduces the committed file byte for byte (asserted in
tests/test_replay_reproducibility.py). Prints the file's sha256 on stdout.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 2000
START_DATE = "2021-01-01"


def build_frame() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    t = np.arange(N_ROWS, dtype=np.float64)
    dates = pd.date_range(START_DATE, periods=N_ROWS, freq="D")

    depth = rng.normal(1200.0, 250.0, N_ROWS).clip(300.0, None)
    weight = rng.normal(80.0, 12.0, N_ROWS)
    hours = rng.integers(4, 18, N_ROWS).astype(np.float64)
    temp = rng.normal(35.0, 6.0, N_ROWS)
    vibration = rng.normal(2.5, 0.8, N_ROWS)
    model = rng.choice(["M1", "M2", "M3"], size=N_ROWS, p=[0.5, 0.3, 0.2])
    region = rng.choice(
        ["east", "west", "north", "south"], size=N_ROWS, p=[0.4, 0.25, 0.2, 0.15]
    )

    signal = (
        40.0
        + 0.004 * t  # mild temporal drift
        + 2.0 * np.sin(2.0 * np.pi * t / 365.25)  # annual seasonality
        + 0.010 * (depth - 1200.0)
        + 0.30 * (weight - 80.0)
        + 0.60 * (hours - 10.0)
        + 0.25 * (temp - 35.0)
        + 1.50 * (vibration - 2.5)
        + np.select([model == "M1", model == "M2", model == "M3"], [0.0, 3.0, 6.5])
        + np.select(
            [region == "east", region == "west", region == "north", region == "south"],
            [0.0, 1.2, -0.8, 2.0],
        )
        + rng.normal(0.0, 1.5, N_ROWS)
    )
    target = np.clip(signal, 0.5, None)
    missing = rng.random(N_ROWS) < 0.01
    target[missing] = np.nan

    return pd.DataFrame(
        {
            "id": np.arange(N_ROWS, dtype=np.int64),
            "date": dates.strftime("%Y-%m-%d"),
            "depth": depth,
            "weight": weight,
            "hours": hours,
            "temp": temp,
            "vibration": vibration,
            "model": model,
            "region": region,
            "target": target,
        }
    )


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
