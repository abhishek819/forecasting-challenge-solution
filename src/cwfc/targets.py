"""Build the two required forecasting targets from VM-level CPU readings."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STEP_SECONDS = 300      # five-minute grid
HOLDOUT_STEPS = 24      # final two hours


def load_cpu_readings(data_dir) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "vm_cpu_readings.csv.gz")


def build_targets(cpu: pd.DataFrame, step_seconds: int = STEP_SECONDS) -> pd.DataFrame:
    """Return one row per five-minute timestamp with the two required targets.

    Rules (documented in the report):
    - a valid observation has a finite avg_cpu inside [0, 100];
    - each VM is counted at most once per timestamp (first row kept);
    - missing readings are skipped, never filled;
    - a timestamp missing from the grid stays NaN, never invented.
    """
    df = cpu[["timestamp", "vm_id", "avg_cpu"]].copy()

    valid = np.isfinite(df["avg_cpu"]) & df["avg_cpu"].between(0, 100)
    df = df[valid]
    df = df.drop_duplicates(subset=["timestamp", "vm_id"], keep="first")

    grouped = df.groupby("timestamp")
    targets = pd.DataFrame(
        {
            "active_vm_count": grouped["vm_id"].nunique(),
            "aggregate_cpu_utilization": grouped["avg_cpu"].mean(),
        }
    ).sort_index()

    start = targets.index.min()
    if ((targets.index - start) % step_seconds != 0).any():
        raise ValueError("Some timestamps are not on the five-minute grid.")

    full_grid = np.arange(start, targets.index.max() + 1, step_seconds)
    targets = targets.reindex(full_grid)      # gaps become NaN, not invented
    targets.index.name = "timestamp"
    return targets


def split_train_holdout(targets: pd.DataFrame, holdout_steps: int = HOLDOUT_STEPS):
    """Chronological split: last `holdout_steps` rows are the holdout."""
    if len(targets) <= holdout_steps:
        raise ValueError("Series is shorter than the holdout horizon.")
    return targets.iloc[:-holdout_steps], targets.iloc[-holdout_steps:]