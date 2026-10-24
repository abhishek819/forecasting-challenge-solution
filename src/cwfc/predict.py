"""Generate predictions.csv for the final two-hour holdout."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .models import mean_reverting_forecast
from .targets import HOLDOUT_STEPS, build_targets, load_cpu_readings, split_train_holdout

COLUMNS = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
TARGETS = ["active_vm_count", "aggregate_cpu_utilization"]


def make_predictions(data_dir) -> pd.DataFrame:
    targets = build_targets(load_cpu_readings(data_dir))
    train, holdout = split_train_holdout(targets)
    # Only the holdout TIMESTAMPS are used; its target values are never read.
    template = pd.DataFrame(index=holdout.index)

    out = pd.DataFrame(index=holdout.index)
    for col in TARGETS:
        history = train[[col]].dropna()          # skip missing, never invent values
        out[col] = mean_reverting_forecast(history, template, col)

    out["active_vm_count"] = out["active_vm_count"].clip(lower=0)
    out["aggregate_cpu_utilization"] = out["aggregate_cpu_utilization"].clip(0, 100)
    out = out.reset_index()
    out["timestamp"] = out["timestamp"].astype("int64")
    return out[COLUMNS]


def validate_predictions(df: pd.DataFrame) -> None:
    if list(df.columns) != COLUMNS:
        raise ValueError(f"columns must be exactly {COLUMNS}")
    if len(df) != HOLDOUT_STEPS:
        raise ValueError(f"expected {HOLDOUT_STEPS} rows, got {len(df)}")
    if not df["timestamp"].is_monotonic_increasing or df["timestamp"].duplicated().any():
        raise ValueError("timestamps must be unique and chronological")
    if not np.isfinite(df[TARGETS].to_numpy(float)).all():
        raise ValueError("predictions must be finite")
    if (df["active_vm_count"] < 0).any():
        raise ValueError("active_vm_count must be >= 0")
    if not df["aggregate_cpu_utilization"].between(0, 100).all():
        raise ValueError("aggregate_cpu_utilization must be in [0, 100]")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Write predictions.csv for the 2-hour holdout.")
    p.add_argument("--data", default="challenge_data")
    p.add_argument("--output", default="predictions.csv")
    args = p.parse_args(argv)

    preds = make_predictions(args.data)
    validate_predictions(preds)
    preds.to_csv(args.output, index=False)       # no index column
    print(f"wrote {len(preds)} rows to {args.output}")


if __name__ == "__main__":
    main()