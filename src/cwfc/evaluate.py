"""Rolling-origin evaluation shared by baselines and (later) the main model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .validation import rolling_origin_splits


def make_frame(train: pd.DataFrame) -> pd.DataFrame:
    """Starter splitter needs a datetime 'timestamp' column (source ts are seconds)."""
    frame = train.reset_index()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    return frame


def evaluate_forecaster(train, forecaster, target, horizon=24, min_train_size=50, step=12):
    """Score `forecaster(train_part, template, target) -> Series` on every fold.

    `train` must be the 110-row training window ONLY (never the holdout).
    Returns (per-fold table, array of errors shaped folds x horizon).
    """
    frame = make_frame(train)
    splits = rolling_origin_splits(frame, horizon=horizon,
                                   min_train_size=min_train_size, step=step)
    if not splits:
        raise ValueError("No validation folds; check horizon/min_train_size/step.")

    rows, errors = [], []
    for k, s in enumerate(splits, 1):
        tr = frame[frame["timestamp"] <= s.train_end]
        va = frame[(frame["timestamp"] >= s.validation_start) &
                   (frame["timestamp"] <= s.validation_end)]
        assert tr["timestamp"].max() < va["timestamp"].min()      # no leakage
        pred = forecaster(tr, va, target)
        err = va[target].to_numpy(float) - pred.to_numpy(float)
        errors.append(err)
        rows.append({"fold": k, "n_train": len(tr),
                     "mae": float(np.mean(np.abs(err))),
                     "rmse": float(np.sqrt(np.mean(err ** 2)))})
    return pd.DataFrame(rows), np.vstack(errors)


def summarize(errors: np.ndarray) -> dict:
    """Pooled MAE/RMSE over all folds and all horizon steps."""
    return {"mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors ** 2)))}

def collect_predictions(train, forecaster, target, horizon=24, min_train_size=50, step=12):
    """Long table: one row per (fold, step ahead) with actual, prediction, error."""
    frame = make_frame(train)
    splits = rolling_origin_splits(frame, horizon=horizon,
                                   min_train_size=min_train_size, step=step)
    rows = []
    for k, s in enumerate(splits, 1):
        tr = frame[frame["timestamp"] <= s.train_end]
        va = frame[(frame["timestamp"] >= s.validation_start) &
                   (frame["timestamp"] <= s.validation_end)]
        pred = forecaster(tr, va, target).to_numpy(float)
        for h, (ts, y, p) in enumerate(zip(va["timestamp"], va[target], pred)):
            rows.append({"fold": k, "step_ahead": h, "timestamp": ts,
                         "actual": float(y), "pred": float(p)})
    out = pd.DataFrame(rows)
    out["error"] = out["actual"] - out["pred"]
    return out