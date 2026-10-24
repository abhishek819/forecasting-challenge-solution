"""Candidate forecasting models. All use only the training slice they are given."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def mean_reverting_forecast(train, template, target, max_phi=0.99):
    """Forecast h steps ahead as mu + phi**h * (last - mu); mu, phi from `train` only."""
    y = train[target].astype(float).to_numpy()
    if len(y) < 3:
        raise ValueError("not enough history")
    mu = y.mean()
    x, z = y[:-1] - mu, y[1:] - mu
    denom = float(np.dot(x, x))
    phi = float(np.dot(x, z) / denom) if denom > 0 else 0.0
    phi = min(max(phi, 0.0), max_phi)
    h = np.arange(1, len(template) + 1)
    return pd.Series(mu + phi ** h * (y[-1] - mu), index=template.index, dtype=float)

def _features(y, i, base):
    """Features known at origin i only (deviations from the mean known at i)."""
    return [y[i] - base,
            y[max(0, i - 2): i + 1].mean() - base,
            y[max(0, i - 11): i + 1].mean() - base]


def ridge_direct_forecast(train, template, target, alpha=0.1, min_origin=11):
    y = train[target].astype(float).to_numpy()
    n, H = len(y), len(template)
    cum = np.cumsum(y) / np.arange(1, n + 1)        # expanding mean, causal
    x_now = np.array([_features(y, n - 1, cum[n - 1])])
    out = np.empty(H)
    for h in range(1, H + 1):
        X = [_features(y, i, cum[i]) for i in range(min_origin, n - h)]
        t = [y[i + h] - cum[i] for i in range(min_origin, n - h)]
        if len(X) < 5:                               # too little data: use the mean
            out[h - 1] = cum[n - 1]
            continue
        model = Ridge(alpha=alpha).fit(np.array(X), np.array(t))
        out[h - 1] = cum[n - 1] + model.predict(x_now)[0]
    return pd.Series(out, index=template.index, dtype=float)