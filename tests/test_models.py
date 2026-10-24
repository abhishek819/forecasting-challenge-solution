import numpy as np
import pandas as pd
from cwfc.models import mean_reverting_forecast, ridge_direct_forecast


def _df(vals):
    return pd.DataFrame({"y": vals})


def test_constant_series_gives_constant_forecast():
    df = _df(np.full(60, 10000.0)); tmpl = _df(np.zeros(24))
    for f in (mean_reverting_forecast, ridge_direct_forecast):
        assert np.allclose(f(df, tmpl, "y"), 10000.0)


def test_forecast_has_right_length_and_is_finite():
    rng = np.random.default_rng(0)
    df = _df(9.4 + rng.normal(0, 0.2, 60)); tmpl = _df(np.zeros(24))
    for f in (mean_reverting_forecast, ridge_direct_forecast):
        out = f(df, tmpl, "y")
        assert len(out) == 24 and np.isfinite(out).all()