import numpy as np
import pandas as pd
from cwfc.baselines import moving_average_forecast, training_mean_forecast
from cwfc.evaluate import evaluate_forecaster
from cwfc.baselines import persistence_forecast


def _series(n=110):
    idx = pd.Index(np.arange(0, n * 300, 300), name="timestamp")
    return pd.DataFrame({"y": np.arange(n, dtype=float)}, index=idx)


def test_moving_average_uses_only_last_window():
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]})
    f = moving_average_forecast(df, df.iloc[:2], "y", window=2)
    assert list(f) == [3.5, 3.5]


def test_training_mean():
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    assert list(training_mean_forecast(df, df.iloc[:1], "y")) == [2.0]


def test_evaluation_makes_four_folds_and_no_leak():
    folds, errs = evaluate_forecaster(_series(), persistence_forecast, "y")
    assert len(folds) == 4 and errs.shape == (4, 24)