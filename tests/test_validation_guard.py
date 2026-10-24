import numpy as np
import pandas as pd
from cwfc.evaluate import make_frame
from cwfc.validation import rolling_origin_splits


def test_validation_never_reaches_holdout():
    idx = pd.Index(np.arange(0, 134 * 300, 300), name="timestamp")
    t = pd.DataFrame({"y": np.arange(134, dtype=float)}, index=idx)
    train, holdout = t.iloc[:110], t.iloc[110:]
    splits = rolling_origin_splits(make_frame(train), horizon=24, min_train_size=50, step=12)
    assert len(splits) == 4
    first_holdout = pd.to_datetime(holdout.index[0], unit="s", utc=True)
    for s in splits:
        assert s.train_end < s.validation_start
        assert s.validation_end < first_holdout