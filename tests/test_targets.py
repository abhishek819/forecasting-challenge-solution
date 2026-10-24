import numpy as np
import pandas as pd

from cwfc.targets import build_targets, split_train_holdout


def test_vm_counted_once_and_gap_not_invented():
    cpu = pd.DataFrame({
        "timestamp": [0, 0, 300, 300, 300, 900],
        "vm_id": ["a", "b", "a", "b", "b", "a"],      # b duplicated at t=300
        "avg_cpu": [10.0, 20.0, 30.0, 40.0, 99.0, 50.0],
    })
    t = build_targets(cpu)
    assert list(t.index) == [0, 300, 600, 900]
    assert t.loc[300, "active_vm_count"] == 2
    assert t.loc[300, "aggregate_cpu_utilization"] == 35.0   # duplicate ignored
    assert np.isnan(t.loc[600, "aggregate_cpu_utilization"])  # gap stays NaN


def test_invalid_readings_are_not_counted():
    cpu = pd.DataFrame({
        "timestamp": [0, 0, 0],
        "vm_id": ["a", "b", "c"],
        "avg_cpu": [10.0, np.nan, 150.0],
    })
    t = build_targets(cpu)
    assert t.loc[0, "active_vm_count"] == 1
    assert t.loc[0, "aggregate_cpu_utilization"] == 10.0


def test_split_is_chronological():
    idx = np.arange(0, 134 * 300, 300)
    t = pd.DataFrame({"active_vm_count": 1.0, "aggregate_cpu_utilization": 2.0}, index=idx)
    train, holdout = split_train_holdout(t)
    assert (len(train), len(holdout)) == (110, 24)
    assert train.index.max() < holdout.index.min()