import numpy as np
import pandas as pd

from cwfc.predict import COLUMNS, make_predictions, validate_predictions


def _write(folder, holdout_value=None, n_vms=20):
    rng = np.random.default_rng(1)
    ts = np.arange(0, 134 * 300, 300)
    rows = []
    for v in range(n_vms):
        base = rng.uniform(2, 20)
        for i, t in enumerate(ts):
            val = float(np.clip(base + rng.normal(0, 1), 0, 100))
            if holdout_value is not None and i >= 110:
                val = holdout_value                      # sabotage the holdout
            rows.append((t, f"vm{v}", val, val, val))
    pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"]) \
      .to_csv(folder / "vm_cpu_readings.csv.gz", index=False)


def test_predictions_ignore_holdout_targets(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    _write(a)
    _write(b, holdout_value=99.0)
    pd.testing.assert_frame_equal(make_predictions(a), make_predictions(b))


def test_prediction_file_contract(tmp_path):
    _write(tmp_path, n_vms=3)                            # tiny "smoke" dataset
    p = make_predictions(tmp_path)
    validate_predictions(p)
    assert list(p.columns) == COLUMNS and len(p) == 24
    assert p["timestamp"].iloc[0] == 110 * 300 and p["timestamp"].iloc[-1] == 133 * 300