import numpy as np
import pandas as pd
import pytest

from app.engine.skill import _csi_pod_far, _event_counts, compute_skill_table


def test_csi_pod_far_hand_computed():
    # 4 cases: hit, miss, false alarm, correct negative (threshold = 10)
    df = pd.DataFrame(
        {
            "forecast_value": [12.0, 4.0, 15.0, 3.0],
            "actual_value": [11.0, 12.0, 5.0, 2.0],
        }
    )
    hits, misses, fa, cn = _event_counts(df, threshold=10.0)
    assert (hits, misses, fa, cn) == (1, 1, 1, 1)

    csi, pod, far = _csi_pod_far(hits, misses, fa)
    assert csi == pytest.approx(1 / 3)
    assert pod == pytest.approx(0.5)
    assert far == pytest.approx(0.5)


def test_compute_skill_table_groups_and_metrics_are_sane():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "model": ["GFS"] * n,
            "region": ["KWG"] * n,
            "variable": ["precipitation"] * n,
            "lead_hours": [72] * n,
            "season": ["sw_monsoon"] * n,
            "regime": ["active_monsoon"] * n,
            "forecast_value": rng.normal(50, 5, n),
            "actual_value": rng.normal(50, 5, n),
            "valid_time": pd.date_range("2026-01-01", periods=n, freq="D"),
        }
    )
    table = compute_skill_table(df)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["n_obs"] == n
    assert row["rmse"] > 0
    assert row["mae"] > 0
    assert 0 <= row["csi"] <= 1
