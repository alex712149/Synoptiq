import numpy as np
from app.blending.blend import compute_weights

def test_weights_are_finite_nonnegative_and_sum_to_one():
    forecast_values = {"GFS": 40.0, "IFS": 170.0, "AIFS": 95.0}
    hist_skill = {"GFS": 0.4, "IFS": 0.7, "AIFS": 0.55}
    weights, source = compute_weights(
        variable="precipitation", lead_hours=72,
        regime_probs={"active_monsoon": 0.8, "normal": 0.2}, region="kerala_western_ghats",
        season="sw_monsoon", historical_skill_by_model=hist_skill, disagreement=50.0,
        forecast_values=forecast_values,
    )
    assert source in ("meta_model", "skill_fallback")
    vals = np.array(list(weights.values()))
    assert np.all(np.isfinite(vals))
    assert np.all(vals >= 0)
    assert abs(vals.sum() - 1.0) < 1e-3

def test_fallback_when_source_missing():
    # Only two of three sources available -> weights must still be valid and
    # only cover the available sources (Section 20: "If AIFS is missing,
    # blend uses remaining sources safely").
    forecast_values = {"GFS": 40.0, "IFS": 170.0}
    hist_skill = {"GFS": 0.4, "IFS": 0.7}
    weights, _ = compute_weights(
        variable="precipitation", lead_hours=72, regime_probs={"normal": 1.0},
        region="bay_of_bengal_east_coast", season="post_monsoon",
        historical_skill_by_model=hist_skill, disagreement=65.0, forecast_values=forecast_values,
    )
    assert set(weights.keys()) == {"GFS", "IFS"}
    assert abs(sum(weights.values()) - 1.0) < 1e-3
