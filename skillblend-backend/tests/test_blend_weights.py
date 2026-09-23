import numpy as np

from app.engine.blend import (
    equal_average_weights,
    skill_weighted_average_weights,
    softmax_weights,
    weighted_blend,
)


def test_softmax_weights_sum_to_one_and_nonnegative():
    weights = softmax_weights(["GFS", "IFS", "AIFS"], np.array([5.0, 2.0, 8.0]))
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert all(w >= 0 for w in weights.values())
    # lowest expected error (IFS=2.0) should get the highest weight
    assert weights["IFS"] == max(weights.values())


def test_equal_average_weights():
    weights = equal_average_weights(["GFS", "IFS", "AIFS"])
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert all(abs(w - 1 / 3) < 1e-9 for w in weights.values())


def test_skill_weighted_average_weights_favor_higher_skill():
    weights = skill_weighted_average_weights(
        ["GFS", "IFS"], {"GFS": 0.9, "IFS": 0.3}
    )
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["GFS"] > weights["IFS"]


def test_skill_weighted_average_handles_zero_skill_gracefully():
    # should not divide by zero / produce nan
    weights = skill_weighted_average_weights(["GFS", "IFS"], {"GFS": 0.0, "IFS": 0.0})
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert not any(np.isnan(w) for w in weights.values())


def test_weighted_blend_matches_manual_computation():
    values = {"GFS": 10.0, "IFS": 20.0}
    weights = {"GFS": 0.25, "IFS": 0.75}
    assert abs(weighted_blend(values, weights) - 17.5) < 1e-9


def test_missing_source_still_produces_valid_weights():
    """Section 20's 'Fallback' test: if AIFS is missing, blend uses remaining sources safely."""
    available = ["GFS", "IFS"]  # AIFS missing
    weights = skill_weighted_average_weights(available, {"GFS": 0.6, "IFS": 0.7})
    assert set(weights) == set(available)
    assert abs(sum(weights.values()) - 1.0) < 1e-9
