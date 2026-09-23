"""
End-to-end integration test against the real trained artifacts. Run
`python -m scripts.seed_and_train` before this test suite, exactly as
CI should. The test skips (rather than fails) if artifacts are not
present, so `pytest` still passes cleanly on a fresh checkout - the
README explains this is expected and shows the one-line fix.
"""
import pandas as pd
import pytest

from app.config import HISTORICAL_PARQUET, LEAD_HOURS_LADDER, PILOT_ZONES, VARIABLES
from app.data.storage import _pkl_path

_SEEDED = HISTORICAL_PARQUET.exists() or _pkl_path(HISTORICAL_PARQUET).exists()

pytestmark = pytest.mark.skipif(
    not _SEEDED, reason="artifacts not seeded - run `python -m scripts.seed_and_train` first"
)


@pytest.fixture(scope="module")
def pipeline():
    from app.services import pipeline as pipeline_module

    pipeline_module.reload_state()
    return pipeline_module


def test_blend_forecast_returns_all_expected_fields(pipeline):
    result = pipeline.blend_forecast("KWG", "precipitation", 72)
    for key in (
        "region", "variable", "unit", "run_time", "valid_time", "lead_hours",
        "regime", "regime_probs", "sources", "disagreement", "final_value",
        "trust", "bust_probability", "abstain", "explanation", "provenance",
    ):
        assert key in result


def test_weights_sum_to_one(pipeline):
    result = pipeline.blend_forecast("BOB", "wind_speed", 48)
    total_weight = sum(s["weight"] for s in result["sources"])
    assert abs(total_weight - 1.0) < 1e-6


def test_weights_are_finite_and_nonnegative(pipeline):
    result = pipeline.blend_forecast("IGP", "temperature", 120)
    for s in result["sources"]:
        assert s["weight"] >= 0
        assert s["weight"] == s["weight"]  # not NaN


def test_extreme_guidance_probabilities_in_bounds(pipeline):
    result = pipeline.extreme_guidance("KWG", 72)
    for g in result["guidance"]:
        assert 0.0 <= g["probability"] <= 1.0


def test_weight_map_covers_full_lead_ladder(pipeline):
    result = pipeline.weight_map("IGP", "precipitation", season="sw_monsoon", regime="active_monsoon")
    leads = [p["lead_hours"] for p in result["points"]]
    assert leads == LEAD_HOURS_LADDER


def test_unknown_region_raises_value_error(pipeline):
    with pytest.raises(ValueError):
        pipeline.blend_forecast("XXX", "precipitation", 72)


def test_all_pilot_zone_variable_combinations_run_without_error(pipeline):
    for region in PILOT_ZONES:
        for variable in VARIABLES:
            result = pipeline.blend_forecast(region, variable, LEAD_HOURS_LADDER[0])
            assert result["final_value"] == result["final_value"]  # not NaN
