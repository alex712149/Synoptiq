import numpy as np

from app.engine.calibration import (
    ExceedanceCalibrator,
    QuantileMapper,
    raw_exceedance_score,
)


def test_quantile_mapper_removes_systematic_multiplicative_bias():
    rng = np.random.default_rng(0)
    actual = rng.gamma(2, 20, 2000)
    biased_forecast = actual * 1.3 + 5  # systematic over-forecast bias

    mapper = QuantileMapper.fit(biased_forecast, actual)
    corrected = np.array([mapper.transform(v) for v in biased_forecast])

    # after correction, mean error should shrink substantially vs before
    error_before = np.abs(biased_forecast - actual).mean()
    error_after = np.abs(corrected - actual).mean()
    assert error_after < error_before


def test_quantile_mapper_calibration_uses_only_training_split():
    """Section 9's leakage guardrail: fit on train, evaluate on a disjoint holdout."""
    rng = np.random.default_rng(1)
    actual_train = rng.gamma(2, 20, 1000)
    forecast_train = actual_train * 1.2

    actual_holdout = rng.gamma(2, 20, 300)
    forecast_holdout = actual_holdout * 1.2

    mapper = QuantileMapper.fit(forecast_train, actual_train)
    corrected_holdout = np.array([mapper.transform(v) for v in forecast_holdout])

    error_before = np.abs(forecast_holdout - actual_holdout).mean()
    error_after = np.abs(corrected_holdout - actual_holdout).mean()
    assert error_after < error_before


def test_raw_exceedance_score_increases_with_value():
    low = raw_exceedance_score(20.0, threshold=50.0, disagreement=10.0)
    high = raw_exceedance_score(80.0, threshold=50.0, disagreement=10.0)
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low


def test_exceedance_calibrator_is_monotonic_and_bounded():
    rng = np.random.default_rng(2)
    raw_scores = rng.uniform(0, 1, 500)
    observed = (rng.uniform(0, 1, 500) < raw_scores).astype(int)  # noisy but correlated

    calibrator = ExceedanceCalibrator.fit(raw_scores, observed)
    low = calibrator.transform(0.1)
    high = calibrator.transform(0.9)
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high >= low
