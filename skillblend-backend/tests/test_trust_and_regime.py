from app.engine.regime import RegimeSignal, classify_regime, dominant_regime, regime_stability
from app.engine.trust import compute_disagreement, compute_trust_breakdown, normalize_disagreement, should_abstain


def test_trust_breakdown_components_are_bounded_0_1():
    tb = compute_trust_breakdown(
        historical_skill_avg=0.8,
        disagreement_norm=0.2,
        lead_hours=72,
        regime_stability=0.9,
        data_quality=1.0,
    )
    for value in (
        tb.historical_skill_component,
        tb.disagreement_component,
        tb.lead_time_component,
        tb.regime_stability_component,
        tb.data_quality_component,
        tb.trust_score,
    ):
        assert 0.0 <= value <= 1.0


def test_trust_score_decreases_with_longer_lead_time():
    short = compute_trust_breakdown(0.8, 0.2, 24, 0.9, 1.0)
    long = compute_trust_breakdown(0.8, 0.2, 168, 0.9, 1.0)
    assert short.trust_score > long.trust_score


def test_trust_score_decreases_with_more_disagreement():
    low_disagreement = compute_trust_breakdown(0.8, 0.1, 72, 0.9, 1.0)
    high_disagreement = compute_trust_breakdown(0.8, 0.9, 72, 0.9, 1.0)
    assert low_disagreement.trust_score > high_disagreement.trust_score


def test_should_abstain_flags_low_trust_or_high_bust_risk():
    assert should_abstain(trust_score=0.2, bust_probability=0.1) is True
    assert should_abstain(trust_score=0.8, bust_probability=0.9) is True
    assert should_abstain(trust_score=0.8, bust_probability=0.1) is False


def test_compute_disagreement_and_normalization():
    values = {"GFS": 40.0, "IFS": 60.0, "AIFS": 50.0}
    spread = compute_disagreement(values)
    assert spread > 0
    norm = normalize_disagreement(spread, "precipitation")
    assert norm > 0


def test_regime_classifier_is_deterministic_and_normalized():
    signal = RegimeSignal(
        mean_precip_forecast=80.0,
        mean_wind_forecast=18.0,
        source_spread_precip=15.0,
        month=7,
        coastal=True,
        wd_sensitivity=0.1,
        monsoon_sensitivity=0.9,
    )
    probs_a = classify_regime(signal)
    probs_b = classify_regime(signal)
    assert probs_a == probs_b  # deterministic, auditable - no randomness
    assert abs(sum(probs_a.values()) - 1.0) < 1e-3  # small tolerance: values are rounded to 4dp for API readability
    # heavy rain + high wind in monsoon season should read as depression-like
    assert dominant_regime(probs_a) in ("depression", "active_monsoon")


def test_regime_stability_is_high_for_confident_distribution_low_for_uniform():
    confident = {"active_monsoon": 0.9, "break_monsoon": 0.025, "western_disturbance": 0.025, "depression": 0.025, "normal": 0.025}
    uniform = {"active_monsoon": 0.2, "break_monsoon": 0.2, "western_disturbance": 0.2, "depression": 0.2, "normal": 0.2}
    assert regime_stability(confident) > regime_stability(uniform)
