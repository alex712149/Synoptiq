"""
Bias correction + probability calibration (blueprint Section 9).

    Raw forecasts -> adaptive blend -> bias correction -> probability
    calibration -> final product.

- Bias correction: quantile mapping (student-feasible baseline per the
  blueprint), fit per (region, variable) on a training split and,
  where enough samples exist, refined per weather regime ("a
  regime-conditioned version can use a separate mapping per weather
  regime").
- Probability calibration: isotonic regression turning a raw
  threshold-exceedance score into a statistically reliable probability
  (used for the extreme-guidance endpoints).

Both calibrators are always fit on a *training* split and evaluated on
a disjoint *holdout* split - see engine/verification-style leakage
test in tests/test_calibration.py - to satisfy the blueprint's
explicit "use held-out temporal data for every calibration step to
avoid leakage" requirement (Section 9, bullet 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class QuantileMapper:
    """Empirical quantile-mapping bias corrector."""

    source_quantiles: np.ndarray
    target_quantiles: np.ndarray
    n_points: int = 101

    @classmethod
    def fit(cls, source_values: np.ndarray, target_values: np.ndarray, n_points: int = 101) -> "QuantileMapper":
        qs = np.linspace(0, 100, n_points)
        source_q = np.percentile(source_values, qs)
        target_q = np.percentile(target_values, qs)
        # enforce monotonicity (percentile is monotone in theory; guard against ties)
        source_q = np.maximum.accumulate(source_q)
        target_q = np.maximum.accumulate(target_q)
        return cls(source_quantiles=source_q, target_quantiles=target_q, n_points=n_points)

    def transform(self, value: float) -> float:
        return float(np.interp(value, self.source_quantiles, self.target_quantiles))


@dataclass
class RegimeAwareBiasCorrector:
    """Per-(region, variable) global mapper, refined per-regime when enough data exists."""

    global_mapper: QuantileMapper
    regime_mappers: dict[str, QuantileMapper] = field(default_factory=dict)
    min_samples_for_regime: int = 40

    @classmethod
    def fit(cls, df, region: str, variable: str, min_samples_for_regime: int = 40) -> "RegimeAwareBiasCorrector":
        subset = df[(df["region"] == region) & (df["variable"] == variable)]
        global_mapper = QuantileMapper.fit(subset["blend_value"].to_numpy(), subset["actual_value"].to_numpy())

        regime_mappers = {}
        for regime, g in subset.groupby("regime", observed=True):
            if len(g) >= min_samples_for_regime:
                regime_mappers[regime] = QuantileMapper.fit(
                    g["blend_value"].to_numpy(), g["actual_value"].to_numpy()
                )
        return cls(global_mapper=global_mapper, regime_mappers=regime_mappers, min_samples_for_regime=min_samples_for_regime)

    def transform(self, value: float, regime: str) -> float:
        mapper = self.regime_mappers.get(regime, self.global_mapper)
        return mapper.transform(value)


# ---------------------------------------------------------------------------
# Extreme-threshold probability: raw score -> isotonic calibration
# ---------------------------------------------------------------------------
def raw_exceedance_score(blended_value: float, threshold: float, disagreement: float, scale_floor: float = 1.0) -> float:
    """
    Logistic-shaped raw probability that the true value exceeds
    `threshold`, before calibration. Higher inter-model disagreement
    widens the transition band (more uncertainty near the threshold),
    which is the same intuition the Trust engine uses for
    'disagreement lowers confidence'.
    """
    scale = max(scale_floor, disagreement)
    z = (blended_value - threshold) / scale
    return float(1.0 / (1.0 + np.exp(-z)))


@dataclass
class ExceedanceCalibrator:
    isotonic: IsotonicRegression

    @classmethod
    def fit(cls, raw_scores: np.ndarray, observed_exceedance: np.ndarray) -> "ExceedanceCalibrator":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_scores, observed_exceedance)
        return cls(isotonic=iso)

    def transform(self, raw_score: float) -> float:
        return float(self.isotonic.predict([raw_score])[0])
