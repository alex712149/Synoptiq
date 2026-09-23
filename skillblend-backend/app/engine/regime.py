"""
Context / Regime engine (blueprint Section 7).

"STUDENT-FRIENDLY REGIME DETECTOR: start with a rule-based regime
classifier using transparent meteorological thresholds/features."

This module implements exactly that: a small set of auditable
if/then thresholds over source-forecast signals (ensemble spread,
rainfall intensity, wind, season, coastal flag) that produce a
regime-probability distribution. Every rule is inspectable, which is
what makes the downstream trust/explanation panel defensible instead
of a black box. An ML regime classifier is listed in the roadmap
(Section 21) as a later upgrade once the end-to-end system is stable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config import REGIMES, season_for_month


@dataclass
class RegimeSignal:
    mean_precip_forecast: float
    mean_wind_forecast: float
    source_spread_precip: float
    month: int
    coastal: bool
    wd_sensitivity: float
    monsoon_sensitivity: float


def classify_regime(signal: RegimeSignal) -> dict[str, float]:
    """
    Deterministic, threshold-based regime probabilities. Returns a
    normalized dict over app.config.REGIMES. Documented thresholds:

      - depression: heavy rain (>=70 mm) AND high wind (>=15 m/s),
        boosted near the coast.
      - active_monsoon: heavy rain (>=25mm) in the SW-monsoon season
        for monsoon-sensitive zones.
      - western_disturbance: cool-season rain/wind bump in a
        WD-sensitive zone during winter/pre-monsoon.
      - break_monsoon: SW-monsoon season but low rainfall signal.
      - normal: default / residual probability mass.
    """
    season = season_for_month(signal.month)
    scores = {r: 0.05 for r in REGIMES}  # small floor so nothing is exactly zero

    if signal.mean_precip_forecast >= 70 and signal.mean_wind_forecast >= 15:
        scores["depression"] += 3.0 * (1.3 if signal.coastal else 1.0)
    elif signal.mean_precip_forecast >= 40 and signal.mean_wind_forecast >= 10:
        scores["depression"] += 1.2

    if season == "sw_monsoon":
        if signal.mean_precip_forecast >= 25:
            scores["active_monsoon"] += 2.5 * (0.5 + signal.monsoon_sensitivity)
        else:
            scores["break_monsoon"] += 1.8

    if season in ("winter", "pre_monsoon") and signal.wd_sensitivity > 0.3:
        if signal.mean_precip_forecast >= 5 or signal.source_spread_precip >= 8:
            scores["western_disturbance"] += 2.0 * signal.wd_sensitivity

    # High disagreement between sources on rainfall nudges away from "normal"
    # confidence and towards an active/organised-system interpretation.
    if signal.source_spread_precip >= 20:
        scores["depression"] += 0.5
        scores["active_monsoon"] += 0.3

    if signal.mean_precip_forecast < 5 and signal.mean_wind_forecast < 8:
        scores["normal"] += 2.0

    total = sum(scores.values())
    return {k: round(v / total, 4) for k, v in scores.items()}


def dominant_regime(regime_probs: dict[str, float]) -> str:
    return max(regime_probs.items(), key=lambda kv: kv[1])[0]


def regime_stability(regime_probs: dict[str, float]) -> float:
    """
    A simple, explainable stability score: 1 - normalized entropy of the
    regime distribution. A confidently single-regime forecast (low
    entropy) is "stable"; a forecast split across several regimes
    (high entropy) signals a fast-changing / ambiguous weather state,
    which the Trust engine treats as a negative signal (Section 10).
    """
    probs = np.array(list(regime_probs.values()))
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(REGIMES))
    return float(np.clip(1 - entropy / max_entropy, 0.0, 1.0))
