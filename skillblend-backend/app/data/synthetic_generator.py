"""
Synthetic forecast generator.

SkillBlend's reference design (blueprint Section 4) ingests live GFS,
ECMWF IFS/AIFS Open Data, ERA5 and IMERG. This prototype runs without
network access to those services, so this module produces a
*physically-motivated* synthetic stand-in: each source has its own
systematic bias/noise characteristics that vary by pilot zone, season,
regime and lead time - deliberately engineered so that "which model to
trust" genuinely depends on context, which is the whole premise of
SkillBlend. Swapping this module for real cfgrib/xarray ingestion is
the only change needed to go from prototype to production; every
downstream module (normalization, skill engine, blending, API) is
unaffected because both paths emit the same canonical schema.

Design of the source biases (used consistently to make the ML story
coherent and inspectable):

- GFS: reliable all-rounder, slightly noisy at long lead times, no
  strong regime specialisation. The "safe baseline".
- IFS: best physical model for organised heavy-rainfall systems
  (active monsoon, depressions) and coastal regions; a bit conservative
  (low bias) on temperature extremes.
- AIFS: best for short-to-medium lead, "normal"/quiescent regimes and
  smooth temperature/wind fields (AI models are known to under-predict
  sharp extremes); degrades faster than IFS at long lead times and
  under-predicts rainfall in active monsoon bursts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import (
    BUST_ERROR_THRESHOLDS,
    HISTORY_DAYS,
    LEAD_HOURS_LADDER,
    PILOT_ZONES,
    RANDOM_SEED,
    REGIMES,
    SOURCES,
    VARIABLES,
    VARIABLE_UNITS,
    season_for_month,
)


def _regime_for_day(rng: np.random.Generator, month: int, zone_monsoon: float, zone_wd: float) -> str:
    """Rule-based-ish regime sampler: probabilities shift with season/zone."""
    season = season_for_month(month)
    if season == "sw_monsoon":
        p_active = 0.45 * zone_monsoon + 0.1
        p_break = 0.15
        p_depression = 0.15 * zone_monsoon
        p_wd = 0.02
        p_normal = max(0.05, 1 - (p_active + p_break + p_depression + p_wd))
    elif season == "winter":
        p_wd = 0.5 * zone_wd + 0.05
        p_active = 0.02
        p_break = 0.05
        p_depression = 0.02
        p_normal = max(0.05, 1 - (p_wd + p_active + p_break + p_depression))
    elif season == "pre_monsoon":
        p_active = 0.05
        p_break = 0.05
        p_depression = 0.08
        p_wd = 0.15 * zone_wd
        p_normal = max(0.05, 1 - (p_active + p_break + p_depression + p_wd))
    else:  # post_monsoon
        p_active = 0.1 * zone_monsoon
        p_break = 0.1
        p_depression = 0.2 * (1 if not zone_monsoon else zone_monsoon)
        p_wd = 0.05 * zone_wd
        p_normal = max(0.05, 1 - (p_active + p_break + p_depression + p_wd))

    probs = np.array([p_active, p_break, p_wd, p_depression, p_normal])
    probs = np.clip(probs, 0.0, None)
    probs = probs / probs.sum()
    return str(rng.choice(REGIMES, p=probs))


def _true_precip(rng, regime: str, zone_monsoon: float, coastal: bool) -> float:
    base = {
        "active_monsoon": 55 * zone_monsoon + 10,
        "break_monsoon": 3,
        "western_disturbance": 12,
        "depression": 90 if coastal else 60,
        "normal": 4,
    }[regime]
    shape = 1.4
    value = rng.gamma(shape, max(base, 0.5) / shape)
    return float(max(0.0, value))


def _true_temperature(rng, regime: str, month: int, zone: str) -> float:
    season_mean = {
        "winter": {"KWG": 27, "BOB": 26, "IGP": 16},
        "pre_monsoon": {"KWG": 31, "BOB": 32, "IGP": 38},
        "sw_monsoon": {"KWG": 28, "BOB": 30, "IGP": 33},
        "post_monsoon": {"KWG": 29, "BOB": 29, "IGP": 26},
    }[season_for_month(month)][zone]
    if regime == "western_disturbance":
        season_mean -= 4
    if regime in ("active_monsoon", "depression"):
        season_mean -= 2
    return float(rng.normal(season_mean, 1.6))


def _true_wind(rng, regime: str, coastal: bool) -> float:
    base = {
        "active_monsoon": 12,
        "break_monsoon": 5,
        "western_disturbance": 8,
        "depression": 20 if coastal else 14,
        "normal": 5,
    }[regime]
    return float(max(0.0, rng.normal(base, base * 0.25 + 1)))


def _source_bias_noise(model: str, variable: str, regime: str, lead_hours: int, coastal: bool):
    """Return (multiplicative_bias, additive_bias, noise_std) for a source."""
    lead_factor = lead_hours / 24.0  # degrade with lead time

    if variable == "precipitation":
        if model == "GFS":
            mult, add = 1.0, 0.0
            noise = 6 + 2.2 * lead_factor
        elif model == "IFS":
            # Best for organised heavy-rain systems and coastal regimes.
            if regime in ("active_monsoon", "depression"):
                mult, add = 1.03, 1.0
                noise = 4 + 1.4 * lead_factor
            else:
                mult, add = 0.97, 0.0
                noise = 5 + 1.8 * lead_factor
            if coastal:
                noise *= 0.85
        else:  # AIFS
            if regime in ("active_monsoon", "depression"):
                # AI models tend to under-predict sharp convective/extreme bursts.
                mult, add = 0.78, -2.0
                noise = 7 + 2.6 * lead_factor
            else:
                mult, add = 1.02, 0.2
                noise = 3.5 + 1.1 * lead_factor
        # All sources degrade faster at long lead; AIFS degrades fastest.
        extra = {"GFS": 1.0, "IFS": 0.9, "AIFS": 1.25}[model]
        noise *= (1 + 0.12 * lead_factor * extra)
        return mult, add, noise

    if variable == "temperature":
        if model == "GFS":
            mult, add, noise = 1.0, 0.0, 0.9 + 0.15 * lead_factor
        elif model == "IFS":
            add = -0.3 if regime in ("active_monsoon", "depression") else 0.1
            mult, noise = 1.0, 0.8 + 0.12 * lead_factor
        else:  # AIFS - smooth, slightly under-predicts extremes
            add = -0.6 if regime == "western_disturbance" else 0.2
            mult, noise = 1.0, 0.7 + 0.20 * lead_factor
        return mult, add, noise

    # wind_speed
    if model == "GFS":
        mult, add, noise = 1.0, 0.0, 1.3 + 0.2 * lead_factor
    elif model == "IFS":
        mult, add = (1.05, 0.3) if regime == "depression" else (0.98, 0.0)
        noise = 1.1 + 0.15 * lead_factor
        if coastal:
            noise *= 0.9
    else:  # AIFS - under-predicts gale-force tails
        mult, add = (0.82, -1.0) if regime == "depression" else (1.0, 0.1)
        noise = 1.4 + 0.28 * lead_factor
    return mult, add, noise


def generate_historical_dataset(
    days: int = HISTORY_DAYS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a long-format synthetic historical dataset covering every
    pilot zone x variable x source x lead-time combination for `days`
    forecast cycles (one 00Z cycle per day, matching the blueprint's
    "1 variable, limited lead-time ladder" MVP guardrail, extended here
    to all three PS variables since it costs nothing extra to simulate).

    Columns emitted match the canonical schema plus an `actual_value`
    reference column (stands in for ERA5/IMERG verification) and the
    `regime`/`season` context used by the Skill and Context engines.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-06-01T00:00:00Z")
    dates = [start + pd.Timedelta(days=i) for i in range(days)]

    rows = []
    for zone_code, zone in PILOT_ZONES.items():
        for run_time in dates:
            month = run_time.month
            regime = _regime_for_day(rng, month, zone.monsoon_sensitivity, zone.wd_sensitivity)
            season = season_for_month(month)

            for lead_hours in LEAD_HOURS_LADDER:
                valid_time = run_time + pd.Timedelta(hours=lead_hours)

                truth = {
                    "precipitation": _true_precip(rng, regime, zone.monsoon_sensitivity, zone.coastal),
                    "temperature": _true_temperature(rng, regime, valid_time.month, zone_code),
                    "wind_speed": _true_wind(rng, regime, zone.coastal),
                }

                for variable in VARIABLES:
                    actual = truth[variable]
                    for model in SOURCES:
                        mult, add, noise_std = _source_bias_noise(
                            model, variable, regime, lead_hours, zone.coastal
                        )
                        noise = rng.normal(0, noise_std)
                        forecast_value = actual * mult + add + noise
                        if variable in ("precipitation", "wind_speed"):
                            forecast_value = max(0.0, forecast_value)

                        rows.append(
                            {
                                "model": model,
                                "run_time": run_time,
                                "valid_time": valid_time,
                                "lead_hours": lead_hours,
                                "variable": variable,
                                "region": zone_code,
                                "lat": zone.lat,
                                "lon": zone.lon,
                                "forecast_value": round(float(forecast_value), 3),
                                "actual_value": round(float(actual), 3),
                                "unit": VARIABLE_UNITS[variable],
                                "regime": regime,
                                "season": season,
                            }
                        )

    df = pd.DataFrame.from_records(rows)
    df["abs_error"] = (df["forecast_value"] - df["actual_value"]).abs()
    df["bust"] = df.apply(
        lambda r: r["abs_error"] > BUST_ERROR_THRESHOLDS[r["variable"]], axis=1
    )
    return df


if __name__ == "__main__":
    data = generate_historical_dataset(days=30)
    print(data.head(10).to_string())
    print(f"rows: {len(data)}")
