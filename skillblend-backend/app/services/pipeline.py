"""
End-to-end pipeline orchestration - this is the module that stitches
together every engine described in the blueprint (Figure 2 / Section
14's "Operational Data Flow") into the single call each API route
needs:

    ingest -> harmonize -> score(historical skill) -> blend(meta-model)
    -> correct(quantile mapping) -> calibrate(isotonic) -> serve

Important realism note: the *live* weather regime used for blending is
always derived from the auditable rule-based classifier
(`engine.regime.classify_regime`) applied to the current raw source
forecasts - never from the "ground truth" regime label that only
exists inside the synthetic generator. The historical Skill Engine, by
contrast, is legitimately allowed to use analyzed/hindcast regimes
when building its offline skill table, exactly as an operational
verification team would use best-estimate reanalysis-informed regime
classifications after the fact. Keeping this separation is what makes
the pipeline a fair simulation of a real deployment rather than a
system that secretly cheats off its own synthetic ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from app.config import (
    BLEND_BASELINE_PATH,
    BUST_BASELINE_PATH,
    CALIBRATORS_PATH,
    EXTREME_THRESHOLDS,
    LEAD_HOURS_LADDER,
    PILOT_ZONES,
    SOURCE_META,
    SOURCES,
    VARIABLE_UNITS,
    season_for_month,
)
from app.data import storage
from app.data.normalization import build_quality_mask
from app.engine import blend as blend_engine
from app.engine import calibration as calib_engine
from app.engine import explain as explain_engine
from app.engine import regime as regime_engine
from app.engine import skill as skill_engine
from app.engine import trust as trust_engine

import joblib


class PipelineNotReady(RuntimeError):
    """Raised when scripts/seed_and_train.py has not been run yet."""


@dataclass
class _State:
    historical: pd.DataFrame
    skill_table: pd.DataFrame
    blend_model: blend_engine.BlendModel | None
    bust_model: trust_engine.BustModel | None
    bias_correctors: dict
    exceedance_calibrators: dict
    blend_baseline: pd.Series | None
    bust_baseline: pd.Series | None


_STATE: _State | None = None


def _load_state(force: bool = False) -> _State:
    global _STATE
    if _STATE is not None and not force:
        return _STATE
    try:
        historical = storage.load_historical()
        skill_table = storage.load_skill_table()
    except FileNotFoundError as exc:
        raise PipelineNotReady(
            "No cached historical data / skill table found. Run "
            "`python -m scripts.seed_and_train` first."
        ) from exc

    blend_model = blend_engine.get_blend_model(force_reload=force)
    bust_model = trust_engine.get_bust_model(force_reload=force)

    bias_correctors, exceedance_calibrators = {}, {}
    if CALIBRATORS_PATH.exists():
        payload = joblib.load(CALIBRATORS_PATH)
        bias_correctors = payload.get("bias_correctors", {})
        exceedance_calibrators = payload.get("exceedance_calibrators", {})

    blend_baseline = joblib.load(BLEND_BASELINE_PATH) if BLEND_BASELINE_PATH.exists() else None
    bust_baseline = joblib.load(BUST_BASELINE_PATH) if BUST_BASELINE_PATH.exists() else None

    _STATE = _State(
        historical=historical,
        skill_table=skill_table,
        blend_model=blend_model,
        bust_model=bust_model,
        bias_correctors=bias_correctors,
        exceedance_calibrators=exceedance_calibrators,
        blend_baseline=blend_baseline,
        bust_baseline=bust_baseline,
    )
    return _STATE


def reload_state() -> None:
    _load_state(force=True)


# ---------------------------------------------------------------------------
# Ingestion simulation (stands in for live GFS/IFS/AIFS polling)
# ---------------------------------------------------------------------------
def latest_run_time(region: str, variable: str, lead_hours: int) -> pd.Timestamp:
    state = _load_state()
    subset = state.historical[
        (state.historical["region"] == region)
        & (state.historical["variable"] == variable)
        & (state.historical["lead_hours"] == lead_hours)
    ]
    if subset.empty:
        raise ValueError(f"no cached data for region={region} variable={variable} lead_hours={lead_hours}")
    return pd.Timestamp(subset["run_time"].max())


def _fetch_rows(region: str, variable: str, lead_hours: int, run_time: pd.Timestamp) -> pd.DataFrame:
    state = _load_state()
    mask = (
        (state.historical["region"] == region)
        & (state.historical["variable"] == variable)
        & (state.historical["lead_hours"] == lead_hours)
        & (state.historical["run_time"] == run_time)
    )
    return state.historical[mask]


def _regime_signal(region: str, lead_hours: int, run_time: pd.Timestamp) -> regime_engine.RegimeSignal:
    """Build the live regime-classifier signal from raw source forecasts only."""
    zone = PILOT_ZONES[region]
    precip_rows = _fetch_rows(region, "precipitation", lead_hours, run_time)
    wind_rows = _fetch_rows(region, "wind_speed", lead_hours, run_time)

    mean_precip = float(precip_rows["forecast_value"].mean()) if not precip_rows.empty else 0.0
    spread_precip = float(precip_rows["forecast_value"].std()) if len(precip_rows) > 1 else 0.0
    mean_wind = float(wind_rows["forecast_value"].mean()) if not wind_rows.empty else 0.0

    month = int(run_time.month)
    return regime_engine.RegimeSignal(
        mean_precip_forecast=mean_precip,
        mean_wind_forecast=mean_wind,
        source_spread_precip=spread_precip,
        month=month,
        coastal=zone.coastal,
        wd_sensitivity=zone.wd_sensitivity,
        monsoon_sensitivity=zone.monsoon_sensitivity,
    )


# ---------------------------------------------------------------------------
# Core blend pipeline
# ---------------------------------------------------------------------------
def blend_forecast(region: str, variable: str, lead_hours: int, run_time: pd.Timestamp | None = None) -> dict:
    state = _load_state()
    region = region.upper()
    variable = variable.lower()
    if region not in PILOT_ZONES:
        raise ValueError(f"unknown region '{region}'. Valid: {sorted(PILOT_ZONES)}")
    if variable not in VARIABLE_UNITS:
        raise ValueError(f"unknown variable '{variable}'. Valid: {sorted(VARIABLE_UNITS)}")
    if lead_hours not in LEAD_HOURS_LADDER:
        raise ValueError(f"unsupported lead_hours '{lead_hours}'. Valid: {LEAD_HOURS_LADDER}")

    if run_time is None:
        run_time = latest_run_time(region, variable, lead_hours)
    run_time = pd.Timestamp(run_time)

    rows = _fetch_rows(region, variable, lead_hours, run_time)
    if rows.empty:
        raise ValueError("no cached source forecasts for this run_time/region/variable/lead_hours")

    records = rows.to_dict(orient="records")
    quality = build_quality_mask(records)
    available_sources = quality.available_sources
    values = {r["model"]: r["forecast_value"] for r in records if r["model"] in available_sources}

    season = season_for_month(int(quality.valid_time.month))
    signal = _regime_signal(region, lead_hours, run_time)
    regime_probs = regime_engine.classify_regime(signal)
    dominant_regime = regime_engine.dominant_regime(regime_probs)
    stability = regime_engine.regime_stability(regime_probs)

    historical_skill_lookup = {
        m: skill_engine.historical_skill_score(
            state.skill_table, m, region, variable, lead_hours, season, dominant_regime
        )
        for m in available_sources
    }

    fallback_used = False
    if state.blend_model is not None and quality.quality_ok:
        weights = state.blend_model.predict_weights(
            region=region,
            variable=variable,
            lead_hours=lead_hours,
            season=season,
            regime=dominant_regime,
            available_sources=available_sources,
            historical_skill_lookup=historical_skill_lookup,
        )
    else:
        fallback_used = True
        weights = blend_engine.skill_weighted_average_weights(available_sources, historical_skill_lookup)

    raw_blend_value = blend_engine.weighted_blend(values, weights)

    corrector = state.bias_correctors.get((region, variable))
    bias_corrected_value = corrector.transform(raw_blend_value, dominant_regime) if corrector else raw_blend_value
    if variable in ("precipitation", "wind_speed"):
        bias_corrected_value = max(0.0, bias_corrected_value)
    final_value = bias_corrected_value

    disagreement = trust_engine.compute_disagreement(values)
    disagreement_norm = trust_engine.normalize_disagreement(disagreement, variable)
    data_quality = len(available_sources) / len(SOURCES)
    historical_skill_avg = float(sum(weights[m] * historical_skill_lookup[m] for m in available_sources))

    trust_breakdown = trust_engine.compute_trust_breakdown(
        historical_skill_avg=historical_skill_avg,
        disagreement_norm=disagreement_norm,
        lead_hours=lead_hours,
        regime_stability=stability,
        data_quality=data_quality,
    )

    bust_feature_row = {
        "lead_hours": lead_hours,
        "disagreement_norm": disagreement_norm,
        "historical_skill_avg": historical_skill_avg,
        "regime_stability": stability,
        "data_quality": data_quality,
    }
    if state.bust_model is not None:
        bust_probability = state.bust_model.predict_proba(bust_feature_row)
    else:
        # Deterministic heuristic fallback if the classifier hasn't been trained yet.
        bust_probability = float(np.clip(1 - trust_breakdown.trust_score, 0.02, 0.98))

    bust_flag = bust_probability > 0.5
    abstain = trust_engine.should_abstain(trust_breakdown.trust_score, bust_probability)

    explanation = _explain(state, bust_feature_row)

    provenance = {m: SOURCE_META[m]["provider"] for m in available_sources}
    provenance["regime_detector"] = "rule-based auditable classifier (engine.regime)"
    provenance["explanation_backend"] = explain_engine.explanation_backend_name()

    sources = [
        {
            "model": m,
            "forecast_value": round(float(values[m]), 3),
            "weight": round(float(weights[m]), 4),
            "historical_skill": round(float(historical_skill_lookup[m]), 4),
        }
        for m in available_sources
    ]

    return {
        "region": region,
        "region_name": PILOT_ZONES[region].name,
        "variable": variable,
        "unit": VARIABLE_UNITS[variable],
        "run_time": run_time,
        "valid_time": quality.valid_time,
        "lead_hours": lead_hours,
        "season": season,
        "regime": dominant_regime,
        "regime_probs": regime_probs,
        "sources": sources,
        "disagreement": round(float(disagreement), 3),
        "raw_blend_value": round(float(raw_blend_value), 3),
        "bias_corrected_value": round(float(bias_corrected_value), 3),
        "final_value": round(float(final_value), 3),
        "trust": trust_breakdown.__dict__,
        "bust_probability": round(float(bust_probability), 4),
        "bust_flag": bool(bust_flag),
        "abstain": bool(abstain),
        "explanation": explanation,
        "fallback_used": bool(fallback_used or not quality.quality_ok),
        "provenance": provenance,
    }


def _explain(state: _State, bust_feature_row: dict) -> list[dict]:
    if state.bust_model is not None and state.bust_baseline is not None:
        predict_fn = lambda df: state.bust_model.classifier.predict_proba(df)[:, _positive_index(state.bust_model)]
        drivers = explain_engine.explain_prediction(
            predict_fn=predict_fn,
            feature_row=pd.Series(bust_feature_row),
            baseline_row=state.bust_baseline,
            feature_columns=trust_engine.BUST_FEATURE_COLUMNS,
            higher_is_better=False,  # higher bust probability is "worse" -> flip direction wording
        )
        return [d.__dict__ for d in drivers]

    # Deterministic fallback if the bust model/baseline aren't available yet.
    ranked = sorted(
        [
            ("historical_skill_avg", 1 - bust_feature_row["historical_skill_avg"]),
            ("disagreement_norm", bust_feature_row["disagreement_norm"]),
            ("regime_stability", 1 - bust_feature_row["regime_stability"]),
            ("data_quality", 1 - bust_feature_row["data_quality"]),
        ],
        key=lambda kv: kv[1],
        reverse=True,
    )
    out = []
    for feature, score in ranked[:3]:
        if score <= 0.01:
            continue
        out.append(
            {
                "feature": feature,
                "contribution": round(float(score), 4),
                "direction": "decreases_trust",
                "detail": f"{explain_engine._readable(feature)} is currently a limiting factor",
            }
        )
    return out


def _positive_index(bust_model: trust_engine.BustModel) -> int:
    classes = list(bust_model.classifier.classes_)
    return classes.index(1) if 1 in classes else int(np.argmax(classes))


# ---------------------------------------------------------------------------
# Extreme-weather guidance
# ---------------------------------------------------------------------------
def extreme_guidance(region: str, lead_hours: int, run_time: pd.Timestamp | None = None) -> dict:
    state = _load_state()
    region = region.upper()
    guidance = []
    valid_time = None
    for variable in ("precipitation", "temperature", "wind_speed"):
        result = blend_forecast(region, variable, lead_hours, run_time)
        valid_time = result["valid_time"]
        threshold = EXTREME_THRESHOLDS[variable]
        raw_score = calib_engine.raw_exceedance_score(result["final_value"], threshold, max(result["disagreement"], 0.5))
        calibrator = state.exceedance_calibrators.get((region, variable))
        if calibrator:
            probability = calibrator.transform(raw_score)
            calibrated = True
        else:
            probability = raw_score
            calibrated = False
        guidance.append(
            {
                "variable": variable,
                "threshold": threshold,
                "unit": VARIABLE_UNITS[variable],
                "probability": round(float(probability), 4),
                "calibrated": calibrated,
            }
        )
    return {"region": region, "lead_hours": lead_hours, "valid_time": valid_time, "guidance": guidance}


# ---------------------------------------------------------------------------
# Weight map (context-driven, not tied to one live instance)
# ---------------------------------------------------------------------------
def weight_map(region: str, variable: str, season: str, regime: str) -> dict:
    state = _load_state()
    region = region.upper()
    variable = variable.lower()
    points = []
    for lead_hours in LEAD_HOURS_LADDER:
        historical_skill_lookup = {
            m: skill_engine.historical_skill_score(state.skill_table, m, region, variable, lead_hours, season, regime)
            for m in SOURCES
        }
        if state.blend_model is not None:
            weights = state.blend_model.predict_weights(
                region=region,
                variable=variable,
                lead_hours=lead_hours,
                season=season,
                regime=regime,
                available_sources=SOURCES,
                historical_skill_lookup=historical_skill_lookup,
            )
        else:
            weights = blend_engine.skill_weighted_average_weights(SOURCES, historical_skill_lookup)

        historical_skill_avg = float(sum(weights[m] * historical_skill_lookup[m] for m in SOURCES))
        trust_breakdown = trust_engine.compute_trust_breakdown(
            historical_skill_avg=historical_skill_avg,
            disagreement_norm=0.3,  # typical/expected value; this is a context map, not one live instance
            lead_hours=lead_hours,
            regime_stability=1.0,
            data_quality=1.0,
        )
        points.append(
            {
                "lead_hours": lead_hours,
                "weights": {m: round(float(w), 4) for m, w in weights.items()},
                "trust_score": trust_breakdown.trust_score,
            }
        )
    return {"region": region, "variable": variable, "regime": regime, "season": season, "points": points}
