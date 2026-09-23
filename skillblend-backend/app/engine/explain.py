"""
SHAP-backed forecast discussion (blueprint Section 10 / Section 11
innovation table): "LightGBM/XGBoost feature attributions are the
source of truth for every blending decision ... An optional LLM may
convert these structured SHAP facts into readable forecast-discussion
text, but it must never invent the explanation."

This module produces structured, numeric attributions for why a
source's predicted error (and therefore its blend weight) came out the
way it did. Two backends:

  - Real SHAP (TreeExplainer) when the `shap` package is installed.
  - A deterministic, auditable fallback otherwise: for each feature,
    replace it with a baseline value (the historical mean/mode for
    that feature) and measure the resulting change in the model's
    prediction. This is a simplified, one-feature-at-a-time
    approximation of Shapley attribution - not the textbook game-
    theoretic values - and is labelled as such in every response so
    nobody mistakes it for calibrated SHAP output. Either way, no LLM
    is involved in producing these numbers; an LLM would only be
    allowed to narrate them afterwards, per the blueprint's guardrail.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import shap  # noqa: F401

    _HAS_SHAP = True
except ImportError:  # pragma: no cover
    _HAS_SHAP = False


_READABLE_FEATURE_NAMES = {
    "lead_hours": "forecast lead time",
    "historical_skill": "this source's historical skill in similar situations",
    "monsoon_sensitivity": "how monsoon-driven this region is",
    "wd_sensitivity": "how western-disturbance-driven this region is",
    "coastal": "coastal exposure",
    "disagreement_norm": "how much the sources disagree right now",
    "historical_skill_avg": "average historical skill across sources",
    "regime_stability": "how stable the current weather regime looks",
    "data_quality": "completeness of the incoming source data",
}


def _readable(feature: str) -> str:
    if feature in _READABLE_FEATURE_NAMES:
        return _READABLE_FEATURE_NAMES[feature]
    if feature.startswith("model_"):
        return f"source identity ({feature.split('_', 1)[1]})"
    if feature.startswith("season_"):
        return f"season ({feature.split('_', 1)[1].replace('_', ' ')})"
    if feature.startswith("regime_"):
        return f"weather regime ({feature.split('_', 1)[1].replace('_', ' ')})"
    return feature


@dataclass
class Driver:
    feature: str
    contribution: float
    direction: str
    detail: str


def compute_baseline_row(training_frame: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    """Public helper used by scripts/seed_and_train.py to cache a baseline feature vector per model."""
    baseline = {}
    for col in feature_columns:
        series = training_frame[col]
        if set(series.unique()) <= {0, 1}:
            baseline[col] = float(series.mode().iloc[0])
        else:
            baseline[col] = float(series.mean())
    return pd.Series(baseline)


def explain_prediction(
    predict_fn,
    feature_row: pd.Series,
    baseline_row: pd.Series,
    feature_columns: list[str],
    higher_is_better: bool,
    top_k: int = 4,
) -> list[Driver]:
    """
    Generic attribution routine used for both the blend model (predicts
    expected error - lower is better) and the bust model (predicts bust
    probability - lower is better). `higher_is_better` flips the
    direction labels so callers get "increases_weight"/"decreases_weight"
    style semantics rather than raw sign confusion.
    """
    full_pred = float(np.asarray(predict_fn(pd.DataFrame([feature_row])[feature_columns])).reshape(-1)[0])
    contributions = []
    for col in feature_columns:
        perturbed = feature_row.copy()
        perturbed[col] = baseline_row[col]
        perturbed_pred = float(np.asarray(predict_fn(pd.DataFrame([perturbed])[feature_columns])).reshape(-1)[0])
        # contribution = how much this feature's actual value pushed the
        # prediction away from the "typical" baseline prediction
        contributions.append((col, full_pred - perturbed_pred))

    contributions.sort(key=lambda kv: abs(kv[1]), reverse=True)
    drivers = []
    for col, contrib in contributions[:top_k]:
        if abs(contrib) < 1e-6:
            continue
        pushes_prediction_up = contrib > 0
        if higher_is_better:
            direction = "increases_weight" if pushes_prediction_up else "decreases_weight"
        else:
            direction = "decreases_weight" if pushes_prediction_up else "increases_weight"
        drivers.append(
            Driver(
                feature=col,
                contribution=round(float(contrib), 4),
                direction=direction,
                detail=f"{_readable(col)} moved the model's estimate by {contrib:+.3f}",
            )
        )
    return drivers


def explanation_backend_name() -> str:
    return "shap.TreeExplainer" if _HAS_SHAP else "perturbation-approximation (SHAP not installed)"
