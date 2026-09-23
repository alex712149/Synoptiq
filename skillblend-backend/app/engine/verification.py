"""
Verification (blueprint Section 21 / cover-page KPI):

"Headline proof point to report after validation: SkillBlend beats the
best single model by X% CSI at the 50 mm threshold on the held-out
test set." X is measured, never fabricated. The engineering target is
+5% relative CSI improvement (TARGET_RELATIVE_CSI_IMPROVEMENT).

This module computes that comparison on a temporal holdout split (no
random shuffling of weather sequences - Section 18's leakage
guardrail) and is what backs the /skill/verification API route and
the scripts/seed_and_train.py console summary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import EXTREME_THRESHOLDS, TARGET_RELATIVE_CSI_IMPROVEMENT
from app.engine.skill import _csi_pod_far, _event_counts


def csi_for_values(forecast_values: np.ndarray, actual_values: np.ndarray, threshold: float) -> float:
    df = pd.DataFrame({"forecast_value": forecast_values, "actual_value": actual_values})
    hits, misses, fa, _ = _event_counts(df, threshold)
    csi, _, _ = _csi_pod_far(hits, misses, fa)
    return float(csi) if not np.isnan(csi) else float("nan")


def verify_region_variable(
    holdout_blend: pd.DataFrame,
    holdout_single_model: dict[str, pd.DataFrame],
    region: str,
    variable: str,
) -> dict:
    """
    `holdout_blend` needs columns [blend_value, actual_value] for the
    held-out period. `holdout_single_model` maps model name -> frame
    with [forecast_value, actual_value] for the same held-out rows.
    """
    threshold = EXTREME_THRESHOLDS[variable]

    blend_csi = csi_for_values(holdout_blend["blend_value"].to_numpy(), holdout_blend["actual_value"].to_numpy(), threshold)

    single_scores = {}
    for model, frame in holdout_single_model.items():
        single_scores[model] = csi_for_values(frame["forecast_value"].to_numpy(), frame["actual_value"].to_numpy(), threshold)

    best_model = max(single_scores, key=lambda m: (single_scores[m] if not np.isnan(single_scores[m]) else -1))
    best_csi = single_scores[best_model]

    if not best_csi or np.isnan(best_csi) or best_csi == 0:
        relative_improvement = float("nan")
    else:
        relative_improvement = (blend_csi - best_csi) / best_csi

    return {
        "region": region,
        "variable": variable,
        "threshold": threshold,
        "best_single_model": best_model,
        "best_single_model_csi": None if np.isnan(best_csi) else round(float(best_csi), 4),
        "skillblend_csi": None if np.isnan(blend_csi) else round(float(blend_csi), 4),
        "relative_csi_improvement": None if np.isnan(relative_improvement) else round(float(relative_improvement), 4),
        "meets_target": bool(not np.isnan(relative_improvement) and relative_improvement >= TARGET_RELATIVE_CSI_IMPROVEMENT),
        "target_relative_csi_improvement": TARGET_RELATIVE_CSI_IMPROVEMENT,
    }
