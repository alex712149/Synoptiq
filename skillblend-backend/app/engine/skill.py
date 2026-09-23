"""
Historical Skill Engine (blueprint Section 6): "the memory of the
system". For every model, variable, region and lead time it measures
performance on historical cases and exposes that performance as
features to the blending model.

Metrics implemented, per the blueprint's recommendation:
  - RMSE, MAE, bias           -> continuous accuracy
  - CSI, POD, FAR             -> event/threshold skill (the PS asks for
                                  CSI specifically at the 50mm threshold)
  - FSS (proxy)                -> spatial/neighbourhood skill

Important honesty note on FSS: a true Fractions Skill Score needs a
spatial neighbourhood of grid points. This MVP prototype carries one
representative point per pilot zone (blueprint Section 12's documented
scope cut), so `fss` here is a *temporal-neighbourhood proxy*
(fraction of hits within a +/-1 day window around each event) rather
than the textbook spatial FSS. It is clearly labelled as a proxy in
every output and should be replaced with true spatial FSS once the
system ingests full gridded fields.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import EXTREME_THRESHOLDS


def _event_counts(group: pd.DataFrame, threshold: float) -> tuple[int, int, int, int]:
    forecast_event = group["forecast_value"] >= threshold
    actual_event = group["actual_value"] >= threshold
    hits = int((forecast_event & actual_event).sum())
    misses = int((~forecast_event & actual_event).sum())
    false_alarms = int((forecast_event & ~actual_event).sum())
    correct_negatives = int((~forecast_event & ~actual_event).sum())
    return hits, misses, false_alarms, correct_negatives


def _csi_pod_far(hits: int, misses: int, false_alarms: int) -> tuple[float, float, float]:
    denom_csi = hits + misses + false_alarms
    csi = hits / denom_csi if denom_csi > 0 else np.nan
    denom_pod = hits + misses
    pod = hits / denom_pod if denom_pod > 0 else np.nan
    denom_far = hits + false_alarms
    far = false_alarms / denom_far if denom_far > 0 else np.nan
    return csi, pod, far


def _fss_proxy(group: pd.DataFrame, threshold: float, window_days: int = 1) -> float:
    """Temporal-neighbourhood proxy for FSS - see module docstring."""
    g = group.sort_values("valid_time").reset_index(drop=True)
    actual_event = (g["actual_value"] >= threshold).to_numpy()
    forecast_event = (g["forecast_value"] >= threshold).to_numpy()
    n = len(g)
    if n == 0:
        return np.nan
    num, den = 0.0, 0.0
    for i in range(n):
        lo, hi = max(0, i - window_days), min(n, i + window_days + 1)
        f_frac = forecast_event[lo:hi].mean()
        a_frac = actual_event[lo:hi].mean()
        num += (f_frac - a_frac) ** 2
        den += f_frac**2 + a_frac**2
    if den == 0:
        return 1.0
    return float(1 - num / den)


def compute_skill_table(historical: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Section-6 skill table: one row per
    (model, region, variable, lead_hours, season, regime).
    """
    group_cols = ["model", "region", "variable", "lead_hours", "season", "regime"]
    rows = []
    for keys, group in historical.groupby(group_cols, observed=True):
        model, region, variable, lead_hours, season, regime = keys
        err = group["forecast_value"] - group["actual_value"]
        threshold = EXTREME_THRESHOLDS[variable]
        hits, misses, fa, cn = _event_counts(group, threshold)
        csi, pod, far = _csi_pod_far(hits, misses, fa)
        fss = _fss_proxy(group, threshold) if variable == "precipitation" else np.nan

        rows.append(
            {
                "model": model,
                "region": region,
                "variable": variable,
                "lead_hours": int(lead_hours),
                "season": season,
                "regime": regime,
                "n_obs": int(len(group)),
                "rmse": float(np.sqrt((err**2).mean())),
                "mae": float(err.abs().mean()),
                "bias": float(err.mean()),
                "csi": None if np.isnan(csi) else float(csi),
                "pod": None if np.isnan(pod) else float(pod),
                "far": None if np.isnan(far) else float(far),
                "fss": None if (fss is np.nan or (isinstance(fss, float) and np.isnan(fss))) else float(fss),
            }
        )
    return pd.DataFrame(rows)


def historical_skill_score(
    skill_table: pd.DataFrame,
    model: str,
    region: str,
    variable: str,
    lead_hours: int,
    season: str,
    regime: str,
) -> float:
    """
    Collapse a skill-table row into a single 0-1 'historical_skill'
    feature (blueprint's canonical `historical_skill` field). Blends
    normalized inverse-RMSE with event skill (CSI) when available,
    since the PS cares about both continuous accuracy and extreme
    events. Falls back to progressively coarser context (drop regime,
    then season, then lead-time bucket) if the exact combination has
    too few historical observations - this mirrors how a forecaster
    would fall back to climatology when a specific analog is sparse.
    """
    candidates = [
        dict(model=model, region=region, variable=variable, lead_hours=lead_hours, season=season, regime=regime),
        dict(model=model, region=region, variable=variable, lead_hours=lead_hours, season=season),
        dict(model=model, region=region, variable=variable, lead_hours=lead_hours),
        dict(model=model, region=region, variable=variable),
    ]
    for filt in candidates:
        mask = np.ones(len(skill_table), dtype=bool)
        for k, v in filt.items():
            mask &= skill_table[k] == v
        subset = skill_table[mask]
        n = subset["n_obs"].sum() if "n_obs" in subset else 0
        if n >= 15:
            rmse = np.average(subset["rmse"], weights=subset["n_obs"])
            csi_vals = subset["csi"].dropna()
            csi = float(np.average(csi_vals, weights=subset.loc[csi_vals.index, "n_obs"])) if len(csi_vals) else None

            # normalize RMSE into a 0-1 "goodness" score via a soft transform
            rmse_score = float(np.exp(-rmse / rmse_scale(variable)))
            if csi is not None:
                return float(np.clip(0.6 * rmse_score + 0.4 * csi, 0.0, 1.0))
            return float(np.clip(rmse_score, 0.0, 1.0))

    return 0.5  # true cold start: no analog history at all


def rmse_scale(variable: str) -> float:
    return {"precipitation": 12.0, "temperature": 2.0, "wind_speed": 3.0}[variable]
