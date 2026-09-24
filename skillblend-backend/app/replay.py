"""
Offline Historical Replay Mode (Section 11 "showstopper": Counterfactual Bust
Replay; Section 17 demo script; Section 18 guardrail: "pre-cache
representative cases, Historical Replay Mode"). This is the DEFAULT judging
path — every replay event is fully precomputed and stored, so opening it
needs zero live model inference and is immune to any API/network flakiness
on demo day.
"""
from __future__ import annotations
import json
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session

from app.models_db import ForecastRow, GroundTruthRow, ReplayEvent
from app.pipeline import run_blend_pipeline

N_EVENTS_PER_REGION = 3


def _pick_candidate_valid_times(db: Session, region: str, variable: str, lead_hours: int) -> pd.DataFrame:
    fdf = pd.read_sql(
        db.query(ForecastRow).filter(ForecastRow.region == region, ForecastRow.variable == variable,
                                      ForecastRow.lead_hours == lead_hours).statement, db.bind)
    gdf = pd.read_sql(
        db.query(GroundTruthRow).filter(GroundTruthRow.region == region,
                                         GroundTruthRow.variable == variable).statement, db.bind)
    merged = fdf.merge(gdf[["region", "valid_time", "variable", "observed_value"]],
                        on=["region", "valid_time", "variable"], how="inner")
    if merged.empty:
        return merged
    agg = merged.groupby("valid_time").agg(
        observed_value=("observed_value", "first"),
        naive_avg=("forecast_value", "mean"),
        spread=("forecast_value", "std"),
    ).reset_index()
    agg["naive_error"] = (agg["naive_avg"] - agg["observed_value"]).abs()
    return agg


def precompute_replay_cache(db: Session, region: str, variable: str = "precipitation",
                             lead_hours: int = 72) -> list[str]:
    agg = _pick_candidate_valid_times(db, region, variable, lead_hours)
    if agg.empty:
        return []

    ids_written = []
    # One clear "bust" case (naive average would have been badly wrong)...
    bust_row = agg.loc[agg["naive_error"].idxmax()]
    # ...one clean high-confidence "success" case...
    calm_row = agg.loc[agg["spread"].idxmin()]
    # ...and one high-disagreement case in between.
    mid_row = agg.iloc[(agg["spread"] - agg["spread"].median()).abs().idxmin()]

    for row, label, severity in [
        (bust_row, "Naive-average bust event", "high"),
        (mid_row, "High model disagreement", "medium"),
        (calm_row, "Calm, high-confidence case", "low"),
    ]:
        vt = row["valid_time"]
        if isinstance(vt, str):
            vt = pd.to_datetime(vt)
        try:
            result = run_blend_pipeline(db, region, variable, vt.to_pydatetime() if hasattr(vt, "to_pydatetime") else vt, lead_hours)
        except ValueError:
            continue
        event_id = f"{region}_{variable}_{lead_hours}h_{vt.strftime('%Y%m%dT%H%M')}"
        payload = json.loads(result.model_dump_json())
        payload["naive_average"] = round(float(row["naive_avg"]), 2)
        payload["naive_average_error"] = round(float(row["naive_error"]), 2)
        payload["observed_value"] = round(float(row["observed_value"]), 2)

        existing = db.get(ReplayEvent, event_id)
        if existing:
            db.delete(existing)
            db.flush()
        db.add(ReplayEvent(
            event_id=event_id, region=region, variable=variable, valid_time=vt,
            label=label, severity=severity, payload=payload,
        ))
        ids_written.append(event_id)
    db.commit()
    return ids_written
