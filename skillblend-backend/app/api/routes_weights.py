from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.deps import run_or_503
from app.config import LEAD_HOURS_LADDER, PILOT_ZONES, REGIMES, SEASONS, VARIABLES
from app.services import pipeline

router = APIRouter(prefix="/api/v1", tags=["weights-and-extremes"])


@router.get("/weights/map")
def get_weight_map(
    region: str = Query(..., description=f"Pilot zone code: {sorted(PILOT_ZONES)}"),
    variable: str = Query(..., description=f"One of: {VARIABLES}"),
    season: str = Query(..., description=f"One of: {SEASONS}"),
    regime: str = Query(..., description=f"One of: {REGIMES}"),
):
    """
    PS expected outcome #2: 'Model-weight maps showing how trust changes
    by location and lead time.' This endpoint sweeps the full lead-time
    ladder for a given region/variable/season/regime combination.
    """
    return run_or_503(pipeline.weight_map, region, variable, season, regime)


@router.get("/extreme/guidance")
def get_extreme_guidance(
    region: str = Query(..., description=f"Pilot zone code: {sorted(PILOT_ZONES)}"),
    lead_hours: int = Query(..., description=f"One of: {LEAD_HOURS_LADDER}"),
):
    """
    PS expected outcome #4: extreme-weather guidance for heavy rainfall,
    heat and high-wind conditions, expressed as calibrated
    threshold-exceedance probabilities from the same blended output
    (Section 9's lightweight MVP scope - no separate deep heads).
    """
    return run_or_503(pipeline.extreme_guidance, region, lead_hours)
