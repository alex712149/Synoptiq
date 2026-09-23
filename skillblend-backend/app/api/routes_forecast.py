from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query

from app.api.deps import run_or_503
from app.config import LEAD_HOURS_LADDER, PILOT_ZONES, VARIABLES
from app.schemas import BlendRequest
from app.services import pipeline

router = APIRouter(prefix="/api/v1/forecast", tags=["forecast"])


@router.get("/blend")
def get_blend(
    region: str = Query(..., description=f"Pilot zone code: {sorted(PILOT_ZONES)}"),
    variable: str = Query(..., description=f"One of: {VARIABLES}"),
    lead_hours: int = Query(..., description=f"One of: {LEAD_HOURS_LADDER}"),
    run_time: Optional[datetime] = Query(None, description="Forecast cycle; defaults to latest cached cycle"),
):
    """
    The single most important endpoint: returns the dynamically blended
    forecast (PS's #1 expected outcome) together with per-source
    contributions, trust breakdown, bust probability and a structured
    explanation - i.e. everything the judge-facing demo walks through.
    """
    return run_or_503(pipeline.blend_forecast, region, variable, lead_hours, run_time)


@router.post("/blend")
def post_blend(request: BlendRequest):
    """Same as GET /blend, as a JSON body - convenient for non-browser clients/dashboards."""
    return run_or_503(pipeline.blend_forecast, request.region, request.variable, request.lead_hours, request.run_time)


@router.get("/latest-cycle")
def get_latest_cycle(
    region: str = Query(...),
    variable: str = Query(...),
    lead_hours: int = Query(...),
):
    ts = run_or_503(pipeline.latest_run_time, region.upper(), variable.lower(), lead_hours)
    return {"region": region.upper(), "variable": variable.lower(), "lead_hours": lead_hours, "run_time": ts}
