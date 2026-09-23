from __future__ import annotations

from fastapi import APIRouter

from app.config import EXTREME_THRESHOLDS, LEAD_HOURS_LADDER, PILOT_ZONES, SOURCE_META, VARIABLE_UNITS

router = APIRouter(prefix="/api/v1", tags=["reference"])


@router.get("/regions")
def list_regions():
    return [
        {
            "code": z.code,
            "name": z.name,
            "lat": z.lat,
            "lon": z.lon,
            "emphasis": z.emphasis,
            "coastal": z.coastal,
        }
        for z in PILOT_ZONES.values()
    ]


@router.get("/sources")
def list_sources():
    return SOURCE_META


@router.get("/variables")
def list_variables():
    return [
        {"variable": v, "unit": unit, "extreme_threshold": EXTREME_THRESHOLDS[v]}
        for v, unit in VARIABLE_UNITS.items()
    ]


@router.get("/lead-times")
def list_lead_times():
    return LEAD_HOURS_LADDER
