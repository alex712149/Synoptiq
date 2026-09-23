from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.config import VERIFICATION_SUMMARY_PATH
from app.data import storage

router = APIRouter(prefix="/api/v1/skill", tags=["skill-and-verification"])


@router.get("/scorecard")
def get_scorecard(
    region: Optional[str] = Query(None),
    variable: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    season: Optional[str] = Query(None),
    regime: Optional[str] = Query(None),
    lead_hours: Optional[int] = Query(None),
    limit: int = Query(200, le=2000),
):
    """
    PS expected outcome #3 ('Improved forecast skill compared with
    individual sources') starts here: the raw Historical Skill Engine
    table (RMSE/MAE/bias/CSI/POD/FAR/FSS-proxy) per model x region x
    variable x lead time x season x regime.
    """
    try:
        table = storage.load_skill_table()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    df = table
    if region:
        df = df[df["region"] == region.upper()]
    if variable:
        df = df[df["variable"] == variable.lower()]
    if model:
        df = df[df["model"] == model.upper()]
    if season:
        df = df[df["season"] == season]
    if regime:
        df = df[df["regime"] == regime]
    if lead_hours:
        df = df[df["lead_hours"] == lead_hours]

    return df.head(limit).to_dict(orient="records")


@router.get("/verification")
def get_verification(
    region: Optional[str] = Query(None),
    variable: Optional[str] = Query(None),
):
    """
    The blueprint's headline KPI: SkillBlend vs the best single model's
    CSI at the 50mm/heat/gale threshold on a strictly held-out temporal
    split, plus whether the pre-registered +5% relative-CSI target was
    met. Numbers are produced once by scripts/seed_and_train.py on a
    holdout it never trained on - never fabricated at request time.
    """
    if not VERIFICATION_SUMMARY_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="No verification summary cached yet. Run scripts/seed_and_train.py first.",
        )
    summary = json.loads(VERIFICATION_SUMMARY_PATH.read_text())
    if region:
        summary = [s for s in summary if s["region"] == region.upper()]
    if variable:
        summary = [s for s in summary if s["variable"] == variable.lower()]
    return summary
