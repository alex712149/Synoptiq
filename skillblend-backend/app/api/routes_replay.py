from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config import REPLAY_DIR
from app.data import storage

router = APIRouter(prefix="/api/v1/replay", tags=["replay"])


@router.get("/events")
def list_replay_events():
    """
    'DEMO DEFAULT: Open Historical Replay Mode first. The judging flow
    is fully offline-safe.' This lists the precomputed Counterfactual
    Bust Replay cases (blueprint Section 11's 'Showstopper' feature).
    """
    events = storage.load_replay_index()
    if not events:
        raise HTTPException(status_code=503, detail="No replay events cached. Run scripts/seed_and_train.py first.")
    return events


@router.get("/events/{event_id}")
def get_replay_event(event_id: str):
    """
    Full counterfactual: raw models vs a naive average vs a single-model
    choice vs SkillBlend, against the reference/verification value -
    exactly the 'Trigger the Forecast Bust view' step of the demo script.
    """
    try:
        return storage.load_replay_event(event_id, REPLAY_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
