from __future__ import annotations

from fastapi import APIRouter

from app.config import BLEND_MODEL_PATH, BUST_MODEL_PATH, HISTORICAL_PARQUET, REPLAY_INDEX_PATH
from app.data.storage import _pkl_path

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
def health():
    historical_ready = HISTORICAL_PARQUET.exists() or _pkl_path(HISTORICAL_PARQUET).exists()
    return {
        "status": "ok" if historical_ready else "not_seeded",
        "historical_data_cached": historical_ready,
        "blend_model_trained": BLEND_MODEL_PATH.exists(),
        "bust_model_trained": BUST_MODEL_PATH.exists(),
        "replay_events_cached": REPLAY_INDEX_PATH.exists(),
    }
