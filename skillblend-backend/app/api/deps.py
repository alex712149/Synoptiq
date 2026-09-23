"""Shared FastAPI dependencies / error translation."""
from __future__ import annotations

from fastapi import HTTPException

from app.services.pipeline import PipelineNotReady


def run_or_503(fn, *args, **kwargs):
    """Translate pipeline errors into clean HTTP responses."""
    try:
        return fn(*args, **kwargs)
    except PipelineNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
