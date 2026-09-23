"""
SkillBlend API - FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --reload --port 8000

Before first use, seed the synthetic historical dataset and train the
models:
    python -m scripts.seed_and_train

Interactive docs at /docs once running.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    routes_forecast,
    routes_health,
    routes_regions,
    routes_replay,
    routes_skill,
    routes_weights,
)

app = FastAPI(
    title="SkillBlend API",
    description=(
        "Hybrid AI-NWP Multi-Model Forecast Blending System - PS 26081 "
        "(NCMRWF, Ministry of Earth Sciences). Learns which forecast "
        "source to trust for a given place, lead time, season and "
        "weather regime, then blends, bias-corrects, calibrates and "
        "explains a single forecast product."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_health.router)
app.include_router(routes_regions.router)
app.include_router(routes_forecast.router)
app.include_router(routes_weights.router)
app.include_router(routes_skill.router)
app.include_router(routes_replay.router)


@app.get("/")
def root():
    return {
        "name": "SkillBlend API",
        "problem_statement": "PS 26081 - Hybrid AI-NWP Multi-Model Forecast Blending System",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
