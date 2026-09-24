from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import Base, engine, SessionLocal
from app.models_db import ForecastRow
from app.routers import regions, forecast, verification, replay

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="SkillBlend API",
    description=(
        "Hybrid AI-NWP Multi-Model Forecast Blending System — SIH PS 26081 "
        "(NCMRWF / Ministry of Earth Sciences). Learns which forecast source "
        "to trust by region, lead time, season and weather regime, then "
        "blends, bias-corrects, calibrates and explains the result."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(regions.router)
app.include_router(forecast.router)
app.include_router(verification.router)
app.include_router(replay.router)


@app.get("/health")
def health():
    db = SessionLocal()
    try:
        n = db.query(ForecastRow).count()
    finally:
        db.close()
    return {
        "status": "ok",
        "forecast_rows_cached": n,
        "note": "Run `bash scripts/bootstrap.sh` once if this is 0 — it seeds "
                "synthetic data, trains the meta-model, and precomputes the "
                "offline replay cache used as the default demo path.",
    }


@app.get("/")
def root():
    return {
        "name": "SkillBlend API",
        "docs": "/docs",
        "quickstart": [
            "GET /regions",
            "GET /replay/events  (offline, demo-day-safe — start here)",
            "GET /forecast/available?region=kerala_western_ghats",
            "GET /forecast/blend?region=...&variable=precipitation&valid_time=...&lead_hours=72",
            "GET /verification/scorecard?region=...",
        ],
    }
