"""
scripts/smoke_test.py

Exercises app.services.pipeline directly, without going through
FastAPI/uvicorn. Useful to sanity-check the core engine logic in any
environment, even one where fastapi/pydantic/lightgbm/shap are not
installed (the engine modules only require pandas/numpy/scikit-learn
plus joblib; FastAPI is only imported by app.main and app.api.*).

Run:
    python -m scripts.smoke_test
"""
from __future__ import annotations

import json

from app.config import LEAD_HOURS_LADDER, PILOT_ZONES, REGIMES, SEASONS, VARIABLES
from app.data import storage
from app.services import pipeline


def _json(obj) -> str:
    return json.dumps(obj, default=str, indent=2)


def main() -> None:
    print("=== health-ish check ===")
    for region in PILOT_ZONES:
        for variable in VARIABLES:
            rt = pipeline.latest_run_time(region, variable, LEAD_HOURS_LADDER[0])
            print(f"{region}/{variable}: latest cached cycle = {rt}")

    print("\n=== blend_forecast example ===")
    result = pipeline.blend_forecast("KWG", "precipitation", 72)
    print(_json(result))

    print("\n=== extreme_guidance example ===")
    guidance = pipeline.extreme_guidance("BOB", 48)
    print(_json(guidance))

    print("\n=== weight_map example ===")
    wmap = pipeline.weight_map("IGP", "temperature", season="winter", regime="western_disturbance")
    print(_json(wmap))

    print("\n=== replay events ===")
    events = storage.load_replay_index()
    print(_json(events))
    if events:
        from app.config import REPLAY_DIR

        detail = storage.load_replay_event(events[0]["event_id"], REPLAY_DIR)
        print("\n=== first replay event detail ===")
        print(_json(detail))

    print("\nsmoke test OK")


if __name__ == "__main__":
    main()
