"""
SkillBlend configuration.

Single source of truth for pilot zones, forecast sources, canonical
variables, lead-time ladder and extreme-event thresholds. Keeping this
centralised is what the blueprint calls the "minimal data contract" -
every module (ingestion, skill engine, blending, API) agrees on the
same vocabulary.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(os.environ.get("SKILLBLEND_ARTIFACTS_DIR", BASE_DIR / "artifacts"))
CACHE_DIR = ARTIFACTS_DIR / "cache"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPLAY_DIR = ARTIFACTS_DIR / "replay"
for _d in (ARTIFACTS_DIR, CACHE_DIR, MODELS_DIR, REPLAY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

HISTORICAL_PARQUET = CACHE_DIR / "historical_forecasts.parquet"
SKILL_TABLE_PARQUET = CACHE_DIR / "skill_table.parquet"
BLEND_MODEL_PATH = MODELS_DIR / "blend_model.joblib"
BUST_MODEL_PATH = MODELS_DIR / "bust_model.joblib"
CALIBRATORS_PATH = MODELS_DIR / "calibrators.joblib"
BLEND_BASELINE_PATH = MODELS_DIR / "blend_baseline.joblib"
BUST_BASELINE_PATH = MODELS_DIR / "bust_baseline.joblib"
VERIFICATION_SUMMARY_PATH = MODELS_DIR / "verification_summary.json"
REPLAY_INDEX_PATH = REPLAY_DIR / "replay_index.json"

# ---------------------------------------------------------------------------
# Forecast sources (Section 4 of the blueprint)
# ---------------------------------------------------------------------------
SOURCES = ["GFS", "IFS", "AIFS"]

SOURCE_META = {
    "GFS": {"kind": "physical_nwp", "provider": "NOAA NOMADS", "role": "Primary baseline source"},
    "IFS": {"kind": "physical_nwp", "provider": "ECMWF Open Data", "role": "High-value comparison source"},
    "AIFS": {"kind": "ai_nwp", "provider": "ECMWF Open Data", "role": "AI-native comparison source"},
}

REFERENCE_SOURCES = {
    "precipitation": "IMERG",
    "temperature": "ERA5",
    "wind_speed": "ERA5",
}

# ---------------------------------------------------------------------------
# Pilot zones (Section 12) - one representative grid point per zone for the
# MVP. This is a deliberate scope cut documented in the blueprint: prove the
# system deeply on 2-3 contrasting regions before claiming national coverage.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PilotZone:
    code: str
    name: str
    lat: float
    lon: float
    emphasis: str
    monsoon_sensitivity: float   # 0-1, how strongly SW monsoon drives rainfall
    wd_sensitivity: float        # 0-1, how strongly western disturbances matter
    coastal: bool


PILOT_ZONES: dict[str, PilotZone] = {
    "KWG": PilotZone(
        code="KWG",
        name="Kerala / Western Ghats",
        lat=10.5,
        lon=76.2,
        emphasis="Heavy rainfall / orographic terrain effects",
        monsoon_sensitivity=0.95,
        wd_sensitivity=0.05,
        coastal=True,
    ),
    "BOB": PilotZone(
        code="BOB",
        name="Bay of Bengal / East Coast",
        lat=17.7,
        lon=83.3,
        emphasis="Coastal systems, monsoon depressions, high wind",
        monsoon_sensitivity=0.75,
        wd_sensitivity=0.10,
        coastal=True,
    ),
    "IGP": PilotZone(
        code="IGP",
        name="Indo-Gangetic Plains / NW India",
        lat=28.6,
        lon=77.2,
        emphasis="Temperature extremes, western-disturbance regime shifts",
        monsoon_sensitivity=0.55,
        wd_sensitivity=0.85,
        coastal=False,
    ),
}

# ---------------------------------------------------------------------------
# Canonical variables + units + extreme thresholds (Section 14 / PS ask)
# ---------------------------------------------------------------------------
VARIABLES = ["precipitation", "temperature", "wind_speed"]

VARIABLE_UNITS = {
    "precipitation": "mm/24h",
    "temperature": "degC",
    "wind_speed": "m/s",
}

EXTREME_THRESHOLDS = {
    "precipitation": 50.0,   # mm/24h - PS-specified CSI evaluation threshold
    "temperature": 40.0,     # degC - heat-wave style guidance
    "wind_speed": 17.0,      # m/s ~ gale-force guidance
}

# CSI target from the blueprint's cover page: pre-registered, not claimed.
TARGET_RELATIVE_CSI_IMPROVEMENT = 0.05

# ---------------------------------------------------------------------------
# Lead-time ladder + seasons + regimes (Sections 6-7)
# ---------------------------------------------------------------------------
LEAD_HOURS_LADDER = [24, 48, 72, 96, 120, 144, 168]

SEASONS = ["winter", "pre_monsoon", "sw_monsoon", "post_monsoon"]


def season_for_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "pre_monsoon"
    if month in (6, 7, 8, 9):
        return "sw_monsoon"
    return "post_monsoon"


REGIMES = ["active_monsoon", "break_monsoon", "western_disturbance", "depression", "normal"]

EVENT_CLASSES = ["normal", "heavy", "extreme"]

# "Bust" = large forecast error, used to train the Forecast Bust Probability
# model (Section 10). Thresholds are on absolute error in native units.
BUST_ERROR_THRESHOLDS = {
    "precipitation": 30.0,
    "temperature": 4.0,
    "wind_speed": 6.0,
}

# ---------------------------------------------------------------------------
# Synthetic historical dataset sizing (Section 21: keep MVP light)
# ---------------------------------------------------------------------------
HISTORY_DAYS = int(os.environ.get("SKILLBLEND_HISTORY_DAYS", 730))
RANDOM_SEED = 26081  # PS number, for reproducibility of the synthetic study
