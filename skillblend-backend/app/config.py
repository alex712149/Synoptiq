"""
SkillBlend config: pilot zones, sources, variables, seasons, regimes.
Mirrors Section 12 (pilot zones) and Section 4 (sources) of the blueprint.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
DB_PATH = BASE_DIR / "artifacts" / "skillblend.db"

MODELS = ["GFS", "IFS", "AIFS"]  # candidate forecast sources

VARIABLES = ["precipitation", "temperature", "wind_speed"]

# Section 12: 2-3 pilot zones with contrasting weather behaviour.
PILOT_ZONES = {
    "kerala_western_ghats": {
        "label": "Kerala / Western Ghats",
        "lat_range": (8.3, 12.8),
        "lon_range": (74.9, 77.4),
        "emphasis": "Heavy rainfall / orographic terrain effects",
        "dominant_regimes": ["active_monsoon", "break_monsoon", "pre_monsoon"],
    },
    "bay_of_bengal_east_coast": {
        "label": "Bay of Bengal / East Coast",
        "lat_range": (12.5, 20.5),
        "lon_range": (80.0, 87.5),
        "emphasis": "Coastal systems, monsoon depressions, high wind",
        "dominant_regimes": ["depression", "active_monsoon", "normal"],
    },
    "indo_gangetic_plains": {
        "label": "Indo-Gangetic Plains / NW India",
        "lat_range": (24.0, 30.5),
        "lon_range": (75.0, 83.0),
        "emphasis": "Western disturbances, heatwaves, seasonal regime shift",
        "dominant_regimes": ["western_disturbance", "heatwave", "normal"],
    },
}

SEASONS = ["pre_monsoon", "sw_monsoon", "post_monsoon", "winter"]

REGIMES = [
    "active_monsoon",
    "break_monsoon",
    "depression",
    "western_disturbance",
    "heatwave",
    "pre_monsoon",
    "normal",
]

LEAD_HOURS = [24, 48, 72, 96, 120]

# Event thresholds used for extreme-weather guidance (Section 14 / PS ask).
THRESHOLDS = {
    "precipitation": {"heavy": 50.0, "unit": "mm/24h"},   # CSI @ 50mm is the pitch KPI
    "temperature": {"heatwave": 40.0, "unit": "deg_C"},
    "wind_speed": {"gale": 62.0, "unit": "km/h"},
}

# Pitch KPI target from the blueprint header (pre-registered, not a claimed result).
TARGET_RELATIVE_CSI_IMPROVEMENT = 0.05  # +5% relative CSI @ 50mm vs best single model

RANDOM_SEED = 42

# --- Strict time-based train/validation/test split (Section: leakage fix) ---
# Split is by TIME, computed once over the full sorted set of unique
# valid_times in the archive, so every row sharing a valid_time lands in the
# same split. Meta-model + bust classifier train on TRAIN only; calibration
# and bust-threshold selection use VALIDATION (frozen meta-model, not
# refit); TEST is touched only by scripts/evaluate_blend.py, after every
# learned artifact (skill table, meta-model, calibration, bust classifier)
# has been frozen.
TRAIN_FRACTION = 0.60
VAL_FRACTION = 0.15
TEST_FRACTION = 0.25  # 1.0 - TRAIN_FRACTION - VAL_FRACTION
