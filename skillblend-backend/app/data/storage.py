"""
Storage layer.

Production design (blueprint Sections 5 & 16): chunked Zarr/NetCDF for
gridded arrays, PostgreSQL/PostGIS for metadata, events and lookups.
This prototype runs single-machine and offline, so it uses a much
lighter local cache that preserves the same *interface* the rest of
the app relies on (`load_historical()`, `save_historical()`, ...),
which is the piece that actually matters for swapping storage engines
later without touching the skill/blend/trust logic.

Serialization note: pyarrow/fastparquet are not guaranteed to be
present in every environment this prototype might run in, so this
module prefers parquet when a parquet engine is importable and
transparently falls back to a joblib pickle otherwise. Either way the
public functions below always return a plain pandas.DataFrame.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from app.config import HISTORICAL_PARQUET, REPLAY_INDEX_PATH, SKILL_TABLE_PARQUET


def _has_parquet_engine() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401

            return True
        except ImportError:
            return False


_PARQUET_OK = _has_parquet_engine()


def _pkl_path(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(".pkl")


def save_frame(df: pd.DataFrame, path: Path) -> Path:
    if _PARQUET_OK:
        df.to_parquet(path, index=False)
        return path
    pkl_path = _pkl_path(path)
    joblib.dump(df, pkl_path)
    return pkl_path


def load_frame(path: Path) -> pd.DataFrame:
    if path.exists() and _PARQUET_OK:
        return pd.read_parquet(path)
    pkl_path = _pkl_path(path)
    if pkl_path.exists():
        return joblib.load(pkl_path)
    raise FileNotFoundError(
        f"no cached data at {path} or {pkl_path} - run scripts/seed_and_train.py first"
    )


def save_historical(df: pd.DataFrame) -> Path:
    return save_frame(df, HISTORICAL_PARQUET)


def load_historical() -> pd.DataFrame:
    return load_frame(HISTORICAL_PARQUET)


def save_skill_table(df: pd.DataFrame) -> Path:
    return save_frame(df, SKILL_TABLE_PARQUET)


def load_skill_table() -> pd.DataFrame:
    return load_frame(SKILL_TABLE_PARQUET)


def save_replay_index(events: list[dict]) -> None:
    REPLAY_INDEX_PATH.write_text(json.dumps(events, default=str, indent=2))


def load_replay_index() -> list[dict]:
    if not REPLAY_INDEX_PATH.exists():
        return []
    return json.loads(REPLAY_INDEX_PATH.read_text())


def save_replay_event(event_id: str, payload: dict, replay_dir: Path) -> None:
    (replay_dir / f"{event_id}.json").write_text(json.dumps(payload, default=str, indent=2))


def load_replay_event(event_id: str, replay_dir: Path) -> dict:
    path = replay_dir / f"{event_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no replay event cached for {event_id}")
    return json.loads(path.read_text())
