"""
Normalization layer (blueprint Section 5): "the invisible hard part".

Responsibilities implemented here:
  - Enforce one canonical schema/unit set for every source.
  - Keep run_time, valid_time and lead_hours as separate, internally
    consistent fields (never re-derive one silently from the others).
  - Attach a quality mask so a missing/stale source cannot silently
    enter the blending model - callers get an explicit `available`
    flag and the blend falls back gracefully (see engine.blend).
  - Timestamps are handled internally in UTC only; any IST conversion
    is a presentation-layer concern (left to a future frontend).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.config import SOURCES, VARIABLE_UNITS


class SchemaValidationError(ValueError):
    pass


REQUIRED_COLUMNS = {
    "model",
    "run_time",
    "valid_time",
    "lead_hours",
    "variable",
    "region",
    "lat",
    "lon",
    "forecast_value",
    "unit",
}


def validate_canonical_frame(df: pd.DataFrame) -> None:
    """Raise if `df` does not satisfy the minimal data contract."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SchemaValidationError(f"missing canonical columns: {sorted(missing)}")

    # Time alignment invariant (also unit-tested): valid_time - run_time == lead_hours
    delta_hours = (df["valid_time"] - df["run_time"]).dt.total_seconds() / 3600.0
    bad = ~delta_hours.round(6).eq(df["lead_hours"].astype(float))
    if bad.any():
        raise SchemaValidationError(
            f"{bad.sum()} rows violate valid_time - run_time == lead_hours"
        )

    # Unit consistency
    for variable, unit in VARIABLE_UNITS.items():
        subset = df[df["variable"] == variable]
        if not subset.empty and not (subset["unit"] == unit).all():
            raise SchemaValidationError(f"inconsistent unit for variable={variable}")


@dataclass
class ContextBundle:
    """Per-instance snapshot passed downstream to skill/regime/blend."""

    region: str
    variable: str
    lead_hours: int
    run_time: pd.Timestamp
    valid_time: pd.Timestamp
    available_sources: list[str]
    missing_sources: list[str]
    quality_ok: bool


def build_quality_mask(
    available_records: list[dict],
    expected_sources: list[str] | None = None,
) -> ContextBundle:
    """
    Given the raw per-source forecast rows returned for one
    (region, variable, run_time, lead_hours) instance, work out which
    sources are actually present and flag data-quality issues so they
    never enter the blend silently (Section 5, bullet 5; Section 21
    "Fallback" test).
    """
    expected_sources = expected_sources or SOURCES
    if not available_records:
        raise SchemaValidationError("no forecast records supplied to quality mask")

    first = available_records[0]
    present = {r["model"] for r in available_records}
    missing = [s for s in expected_sources if s not in present]

    return ContextBundle(
        region=first["region"],
        variable=first["variable"],
        lead_hours=int(first["lead_hours"]),
        run_time=pd.Timestamp(first["run_time"]),
        valid_time=pd.Timestamp(first["valid_time"]),
        available_sources=sorted(present),
        missing_sources=missing,
        quality_ok=len(present) >= 2,  # need at least 2 sources for a meaningful blend
    )


def to_canonical_records(df: pd.DataFrame) -> list[dict]:
    """Convert a normalized frame back into a list[dict] canonical rows."""
    validate_canonical_frame(df)
    return df[list(REQUIRED_COLUMNS)].to_dict(orient="records")
