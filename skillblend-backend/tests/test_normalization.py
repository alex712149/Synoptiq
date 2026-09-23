"""
Covers blueprint Section 20's 'Time alignment' and 'Fallback' test rows:
  - Forecast valid_time - run_time equals lead_hours
  - If a source is missing, the blend uses remaining sources safely
"""
import pandas as pd
import pytest

from app.data.normalization import (
    SchemaValidationError,
    build_quality_mask,
    validate_canonical_frame,
)


def _canonical_row(**overrides):
    row = {
        "model": "GFS",
        "run_time": pd.Timestamp("2026-01-01T00:00:00Z"),
        "valid_time": pd.Timestamp("2026-01-04T00:00:00Z"),
        "lead_hours": 72,
        "variable": "precipitation",
        "region": "KWG",
        "lat": 10.5,
        "lon": 76.2,
        "forecast_value": 42.0,
        "unit": "mm/24h",
    }
    row.update(overrides)
    return row


def test_time_alignment_passes_for_consistent_rows():
    df = pd.DataFrame([_canonical_row(), _canonical_row(model="IFS")])
    validate_canonical_frame(df)  # should not raise


def test_time_alignment_rejects_inconsistent_lead_hours():
    bad_row = _canonical_row(lead_hours=48)  # valid_time - run_time is actually 72h
    df = pd.DataFrame([bad_row])
    with pytest.raises(SchemaValidationError):
        validate_canonical_frame(df)


def test_unit_consistency_enforced():
    df = pd.DataFrame([_canonical_row(unit="mm")])  # wrong unit for precipitation
    with pytest.raises(SchemaValidationError):
        validate_canonical_frame(df)


def test_quality_mask_all_sources_present():
    records = [_canonical_row(model=m) for m in ("GFS", "IFS", "AIFS")]
    ctx = build_quality_mask(records)
    assert ctx.quality_ok is True
    assert ctx.missing_sources == []
    assert set(ctx.available_sources) == {"GFS", "IFS", "AIFS"}


def test_quality_mask_fallback_when_source_missing():
    """A missing AIFS source should be flagged, not silently ignored."""
    records = [_canonical_row(model=m) for m in ("GFS", "IFS")]
    ctx = build_quality_mask(records, expected_sources=["GFS", "IFS", "AIFS"])
    assert ctx.missing_sources == ["AIFS"]
    assert ctx.quality_ok is True  # 2 sources is still enough for a meaningful blend


def test_quality_mask_flags_insufficient_sources():
    records = [_canonical_row(model="GFS")]
    ctx = build_quality_mask(records, expected_sources=["GFS", "IFS", "AIFS"])
    assert ctx.quality_ok is False  # only 1 source - blend should fall back
