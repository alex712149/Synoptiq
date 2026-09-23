"""
Canonical data contract (blueprint Section 15) expressed as Pydantic
models, plus the request/response shapes for the API layer.

Every module in the system is expected to agree on this shape so that
ingestion, the skill engine, the blending model and the API can be
developed independently ("Vibe-Coding friendly").
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SourceForecast(BaseModel):
    """One raw candidate forecast (blueprint 'Forecast source')."""

    model: str = Field(..., description="Forecast source identifier, e.g. GFS/IFS/AIFS")
    run_time: datetime
    valid_time: datetime
    lead_hours: int
    region: str
    variable: str
    forecast_value: float
    unit: str


class ForecastRecord(BaseModel):
    """The full canonical row described in Section 15's data-contract table."""

    model: str
    run_time: datetime
    valid_time: datetime
    lead_hours: int
    variable: str
    region: str
    lat: float
    lon: float
    forecast_value: float
    regime_probs: dict[str, float]
    historical_skill: float
    model_weight: float
    trust_score: float
    bust_probability: float


class BlendRequest(BaseModel):
    region: str = Field(..., description="Pilot zone code, e.g. KWG, BOB, IGP")
    variable: str = Field(..., description="precipitation | temperature | wind_speed")
    lead_hours: int = Field(..., description="Forecast lead time in hours")
    run_time: Optional[datetime] = Field(
        None, description="Forecast cycle to blend; defaults to the latest cached cycle"
    )

    @field_validator("region")
    @classmethod
    def _upper_region(cls, v: str) -> str:
        return v.upper()

    @field_validator("variable")
    @classmethod
    def _lower_variable(cls, v: str) -> str:
        return v.lower()


class SourceContribution(BaseModel):
    model: str
    forecast_value: float
    weight: float
    historical_skill: float


class TrustBreakdown(BaseModel):
    historical_skill_component: float
    disagreement_component: float
    lead_time_component: float
    regime_stability_component: float
    data_quality_component: float
    trust_score: float


class ExplanationDriver(BaseModel):
    feature: str
    contribution: float
    direction: str  # "increases_trust" | "decreases_trust" | "increases_weight" | "decreases_weight"
    detail: str


class BlendResponse(BaseModel):
    region: str
    region_name: str
    variable: str
    unit: str
    run_time: datetime
    valid_time: datetime
    lead_hours: int
    season: str
    regime: str
    regime_probs: dict[str, float]
    sources: list[SourceContribution]
    disagreement: float
    raw_blend_value: float
    bias_corrected_value: float
    final_value: float
    trust: TrustBreakdown
    bust_probability: float
    bust_flag: bool
    abstain: bool
    explanation: list[ExplanationDriver]
    fallback_used: bool
    provenance: dict[str, str]


class ExtremeProbability(BaseModel):
    variable: str
    threshold: float
    unit: str
    probability: float
    calibrated: bool


class ExtremeGuidanceResponse(BaseModel):
    region: str
    lead_hours: int
    valid_time: datetime
    guidance: list[ExtremeProbability]


class WeightMapPoint(BaseModel):
    lead_hours: int
    weights: dict[str, float]
    trust_score: float


class WeightMapResponse(BaseModel):
    region: str
    variable: str
    regime: str
    season: str
    points: list[WeightMapPoint]


class SkillScorecardRow(BaseModel):
    model: str
    region: str
    variable: str
    lead_hours: int
    season: str
    regime: str
    n_obs: int
    rmse: Optional[float] = None
    mae: Optional[float] = None
    bias: Optional[float] = None
    csi: Optional[float] = None
    pod: Optional[float] = None
    far: Optional[float] = None
    fss: Optional[float] = None


class VerificationSummary(BaseModel):
    region: str
    variable: str
    threshold: float
    best_single_model: str
    best_single_model_csi: float
    skillblend_csi: float
    relative_csi_improvement: float
    meets_target: bool
    target_relative_csi_improvement: float


class ReplayEventSummary(BaseModel):
    event_id: str
    region: str
    variable: str
    valid_time: datetime
    label: str
    headline: str


class ReplayEventDetail(BaseModel):
    event_id: str
    region: str
    variable: str
    unit: str
    valid_time: datetime
    lead_hours: int
    label: str
    headline: str
    raw_sources: list[SourceContribution]
    naive_average: float
    single_model_choice: dict[str, str | float]
    skillblend_blend: float
    reference_value: float
    naive_average_error: float
    single_model_error: float
    skillblend_error: float
    narrative: list[str]
