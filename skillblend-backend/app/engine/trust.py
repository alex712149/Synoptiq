"""
Forecast Trust, Disagreement and Bust Detection (blueprint Section 10).

"Trust is not another forecast value. It is a decision-support signal
combining historical skill, current model agreement, lead time, regime
stability and data quality."

This module implements:
  1. `compute_disagreement`      - spread across available sources.
  2. `compute_trust_breakdown`   - the five-signal composite trust
                                    score, returned as a fully itemised
                                    breakdown so the dashboard/API can
                                    show *why* trust is what it is
                                    (Section 10's worked example).
  3. `BustModel`                 - a binary classifier estimating
                                    P(large forecast error), i.e. the
                                    Forecast Bust Probability, trained
                                    on backtested blend residuals.
  4. `should_abstain`            - "Forecast abstention" innovation
                                    feature (Section 11): the system
                                    can say "low confidence - review"
                                    instead of forcing a number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb

    _BACKEND = "lightgbm"
except ImportError:  # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingClassifier

    _BACKEND = "sklearn_hgbc"

import joblib

from app.config import BUST_MODEL_PATH

_DISAGREEMENT_SCALE = {"precipitation": 25.0, "temperature": 2.5, "wind_speed": 4.0}
_TRUST_WEIGHTS = {
    "historical_skill": 0.35,
    "disagreement": 0.25,
    "lead_time": 0.15,
    "regime_stability": 0.15,
    "data_quality": 0.10,
}

BUST_FEATURE_COLUMNS = [
    "lead_hours",
    "disagreement_norm",
    "historical_skill_avg",
    "regime_stability",
    "data_quality",
]


def compute_disagreement(values: dict[str, float]) -> float:
    """Raw spread (population std-dev) across available source forecasts."""
    arr = np.array(list(values.values()), dtype=float)
    if len(arr) < 2:
        return 0.0
    return float(arr.std())


def normalize_disagreement(disagreement: float, variable: str) -> float:
    """0 (identical sources) .. ~1+ (wildly disagreeing sources)."""
    return float(disagreement / _DISAGREEMENT_SCALE[variable])


@dataclass
class TrustBreakdown:
    historical_skill_component: float
    disagreement_component: float
    lead_time_component: float
    regime_stability_component: float
    data_quality_component: float
    trust_score: float


def compute_trust_breakdown(
    historical_skill_avg: float,
    disagreement_norm: float,
    lead_hours: int,
    regime_stability: float,
    data_quality: float,
    max_lead_hours: int = 168,
) -> TrustBreakdown:
    skill_c = float(np.clip(historical_skill_avg, 0, 1))
    disagreement_c = float(np.clip(1 - disagreement_norm, 0, 1))
    lead_time_c = float(np.clip(1 - (lead_hours / max_lead_hours), 0.05, 1))
    regime_c = float(np.clip(regime_stability, 0, 1))
    quality_c = float(np.clip(data_quality, 0, 1))

    trust = (
        _TRUST_WEIGHTS["historical_skill"] * skill_c
        + _TRUST_WEIGHTS["disagreement"] * disagreement_c
        + _TRUST_WEIGHTS["lead_time"] * lead_time_c
        + _TRUST_WEIGHTS["regime_stability"] * regime_c
        + _TRUST_WEIGHTS["data_quality"] * quality_c
    )
    return TrustBreakdown(
        historical_skill_component=round(skill_c, 4),
        disagreement_component=round(disagreement_c, 4),
        lead_time_component=round(lead_time_c, 4),
        regime_stability_component=round(regime_c, 4),
        data_quality_component=round(quality_c, 4),
        trust_score=round(float(np.clip(trust, 0, 1)), 4),
    )


def should_abstain(trust_score: float, bust_probability: float, threshold_trust: float = 0.35, threshold_bust: float = 0.55) -> bool:
    """
    Innovation feature (Section 11, 'Forecast abstention'): rather than
    always forcing a clean number, flag when the system itself does not
    trust its own output so a human forecaster should review it.
    """
    return trust_score < threshold_trust or bust_probability > threshold_bust


@dataclass
class BustModel:
    classifier: object
    backend: str = _BACKEND
    feature_columns: list = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = list(BUST_FEATURE_COLUMNS)

    @classmethod
    def train(cls, training_frame: pd.DataFrame) -> "BustModel":
        X = training_frame[BUST_FEATURE_COLUMNS]
        y = training_frame["bust"].astype(int)
        if _BACKEND == "lightgbm":
            clf = lgb.LGBMClassifier(
                n_estimators=200,
                num_leaves=15,
                learning_rate=0.05,
                min_child_samples=25,
                random_state=26081,
                verbosity=-1,
            )
        else:
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=26081)
        clf.fit(X, y)
        return cls(classifier=clf)

    def save(self, path=BUST_MODEL_PATH) -> None:
        joblib.dump({"classifier": self.classifier, "backend": self.backend}, path)

    @classmethod
    def load(cls, path=BUST_MODEL_PATH) -> "BustModel":
        payload = joblib.load(path)
        return cls(classifier=payload["classifier"], backend=payload["backend"])

    def predict_proba(self, feature_row: dict) -> float:
        df = pd.DataFrame([feature_row])[self.feature_columns]
        proba = self.classifier.predict_proba(df)[0]
        classes = list(self.classifier.classes_)
        idx = classes.index(1) if 1 in classes else int(np.argmax(proba))
        return float(proba[idx])


_BUST_MODEL_CACHE: BustModel | None = None


def get_bust_model(force_reload: bool = False) -> BustModel | None:
    global _BUST_MODEL_CACHE
    if _BUST_MODEL_CACHE is not None and not force_reload:
        return _BUST_MODEL_CACHE
    if not BUST_MODEL_PATH.exists():
        return None
    _BUST_MODEL_CACHE = BustModel.load()
    return _BUST_MODEL_CACHE
