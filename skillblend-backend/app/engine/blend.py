"""
Adaptive blending model (blueprint Section 8).

    F_final(x,t) = sum_i w_i(x,t,c) * F_i(x,t),  sum_i w_i = 1

Implementation follows the blueprint's "practical student
implementation": a gradient-boosted meta-model predicts one
error/skill score per source given context; scores are converted to
normalized positive weights with softmax, which keeps weights
interpretable and non-negative (no arbitrary negative weights).

Model choice: LightGBM/XGBoost are the blueprint's recommendation for
"fast iteration + interpretability". This module tries LightGBM first,
then falls back to scikit-learn's HistGradientBoostingRegressor if
LightGBM/XGBoost are not installed in the runtime environment - the
feature contract (see `_FEATURE_COLUMNS`) is identical either way, so
callers and the SHAP/explanation layer do not need to know which
backend trained the model. This graceful-degradation approach mirrors
the blueprint's own "Feasibility guardrails" table (Section 18).

Baselines implemented alongside the learned model, exactly matching
the blueprint's staged table in Section 8:
  - equal_average          ("No intelligence" - benchmark only)
  - skill_weighted_average ("Skill-aware" - robust fallback, used
                             automatically when the learned model or a
                             source is unavailable)
  - meta_model_blend       ("Context-aware" / "Final" - the adaptive,
                             softmax-normalized blend)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb

    _BACKEND = "lightgbm"
except ImportError:  # pragma: no cover - exercised when lightgbm isn't installed
    from sklearn.ensemble import HistGradientBoostingRegressor

    _BACKEND = "sklearn_hgbr"

import joblib

from app.config import BLEND_MODEL_PATH, PILOT_ZONES, REGIMES, SEASONS, SOURCES

_FEATURE_COLUMNS = (
    ["lead_hours", "historical_skill", "monsoon_sensitivity", "wd_sensitivity", "coastal"]
    + [f"model_{m}" for m in SOURCES]
    + [f"season_{s}" for s in SEASONS]
    + [f"regime_{r}" for r in REGIMES]
)


def _zone_static(region: str) -> tuple[float, float, int]:
    zone = PILOT_ZONES[region]
    return zone.monsoon_sensitivity, zone.wd_sensitivity, int(zone.coastal)


def build_feature_row(
    model: str,
    lead_hours: int,
    season: str,
    regime: str,
    region: str,
    historical_skill: float,
) -> dict:
    monsoon_sens, wd_sens, coastal = _zone_static(region)
    row = {
        "lead_hours": lead_hours,
        "historical_skill": historical_skill,
        "monsoon_sensitivity": monsoon_sens,
        "wd_sensitivity": wd_sens,
        "coastal": coastal,
    }
    for m in SOURCES:
        row[f"model_{m}"] = int(m == model)
    for s in SEASONS:
        row[f"season_{s}"] = int(s == season)
    for r in REGIMES:
        row[f"regime_{r}"] = int(r == regime)
    return row


def build_feature_frame(df: pd.DataFrame, historical_skill_col: str = "historical_skill") -> pd.DataFrame:
    """
    Vectorized counterpart to `build_feature_row`, used to build large
    training frames efficiently. `df` must contain columns: model,
    lead_hours, season, regime, region, and `historical_skill_col`.
    Returns a frame with exactly `_FEATURE_COLUMNS`, in order.
    """
    out = pd.DataFrame(index=df.index)
    out["lead_hours"] = df["lead_hours"].astype(float)
    out["historical_skill"] = df[historical_skill_col].astype(float)
    zone_static = df["region"].map(lambda r: _zone_static(r))
    out["monsoon_sensitivity"] = zone_static.map(lambda t: t[0])
    out["wd_sensitivity"] = zone_static.map(lambda t: t[1])
    out["coastal"] = zone_static.map(lambda t: t[2]).astype(float)
    for m in SOURCES:
        out[f"model_{m}"] = (df["model"] == m).astype(float)
    for s in SEASONS:
        out[f"season_{s}"] = (df["season"] == s).astype(float)
    for r in REGIMES:
        out[f"regime_{r}"] = (df["regime"] == r).astype(float)
    return out[_FEATURE_COLUMNS]


@dataclass
class BlendModel:
    regressor: object
    backend: str = _BACKEND
    feature_columns: list = field(default_factory=lambda: list(_FEATURE_COLUMNS))

    # -- training -----------------------------------------------------
    @classmethod
    def train(cls, training_frame: pd.DataFrame) -> "BlendModel":
        """
        `training_frame` must contain one row per historical
        (model, region, variable, lead_hours, season, regime, run_time)
        instance with columns: the _FEATURE_COLUMNS plus `abs_error`
        (the target - the meta-model learns to predict how wrong each
        source tends to be in that context, which is then converted to
        a trust weight at inference time).
        """
        X = training_frame[_FEATURE_COLUMNS]
        y = training_frame["abs_error"]

        if _BACKEND == "lightgbm":
            regressor = lgb.LGBMRegressor(
                n_estimators=250,
                num_leaves=31,
                learning_rate=0.05,
                min_child_samples=25,
                random_state=26081,
                verbosity=-1,
            )
        else:
            regressor = HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.05,
                random_state=26081,
            )
        regressor.fit(X, y)
        return cls(regressor=regressor)

    # -- persistence ----------------------------------------------------
    def save(self, path=BLEND_MODEL_PATH) -> None:
        joblib.dump({"regressor": self.regressor, "backend": self.backend}, path)

    @classmethod
    def load(cls, path=BLEND_MODEL_PATH) -> "BlendModel":
        payload = joblib.load(path)
        return cls(regressor=payload["regressor"], backend=payload["backend"])

    # -- inference --------------------------------------------------------
    def predict_expected_error(self, feature_rows: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.regressor.predict(feature_rows[self.feature_columns]))

    def predict_weights(
        self,
        region: str,
        variable: str,
        lead_hours: int,
        season: str,
        regime: str,
        available_sources: list[str],
        historical_skill_lookup: dict[str, float],
        temperature: float = 1.0,
    ) -> dict[str, float]:
        """
        Predict per-source expected error in this context, then convert
        to softmax weights over *negative* error (lower predicted error
        -> higher weight). `temperature` controls how sharply weights
        concentrate on the best source; 1.0 is a reasonable default,
        higher values flatten towards equal weighting.
        """
        rows = [
            build_feature_row(
                model=m,
                lead_hours=lead_hours,
                season=season,
                regime=regime,
                region=region,
                historical_skill=historical_skill_lookup.get(m, 0.5),
            )
            for m in available_sources
        ]
        feature_df = pd.DataFrame(rows)
        expected_errors = self.predict_expected_error(feature_df)
        return softmax_weights(available_sources, expected_errors, temperature=temperature)


def softmax_weights(sources: list[str], expected_errors: np.ndarray, temperature: float = 1.0) -> dict[str, float]:
    scores = -np.asarray(expected_errors, dtype=float) / max(temperature, 1e-6)
    scores = scores - scores.max()  # numerical stability
    exp = np.exp(scores)
    weights = exp / exp.sum()
    return {s: float(w) for s, w in zip(sources, weights)}


def equal_average_weights(available_sources: list[str]) -> dict[str, float]:
    n = len(available_sources)
    return {s: 1.0 / n for s in available_sources}


def skill_weighted_average_weights(
    available_sources: list[str], historical_skill_lookup: dict[str, float]
) -> dict[str, float]:
    """Robust fallback: weight proportional to historical skill (Section 8, 'Skill-aware')."""
    raw = {s: max(historical_skill_lookup.get(s, 0.5), 1e-3) for s in available_sources}
    total = sum(raw.values())
    return {s: v / total for s, v in raw.items()}


def weighted_blend(values: dict[str, float], weights: dict[str, float]) -> float:
    return float(sum(values[s] * weights[s] for s in weights))


_MODEL_CACHE: BlendModel | None = None


def get_blend_model(force_reload: bool = False) -> BlendModel | None:
    """Lazily load the trained blend model; returns None if not yet trained."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None and not force_reload:
        return _MODEL_CACHE
    if not BLEND_MODEL_PATH.exists():
        return None
    _MODEL_CACHE = BlendModel.load()
    return _MODEL_CACHE
