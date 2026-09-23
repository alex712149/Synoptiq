# SkillBlend API

**Hybrid AI–NWP Multi-Model Forecast Blending System**
SIH Problem Statement 26081 · NCMRWF · Ministry of Earth Sciences

This is a working backend prototype for the PS, built from the
`SkillBlend_PS26081_Complete_Blueprint` document: it implements the
Historical Skill Engine, the auditable rule-based Context/Regime
engine, an adaptive LightGBM-style blending meta-model, quantile-map
bias correction, isotonic probability calibration, a Forecast Trust +
Bust-Probability layer, SHAP-style explanations, and an offline-safe
Counterfactual Bust Replay mode — all exposed through a FastAPI
service.

## 1. The honest headline: what's real vs. simulated here

- **Real and load-bearing:** every algorithm in the blueprint —
  normalization/quality-masking, the historical skill engine (RMSE,
  MAE, bias, CSI, POD, FAR, an FSS proxy), the context-aware
  meta-model that produces softmax blend weights, regime-conditioned
  quantile-mapping bias correction, isotonic exceedance calibration,
  the five-signal trust score, a trained Forecast Bust Probability
  classifier, and structured (SHAP or perturbation-based) explanations
  — is implemented in `app/engine/` and wired together in
  `app/services/pipeline.py`. Nothing in that chain is hard-coded or
  faked; every number in an API response is computed from the cached
  historical data through that exact pipeline.
- **Simulated because this environment has no network access to
  NOAA/ECMWF/Copernicus/NASA:** `app/data/synthetic_generator.py`
  stands in for live GFS/IFS/AIFS/ERA5/IMERG ingestion. It generates a
  physically-motivated synthetic dataset where each source has its
  own, deliberately different bias/noise characteristics by region,
  season, lead time and weather regime (documented in the module's
  docstring) — so "which model to trust when" is a genuine, learnable
  signal, not a random one. **Swapping this module for real
  cfgrib/xarray ingestion against the sources in Section 4 of the
  blueprint is the only change needed to go from prototype to
  production**; every downstream engine is agnostic to where the rows
  came from because they all speak the same canonical schema
  (`app/schemas.py`, `app/data/normalization.py`).
- Also documented as deliberate MVP scope cuts (all called out
  in-code, matching the blueprint's own guardrails):
  - One representative grid point per pilot zone rather than a full
    grid (blueprint Section 12's documented scope cut). This means the
    "FSS" metric in the skill table is a **temporal-neighbourhood
    proxy**, not the textbook spatial Fractions Skill Score — see the
    docstring in `app/engine/skill.py`.
  - LightGBM/XGBoost and SHAP are the blueprint's recommended stack;
    this code **tries them first and transparently falls back** to
    `sklearn.ensemble.HistGradientBoosting{Regressor,Classifier}` and a
    deterministic perturbation-based explanation method when those
    packages aren't installed, so the system degrades gracefully
    instead of failing outright (see `app/engine/blend.py`,
    `app/engine/trust.py`, `app/engine/explain.py`). Install
    `lightgbm`/`shap` (already in `requirements.txt`) for the intended
    behavior.
  - Storage uses a local joblib-pickle cache if `pyarrow`/`fastparquet`
    aren't installed, and local JSON/disk instead of Zarr +
    PostgreSQL/PostGIS (`app/data/storage.py`). The public functions
    are the seam to swap in real Zarr/Postgres later without touching
    any engine code.

## 2. Architecture (mirrors blueprint Figure 2)

```
app/
  config.py            # pilot zones, sources, variables, thresholds, lead ladder
  schemas.py            # canonical data contract (Pydantic)
  data/
    synthetic_generator.py  # stands in for live GFS/IFS/AIFS/ERA5/IMERG ingestion
    normalization.py        # canonical schema, time-alignment invariant, quality mask
    storage.py               # local cache (Zarr/PostGIS stand-in)
  engine/
    regime.py            # rule-based, auditable weather-regime classifier
    skill.py              # Historical Skill Engine (RMSE/MAE/bias/CSI/POD/FAR/FSS-proxy)
    blend.py               # adaptive meta-model -> softmax blend weights (+ fallbacks)
    calibration.py         # quantile-mapping bias correction + isotonic calibration
    trust.py                # Trust score + Forecast Bust Probability model
    explain.py              # SHAP / perturbation-based structured explanations
    verification.py         # SkillBlend vs. best-single-model CSI on a holdout split
  services/
    pipeline.py            # orchestrates the full ingest -> blend -> serve flow
  api/                     # FastAPI routers
  main.py                   # FastAPI app

scripts/
  seed_and_train.py    # one-shot: generate data, train every model, cache calibrators
                         # + replay events (implements blueprint Section 13, steps 2-13)
  smoke_test.py         # exercises the pipeline directly, no FastAPI/uvicorn required

tests/                  # pytest suite covering blueprint Section 20's test table
```

## 3. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# One-time: generate the synthetic historical dataset and train every
# model (blend meta-model, bias correctors, exceedance calibrators,
# bust classifier, replay events). Takes ~15-20s for the default
# 730-day history.
python -m scripts.seed_and_train

# Run the API
uvicorn app.main:app --reload --port 8000
# -> interactive docs at http://localhost:8000/docs
```

Sanity-check the engine layer without FastAPI installed:

```bash
python -m scripts.smoke_test
```

Run the tests (they skip the integration suite gracefully if you
haven't run `seed_and_train` yet):

```bash
pytest
```

Environment variables (optional):

- `SKILLBLEND_HISTORY_DAYS` — synthetic history length in days (default 730)
- `SKILLBLEND_ARTIFACTS_DIR` — where cached data/models/replay live (default `./artifacts`)

## 4. API reference (all routes under `/api/v1`)

| Route | PS expected outcome it serves |
|---|---|
| `GET /forecast/blend?region=&variable=&lead_hours=` | **#1 Dynamically blended forecast** — per-source contributions, weights, trust breakdown, bust probability, explanation |
| `GET /weights/map?region=&variable=&season=&regime=` | **#2 Model-weight maps** across the full lead-time ladder |
| `GET /skill/scorecard?...` | **#3 Improved forecast skill** — raw Historical Skill Engine table |
| `GET /skill/verification?...` | **#3** headline KPI — SkillBlend vs. best single model CSI on a strict holdout, vs. the +5% target |
| `GET /extreme/guidance?region=&lead_hours=` | **#4 Extreme-weather guidance** — calibrated P(rain>50mm), P(temp>40°C), P(wind>17m/s) |
| `GET /replay/events`, `GET /replay/events/{id}` | Offline-safe Counterfactual Bust Replay (demo-day reliability) |
| `GET /regions`, `/sources`, `/variables`, `/lead-times` | Reference metadata |
| `GET /health` | readiness check |

Example:

```bash
curl "http://localhost:8000/api/v1/forecast/blend?region=KWG&variable=precipitation&lead_hours=72"
```

## 5. The judge-facing demo (matches blueprint Section 17)

1. `GET /replay/events` — **offline-safe by default**, no live network
   dependency during judging.
2. Open one event: `GET /replay/events/{event_id}` shows the three raw
   source forecasts disagreeing, the detected regime, a naive-average
   baseline, a "trust one model always" baseline, and SkillBlend's
   actual result against the verified value — with a plain-language
   narrative built from the same numbers, never invented.
3. `GET /forecast/blend` for a live-cycle example — walk through the
   weight breakdown, trust panel and explanation drivers.
4. `GET /weights/map` — animate how the learned weights shift across
   lead time for a chosen region/season/regime.
5. `GET /skill/verification` — the headline number: SkillBlend's CSI
   vs. the best single model's CSI at the 50mm threshold, and whether
   the pre-registered +5% relative-improvement target was met. **This
   number is measured on a temporal holdout the models never trained
   on, in every run** — it is not fabricated, and in some
   region/variable combinations it will legitimately come out
   negative, which is the honest result of the synthetic experiment.

## 6. Notable design choices worth defending to judges

- **No cheating off ground truth at inference time.** The live weather
  regime used to blend/trust/calibrate a forecast always comes from
  the rule-based classifier applied to the current raw source
  forecasts (`services/pipeline._regime_signal`) — never from the
  hidden "true" regime label that only exists inside the synthetic
  generator. The offline Historical Skill Engine table is legitimately
  allowed to use analyzed/hindcast regime labels when scoring past
  performance, exactly as an operational verification team would with
  reanalysis-informed classifications after the fact. Keeping that
  separation is what makes the backtest CSI numbers meaningful instead
  of trivially inflated.
- **Forecast abstention** (`engine.trust.should_abstain`): when trust
  is low or bust risk is high, the API flags `"abstain": true` instead
  of quietly returning a confident-looking number.
- **Graceful degradation everywhere:** missing source → quality mask
  falls back to skill-weighted averaging instead of failing; no
  LightGBM/SHAP installed → sklearn/perturbation fallback; no
  pyarrow → pickle cache. All three fallbacks are exercised by the
  test suite, not just assumed to work.
- **Everything numeric in the trust/weight panels is attributable.**
  The explanation layer never lets an LLM invent a reason — it only
  ever narrates numbers the models actually produced (matching the
  blueprint's explicit SHAP guardrail in Section 10).

## 7. Known limitations (stated plainly, not hidden)

- Single representative point per pilot zone, not a full grid — see
  the FSS-proxy note above. Swapping in gridded data mainly touches
  `synthetic_generator.py` / a real ingestion module and
  `engine/skill.py`'s FSS function.
- The Forecast Bust classifier's positive class is rare by
  construction (large errors are uncommon in a reasonably-tuned
  synthetic world), so with a short history its explanations can come
  back sparse — this improves with the default 730-day history and
  would improve further with more historical seasons in a real
  deployment.
- Regime detection is intentionally simple/rule-based per the
  blueprint's own "student-friendly" guidance — an ML regime
  classifier is explicitly future work (blueprint Section 21), not
  something this prototype claims to have solved.
