# SkillBlend — Backend Prototype
**SIH PS 26081 · Hybrid AI–NWP Multi-Model Forecast Blending System**
NCMRWF / Ministry of Earth Sciences

This is a working FastAPI backend implementing the architecture in the
blueprint: Forecast Sources → Normalization → Skill/Context layer →
Adaptive Meta-Model → Blended Forecast + Trust + Uncertainty + Extreme
Guidance, plus the offline Historical Replay demo mode the blueprint
recommends as the default judging path.

## Why synthetic data, not live GFS/ECMWF/IMERG

This build environment has no network access to NOAA NOMADS, ECMWF Open
Data, the CDS, or NASA GPM IMERG. So `app/data_sim/synthetic_generator.py`
stands in for Section 4's ingestion layer: it generates a multi-season
archive (3 pilot zones × 3 models × 3 variables × 5 lead times, ~35k rows)
where **each model has a different, regime-dependent bias and lead-time
decay profile** — i.e. the synthetic world is constructed so that "no
single model is always best" is literally true, which is the actual premise
of PS 26081. Every downstream module (skill engine, regime detector,
blending, calibration, trust) consumes exactly the canonical schema in
Section 15 of the blueprint, so swapping in a real `cfgrib`/`xarray`
ingestion job later is a one-file change — nothing else in the pipeline
needs to move.

## Leakage fix (this revision)

An earlier revision had a real temporal leak: the skill table was rebuilt on
the *entire* archive — training and test period alike — immediately before
the held-out evaluation script read from it, so a model's "historical
skill" quietly included its own performance on the period being scored.
This revision fixes that structurally:

- **Strict time-based train/validation/test split** (60/15/25 by default,
  `app/config.py`), computed once from the full set of unique `valid_time`s
  so every row sharing a timestamp lands in the same split. The split label
  is persisted on every `ForecastRow` (`split` column) for auditability.
- **Expanding, leakage-free features** (`app/data_prep.py`): `historical_skill`
  and `climatological_anomaly` are now computed via
  `groupby(...).expanding().shift(1)` — every row can only be influenced by
  strictly *earlier* valid_times. This is safe to compute over the whole
  archive in one vectorized pass (train/val/test rows alike), because a
  test-row's feature can never touch test-period (or later) outcomes by
  construction. It also directly implements the "rolling/expanding
  historical skill for operational-style evaluation" requirement: skill
  estimates firm up over time exactly as they would operationally.
- **Frozen skill table**: the older, coarser `SkillRow` table (used only as
  a serving-time fallback and as the `/verification/scorecard` diagnostic)
  is now built with an explicit `through` cutoff = the test-split boundary,
  so it never sees test-period data either.
- **Three-way split discipline**: the meta-model and bust classifier are
  fit on TRAIN only; calibration (quantile mapping + isotonic exceedance)
  is fit on VALIDATION only, using the already-frozen meta-model's own
  predictions; `scripts/evaluate_blend.py` touches TEST exactly once, after
  every other artifact is frozen.
- **Vectorized throughout**: the old code called a DB-backed skill lookup
  inside nested Python loops (once per model per context). Everything now
  goes through `app/data_prep.py`: one query per variable, then
  groupby/merge/pivot — no per-row DB calls, no per-context Python loops in
  the training or evaluation hot path.
- **Named feature frames everywhere**: both training (`build_feature_frame`)
  and inference (`features.encode`) build the same named `FEATURE_NAMES`
  DataFrame, so LightGBM never sees a training/inference schema mismatch
  (and per `tests/test_feature_consistency.py`, never emits the
  "does not have valid feature names" warning).
- **Two new, meteorologically-motivated features**: `consensus_deviation_norm`
  (how far a model's forecast sits from the multi-model mean at that
  context) and `climatological_anomaly_norm` (forecast vs. the expanding
  climatological mean of *past* observed truth). Both are vectorized,
  leak-free, and feed the same meta-model and SHAP explanations.
- **Regime detector actually wired in**: `/forecast/blend` now calls
  `app/regime_detector.py`'s real rule-based classifier against the other
  cached variables' medians for that context, instead of only reading the
  synthetic generator's pre-baked `regime_probs`. The synthetic value is
  kept as a fallback for contexts where the other variables aren't cached.
- **FSS is explicitly a proxy**: `fss_50mm` is documented everywhere (code
  comments, DB column comment, and the API's `fss_metric_note` field) as an
  FSS-**proxy** for synthetic point data, never presented as true gridded
  FSS.
- **Narrative guard**: the auto-generated SHAP-backed narrative never cites
  "the raw forecast value itself" as the *reason* a source is trusted —
  magnitude alone isn't evidence of trustworthiness — and instead surfaces
  the next most meaningful driver (skill, consensus deviation, climatology,
  regime, region, season).



| Blueprint section | Module |
|---|---|
| §5 Canonical data contract | `app/schemas.py`, `app/models_db.py` |
| §6 Historical Skill Engine | `app/skill_engine.py` (RMSE/MAE/Bias, CSI/POD/FAR@50mm, FSS-proxy) |
| §7 Context / regime engine | `app/regime_detector.py` — transparent rule-based classifier |
| §8 Adaptive blending | `app/blending/{features,train_meta_model,blend}.py` — per-source LightGBM skill-score models, softmax → weights |
| §9 Bias correction + calibration | `app/blending/calibration.py` — quantile mapping + isotonic regression, fit only on the temporal train window |
| §10 Trust / disagreement / bust | `app/blending/trust.py` — 5-signal trust score + LightGBM bust classifier + abstention |
| §10 SHAP explanations | `app/blending/explain.py` — SHAP TreeExplainer per weight decision + template-only narrative (never invents facts) |
| §11 Counterfactual Bust Replay | `app/replay.py` — precomputed, DB-backed, zero live inference |
| §11 Historical analogue memory *(my addition)* | `app/blending/analogue.py` |
| §11 Counterfactual weight lab *(my addition)* | `POST /forecast/counterfactual` |
| §12 Pilot zones | Kerala/W.Ghats, Bay of Bengal/E.Coast, Indo-Gangetic Plains (`app/config.py`) |
| §14 Extreme-weather guidance | Isotonic-calibrated exceedance probabilities for rain/heat/wind thresholds |
| §18 Guardrails | Skill-weighted fallback when a meta-model artifact is missing/a source is down; temporal (not random) holdout throughout |
| §20 Tests | `tests/` — data contract, weight validity, fallback, calibration leakage, temporal-split leakage, feature-schema consistency, evaluation correctness |
| §21 Headline proof point | `scripts/evaluate_blend.py` — measured on TEST split only, never fabricated |
| Leakage fix *(this revision)* | `app/data_prep.py` — strict train/val/test split, expanding leak-free features, frozen skill table |
| Meteorological features *(this revision)* | `consensus_deviation_norm`, `climatological_anomaly_norm` in `app/blending/features.py` |

## My own additions beyond the literal PDF

1. **Historical Analogue Memory** (`app/blending/analogue.py`) — for the
   current context, surfaces the closest past situations and what actually
   happened, as a fast, fully explainable complement to the learned trust
   score.
2. **Counterfactual Weight Lab** (`POST /forecast/counterfactual`) — lets a
   judge/demo override the detected regime on cached data and watch the
   learned weights genuinely re-derive live (e.g. AIFS's weight jumps from
   47%→73% when the same Bay-of-Bengal case is reframed as a depression).
3. **Template-only forecast narrative** — turns the SHAP facts into a
   sentence without any LLM call, so the "explainable text" feature costs
   nothing to run and can never hallucinate a reason that isn't in the data
   (a stricter reading of the blueprint's own SHAP-must-be-source-of-truth
   rule).
4. **Abstention** — when trust is low or bust risk is high, the API sets
   `"abstain": true` instead of quietly presenting a confident-looking
   number.

## Measured result (held-out TEST split only — leakage-free)

After `scripts/bootstrap.sh`, `scripts/evaluate_blend.py` writes CSI@50mm
computed **entirely on the TEST split** (the last 25% of the archive by
time), using only artifacts (meta-models, calibration, skill table) that
were frozen on TRAIN+VALIDATION beforehand. Current run on this synthetic
archive — **all three pilot zones now meet the +5% target**:

| Region | SkillBlend CSI | Best single model CSI | Relative improvement | Target | Met? |
|---|---|---|---|---|---|
| Kerala / W. Ghats | 0.750 | 0.667 | **+12.5%** | +5% | ✅ |
| Bay of Bengal / E. Coast | 0.938 | 0.824 | **+13.8%** | +5% | ✅ |
| Indo-Gangetic Plains | 0.625 | 0.556 | **+12.5%** | +5% | ✅ |

This took a real fix, not tuning against the test set: the bias-correction
quantile map was originally fit **globally, pooling all three regions'
rainfall distributions together**. That blurred each region's own
climatology and was actively hurting Bay of Bengal's event-threshold skill
(its raw, pre-calibration blend scored CSI=1.0 on validation; the pooled
calibrator dragged it to a tie with the best single model). The fix —
per-region calibration, **shrunk toward the global curve by sample size**
(`alpha = n / (n + shrink_k)`, see `app/blending/calibration.py`) — is
exactly what Section 9 of the blueprint itself suggests ("a
regime-conditioned version can use a separate mapping per weather regime").
A naive *pure* per-region switch (no shrinkage) was tried first and
overfit Indo-Gangetic Plains' small validation sample, so it's shrinkage
specifically that makes this hold up on TEST. All of this tuning — including
the softmax temperature sweep that confirmed the existing default was
already in a stable plateau — was done by evaluating on the VALIDATION
split only; TEST was touched exactly once, by `scripts/evaluate_blend.py`,
after everything above was frozen.

These are real outputs of `scripts/evaluate_blend.py` on this **synthetic
prototype** archive, not literature figures. Replace them with your own
once you swap in real forecast/verification data, and re-run the script
before quoting a number in the pitch (exactly as the blueprint's KPI note
asks). `/verification/scorecard` returns these same numbers via
`headline_metrics_period` / `is_synthetic_prototype_result` fields so a
frontend can label them correctly without guessing.

## Running it

```bash
cd skillblend-backend
pip install -r requirements.txt   # or: pip install -r requirements.txt --break-system-packages

bash scripts/bootstrap.sh         # seeds data, trains, evaluates, precomputes replay cache (~1 min)

uvicorn app.main:app --reload --port 8000
# -> http://127.0.0.1:8000/docs
```

Judge-facing demo path (fully offline, matches Section 17):
```
GET /replay/events
GET /replay/events/{event_id}
```

Live pipeline (needs the archive from bootstrap.sh):
```
GET  /regions
GET  /forecast/available?region=kerala_western_ghats&variable=precipitation&lead_hours=72
GET  /forecast/raw?region=...&variable=...&valid_time=...&lead_hours=...
GET  /forecast/blend?region=...&variable=...&valid_time=...&lead_hours=...
POST /forecast/counterfactual   {"region":..., "regime_override":"depression"}
GET  /verification/scorecard?region=...
```

## Cost / hosting

Everything here runs on SQLite + LightGBM on CPU — no GPU, no paid API keys,
no external service. Matches the blueprint's own guardrail (§19): the
judging path never depends on paid infrastructure. Deploy the FastAPI app to
any free-tier host (Render/Railway free tier, Fly.io) and it will run as-is.

## Honest gaps vs. the full blueprint (next steps, not done here)

- Real GRIB2/NetCDF ingestion from NOMADS/ECMWF/IMERG (`cfgrib`/`xarray`) —
  stubbed out by the synthetic generator; the canonical schema is already
  ready for it.
- Zarr/PostGIS storage — SQLite stands in for the prototype; swap the
  `database.py` connection string for Postgres+PostGIS and mirror arrays to
  Zarr when moving past the MVP.
- The React/MapLibre dashboard itself (Section 16/17) — this repo is the
  API only; every response is already dashboard-shaped JSON
  (weights + SHAP drivers + trust signals + narrative) for a frontend to
  consume directly.
