"""
scripts/seed_and_train.py

One-shot setup script implementing blueprint Section 13's build order,
steps 2-13, over the synthetic data source (Section 4's live-ingestion
steps 3 are simulated - see app/data/synthetic_generator.py):

  1. Generate the synthetic historical dataset (stands in for cached
     GFS/IFS/AIFS + ERA5/IMERG reference windows).
  2. Temporal holdout split (train / holdout by date - never shuffled,
     per Section 18's leakage guardrail).
  3. Build the offline Historical Skill Engine table on TRAIN only.
  4. Train the adaptive blending meta-model on TRAIN only.
  5. Backtest the trained blend in-sample on TRAIN to fit the
     regime-aware quantile-mapping bias correctors.
  6. Fit isotonic exceedance-probability calibrators on the
     bias-corrected TRAIN backtest.
  7. Train the Forecast Bust Probability classifier on TRAIN.
  8. Score everything out-of-sample on HOLDOUT to produce the
     blueprint's headline verification numbers (SkillBlend vs best
     single model CSI, Section 21).
  9. Precompute a handful of Counterfactual Bust Replay events
     (Section 11) so the demo can run fully offline.

Run:
    python -m scripts.seed_and_train
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from app.config import (
    BLEND_BASELINE_PATH,
    BUST_BASELINE_PATH,
    CALIBRATORS_PATH,
    EXTREME_THRESHOLDS,
    HISTORY_DAYS,
    PILOT_ZONES,
    REPLAY_DIR,
    SOURCES,
    VARIABLES,
    VERIFICATION_SUMMARY_PATH,
)
from app.data import storage
from app.data.synthetic_generator import generate_historical_dataset
from app.engine import blend as blend_engine
from app.engine import calibration as calib_engine
from app.engine import explain as explain_engine
from app.engine import regime as regime_engine
from app.engine import skill as skill_engine
from app.engine import trust as trust_engine
from app.engine import verification as verification_engine

INSTANCE_KEYS = ["run_time", "region", "variable", "lead_hours"]
HOLDOUT_FRACTION = 0.15


def log(msg: str) -> None:
    print(f"[seed_and_train] {msg}")


def _split_train_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.sort(df["run_time"].unique())
    cutoff = dates[int(len(dates) * (1 - HOLDOUT_FRACTION))]
    train_df = df[df["run_time"] < cutoff].copy()
    holdout_df = df[df["run_time"] >= cutoff].copy()
    log(f"temporal split: train up to {pd.Timestamp(cutoff).date()}, "
        f"{len(train_df)} train rows / {len(holdout_df)} holdout rows")
    return train_df, holdout_df


def _attach_historical_skill(df: pd.DataFrame, skill_table: pd.DataFrame) -> pd.DataFrame:
    """Vectorized merge-based lookup, with a constant-fallback for any
    (rare) context combination absent from the training skill table."""
    group_cols = ["model", "region", "variable", "lead_hours", "season", "regime"]
    merged = df.merge(
        skill_table[group_cols + ["rmse", "csi"]],
        on=group_cols,
        how="left",
    )
    rmse_scale = merged["variable"].map(skill_engine.rmse_scale)
    rmse_score = np.exp(-merged["rmse"] / rmse_scale)
    csi = merged["csi"]
    historical_skill = np.where(
        csi.notna(), 0.6 * rmse_score + 0.4 * csi, rmse_score
    )
    merged["historical_skill"] = np.where(merged["rmse"].isna(), 0.5, historical_skill)
    return merged


def _softmax_series(s: pd.Series) -> pd.Series:
    scores = -s.to_numpy(dtype=float)
    scores = scores - scores.max()
    w = np.exp(scores)
    w = w / w.sum()
    return pd.Series(w, index=s.index)


def _regime_stability_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the auditable rule-based regime-stability signal per
    (region, run_time, lead_hours) instance, exactly as the live
    pipeline would (see services/pipeline._regime_signal), so the bust
    model trains on features consistent with what it will see in
    production.
    """
    precip = df[df["variable"] == "precipitation"].groupby(["region", "run_time", "lead_hours"], observed=True)[
        "forecast_value"
    ].agg(mean_precip="mean", spread_precip="std")
    wind = df[df["variable"] == "wind_speed"].groupby(["region", "run_time", "lead_hours"], observed=True)[
        "forecast_value"
    ].agg(mean_wind="mean")
    combo = precip.join(wind, how="outer").reset_index().fillna(0.0)

    stabilities = []
    for _, row in combo.iterrows():
        zone = PILOT_ZONES[row["region"]]
        signal = regime_engine.RegimeSignal(
            mean_precip_forecast=row["mean_precip"],
            mean_wind_forecast=row["mean_wind"],
            source_spread_precip=row["spread_precip"],
            month=int(pd.Timestamp(row["run_time"]).month),
            coastal=zone.coastal,
            wd_sensitivity=zone.wd_sensitivity,
            monsoon_sensitivity=zone.monsoon_sensitivity,
        )
        probs = regime_engine.classify_regime(signal)
        stabilities.append(regime_engine.regime_stability(probs))
    combo["regime_stability"] = stabilities
    return combo[["region", "run_time", "lead_hours", "regime_stability"]]


def _backtest_instances(df_with_skill: pd.DataFrame, blend_model: blend_engine.BlendModel) -> pd.DataFrame:
    """
    Batch-score every row with the trained blend model, softmax-combine
    per instance, and collapse to one row per (run_time, region,
    variable, lead_hours) instance with blend_value / actual_value /
    disagreement / historical_skill_avg.
    """
    features = blend_engine.build_feature_frame(df_with_skill)
    df_with_skill = df_with_skill.copy()
    df_with_skill["expected_error"] = blend_model.predict_expected_error(features)
    df_with_skill["bl_weight"] = df_with_skill.groupby(INSTANCE_KEYS)["expected_error"].transform(_softmax_series)
    df_with_skill["weighted_value"] = df_with_skill["forecast_value"] * df_with_skill["bl_weight"]
    df_with_skill["weighted_skill"] = df_with_skill["historical_skill"] * df_with_skill["bl_weight"]

    agg = df_with_skill.groupby(INSTANCE_KEYS + ["season", "regime"], observed=True).agg(
        blend_value=("weighted_value", "sum"),
        actual_value=("actual_value", "first"),
        disagreement=("forecast_value", "std"),
        historical_skill_avg=("weighted_skill", "sum"),
    ).reset_index()
    agg["disagreement"] = agg["disagreement"].fillna(0.0)
    return agg


def main() -> None:
    t0 = time.time()

    log(f"generating {HISTORY_DAYS}-day synthetic historical dataset ...")
    historical = generate_historical_dataset(days=HISTORY_DAYS)
    storage.save_historical(historical)
    log(f"saved {len(historical)} rows")

    train_df, holdout_df = _split_train_holdout(historical)

    log("computing historical skill table (train only) ...")
    skill_table = skill_engine.compute_skill_table(train_df)
    storage.save_skill_table(skill_table)
    log(f"skill table: {len(skill_table)} context rows")

    log("attaching historical-skill feature + training blend meta-model ...")
    train_with_skill = _attach_historical_skill(train_df, skill_table)
    blend_train_frame = blend_engine.build_feature_frame(train_with_skill)
    blend_train_frame["abs_error"] = train_with_skill["abs_error"].to_numpy()
    blend_model = blend_engine.BlendModel.train(blend_train_frame)
    blend_model.save()
    blend_baseline = explain_engine.compute_baseline_row(blend_train_frame, blend_model.feature_columns)
    joblib.dump(blend_baseline, BLEND_BASELINE_PATH)
    log(f"blend model trained (backend={blend_model.backend})")

    log("backtesting blend on TRAIN to fit bias correctors + regime stability ...")
    train_instances = _backtest_instances(train_with_skill, blend_model)
    stability_lookup = _regime_stability_lookup(train_df)
    train_instances = train_instances.merge(stability_lookup, on=["region", "run_time", "lead_hours"], how="left")
    train_instances["regime_stability"] = train_instances["regime_stability"].fillna(0.5)
    train_instances["data_quality"] = 1.0  # all 3 sources always present in the synthetic feed

    bias_correctors = {}
    for region in PILOT_ZONES:
        for variable in VARIABLES:
            bias_correctors[(region, variable)] = calib_engine.RegimeAwareBiasCorrector.fit(
                train_instances, region, variable
            )

    train_instances["corrected_value"] = train_instances.apply(
        lambda r: bias_correctors[(r["region"], r["variable"])].transform(r["blend_value"], r["regime"]), axis=1
    )
    train_instances["corrected_value"] = train_instances.apply(
        lambda r: max(0.0, r["corrected_value"]) if r["variable"] in ("precipitation", "wind_speed") else r["corrected_value"],
        axis=1,
    )

    log("fitting isotonic exceedance-probability calibrators ...")
    exceedance_calibrators = {}
    for region in PILOT_ZONES:
        for variable in VARIABLES:
            subset = train_instances[(train_instances["region"] == region) & (train_instances["variable"] == variable)]
            threshold = EXTREME_THRESHOLDS[variable]
            raw_scores = subset.apply(
                lambda r: calib_engine.raw_exceedance_score(r["corrected_value"], threshold, max(r["disagreement"], 0.5)),
                axis=1,
            ).to_numpy()
            observed = (subset["actual_value"] >= threshold).astype(int).to_numpy()
            if observed.sum() >= 5 and (len(observed) - observed.sum()) >= 5:
                exceedance_calibrators[(region, variable)] = calib_engine.ExceedanceCalibrator.fit(raw_scores, observed)

    joblib.dump(
        {"bias_correctors": bias_correctors, "exceedance_calibrators": exceedance_calibrators},
        CALIBRATORS_PATH,
    )
    log(f"saved {len(bias_correctors)} bias correctors, {len(exceedance_calibrators)} exceedance calibrators")

    log("training Forecast Bust Probability classifier ...")
    train_instances["disagreement_norm"] = train_instances.apply(
        lambda r: trust_engine.normalize_disagreement(r["disagreement"], r["variable"]), axis=1
    )
    train_instances["bust"] = train_instances.apply(
        lambda r: abs(r["corrected_value"] - r["actual_value"]) > trust_engine_bust_threshold(r["variable"]), axis=1
    )
    bust_model = trust_engine.BustModel.train(train_instances)
    bust_model.save()
    bust_baseline = explain_engine.compute_baseline_row(train_instances, trust_engine.BUST_FEATURE_COLUMNS)
    joblib.dump(bust_baseline, BUST_BASELINE_PATH)
    log(f"bust model trained (backend={bust_model.backend}); "
        f"base rate={train_instances['bust'].mean():.3f}")

    log("scoring HOLDOUT out-of-sample for verification ...")
    holdout_with_skill = _attach_historical_skill(holdout_df, skill_table)
    holdout_instances = _backtest_instances(holdout_with_skill, blend_model)
    holdout_stability_lookup = _regime_stability_lookup(holdout_df)
    holdout_instances = holdout_instances.merge(
        holdout_stability_lookup, on=["region", "run_time", "lead_hours"], how="left"
    )
    holdout_instances["regime_stability"] = holdout_instances["regime_stability"].fillna(0.5)
    holdout_instances["corrected_value"] = holdout_instances.apply(
        lambda r: bias_correctors[(r["region"], r["variable"])].transform(r["blend_value"], r["regime"]), axis=1
    )
    holdout_instances["corrected_value"] = holdout_instances.apply(
        lambda r: max(0.0, r["corrected_value"]) if r["variable"] in ("precipitation", "wind_speed") else r["corrected_value"],
        axis=1,
    )

    verification_results = []
    for region in PILOT_ZONES:
        for variable in VARIABLES:
            blend_subset = holdout_instances[
                (holdout_instances["region"] == region) & (holdout_instances["variable"] == variable)
            ][["corrected_value", "actual_value"]].rename(columns={"corrected_value": "blend_value"})
            if blend_subset.empty:
                continue
            single_model_frames = {
                m: holdout_df[
                    (holdout_df["region"] == region) & (holdout_df["variable"] == variable) & (holdout_df["model"] == m)
                ][["forecast_value", "actual_value"]]
                for m in SOURCES
            }
            verification_results.append(
                verification_engine.verify_region_variable(blend_subset, single_model_frames, region, variable)
            )
    VERIFICATION_SUMMARY_PATH.write_text(json.dumps(verification_results, indent=2))
    log("verification summary:")
    for r in verification_results:
        rel = r["relative_csi_improvement"]
        rel_str = f"{rel:+.1%}" if rel is not None else "n/a"
        log(
            f"  {r['region']}/{r['variable']}: SkillBlend CSI={r['skillblend_csi']} "
            f"vs best single ({r['best_single_model']})={r['best_single_model_csi']} "
            f"-> {rel_str} (target {r['target_relative_csi_improvement']:+.0%})"
        )

    log("building Counterfactual Bust Replay cases ...")
    _build_replay_events(holdout_df, holdout_instances, skill_table)

    log(f"done in {time.time() - t0:.1f}s")


def trust_engine_bust_threshold(variable: str) -> float:
    from app.config import BUST_ERROR_THRESHOLDS

    return BUST_ERROR_THRESHOLDS[variable]


def _best_operator_default_model(skill_table: pd.DataFrame, region: str, variable: str) -> str:
    subset = skill_table[(skill_table["region"] == region) & (skill_table["variable"] == variable)]
    if subset.empty:
        return SOURCES[0]
    scored = subset.groupby("model", observed=True).apply(
        lambda g: np.average(g["rmse"], weights=g["n_obs"])
    )
    return str(scored.idxmin())


def _build_replay_events(holdout_df: pd.DataFrame, holdout_instances: pd.DataFrame, skill_table: pd.DataFrame) -> None:
    precip_extreme = holdout_instances[
        (holdout_instances["variable"] == "precipitation")
        & (holdout_instances["actual_value"] >= EXTREME_THRESHOLDS["precipitation"])
    ].copy()
    if precip_extreme.empty:
        log("no extreme-rainfall holdout instances found - skipping replay generation")
        storage.save_replay_index([])
        return

    precip_extreme["naive_average"] = np.nan
    events_index = []
    picked = (
        precip_extreme.sort_values("actual_value", ascending=False)
        .groupby("region", observed=True)
        .head(2)
        .reset_index(drop=True)
    )

    for i, row in picked.iterrows():
        region, run_time, lead_hours = row["region"], row["run_time"], row["lead_hours"]
        raw_rows = holdout_df[
            (holdout_df["region"] == region)
            & (holdout_df["variable"] == "precipitation")
            & (holdout_df["run_time"] == run_time)
            & (holdout_df["lead_hours"] == lead_hours)
        ]
        raw_sources = [
            {"model": r["model"], "forecast_value": round(float(r["forecast_value"]), 2), "weight": None, "historical_skill": None}
            for _, r in raw_rows.iterrows()
        ]
        naive_average = float(raw_rows["forecast_value"].mean())
        default_model = _best_operator_default_model(skill_table, region, "precipitation")
        default_value = float(raw_rows[raw_rows["model"] == default_model]["forecast_value"].iloc[0])

        actual = float(row["actual_value"])
        skillblend_value = float(row["corrected_value"])

        naive_error = abs(naive_average - actual)
        single_error = abs(default_value - actual)
        skillblend_error = abs(skillblend_value - actual)

        event_id = f"EVT-{region}-{pd.Timestamp(run_time).strftime('%Y%m%d')}-L{lead_hours}"
        narrative = [
            f"Raw sources disagreed materially: "
            + ", ".join(f"{s['model']}={s['forecast_value']:.1f}mm" for s in raw_sources)
            + f" for {PILOT_ZONES[region].name} at +{lead_hours}h lead.",
            f"The rule-based regime classifier read this as '{row['regime']}' "
            f"(context stability {row['regime_stability']:.0%}).",
            f"A naive equal-weight average would have given {naive_average:.1f}mm "
            f"({naive_error:.1f}mm error against the verified {actual:.1f}mm).",
            f"Always trusting {default_model} alone (its best overall historical RMSE for this region/variable) "
            f"would have given {default_value:.1f}mm ({single_error:.1f}mm error).",
            f"SkillBlend's context-aware blend gave {skillblend_value:.1f}mm "
            f"({skillblend_error:.1f}mm error) after bias correction.",
        ]
        if skillblend_error < min(naive_error, single_error):
            headline = f"SkillBlend cut the error vs. both the naive average and the default single model."
        elif skillblend_error < naive_error:
            headline = "SkillBlend beat the naive average but not the single best model on this particular case."
        else:
            headline = "A challenging case where SkillBlend did not clearly beat the simple baselines - useful for honest reporting."

        detail = {
            "event_id": event_id,
            "region": region,
            "variable": "precipitation",
            "unit": "mm/24h",
            "valid_time": pd.Timestamp(run_time) + pd.Timedelta(hours=int(lead_hours)),
            "lead_hours": int(lead_hours),
            "label": f"{PILOT_ZONES[region].name} - {pd.Timestamp(run_time).date()}",
            "headline": headline,
            "raw_sources": raw_sources,
            "naive_average": round(naive_average, 2),
            "single_model_choice": {"model": default_model, "forecast_value": round(default_value, 2)},
            "skillblend_blend": round(skillblend_value, 2),
            "reference_value": round(actual, 2),
            "naive_average_error": round(naive_error, 2),
            "single_model_error": round(single_error, 2),
            "skillblend_error": round(skillblend_error, 2),
            "narrative": narrative,
        }
        storage.save_replay_event(event_id, detail, REPLAY_DIR)
        events_index.append(
            {
                "event_id": event_id,
                "region": region,
                "variable": "precipitation",
                "valid_time": detail["valid_time"],
                "label": detail["label"],
                "headline": headline,
            }
        )

    storage.save_replay_index(events_index)
    log(f"cached {len(events_index)} replay events")


if __name__ == "__main__":
    main()
