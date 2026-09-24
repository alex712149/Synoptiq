"""
Held-out temporal evaluation (Section 13 step 15, Section 21 headline proof
point) — LEAKAGE-FREE VERSION.

Everything used here was frozen before this script runs: the per-source
meta-models and bust classifiers (trained on TRAIN only), the calibration
(fit on VALIDATION only, using the frozen meta-model's own predictions), and
the skill table (built from TRAIN+VALIDATION only). This script itself only
ever reads TEST-split rows, and the "historical_skill_expanding"/
"climatological_anomaly_norm" features those rows carry were computed via
groupby().expanding().shift(1) — i.e. every TEST row's features are built
only from what was already known strictly before that row's own valid_time,
never from the test outcome being scored or from anything after it.

Writes artifacts/blend_eval_<region>_<variable>.json, which
/verification/scorecard reads — so the API only ever reports a number this
script actually measured, never a placeholder.
"""
import sys, pathlib, json, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from app.database import SessionLocal
from app.data_prep import prepare_variable_frame, build_batch_blend_inputs
from app.blending.train_meta_model import get_global_split_boundaries
from app.blending.blend import batch_blend
from app.blending.calibration import apply_quantile_map
from app.skill_engine import _csi_pod_far
from app.config import PILOT_ZONES, ARTIFACTS_DIR, THRESHOLDS


def evaluate(db, region: str, val_start, test_start, variable: str = "precipitation"):
    df = prepare_variable_frame(db, variable, val_start, test_start)
    if df.empty:
        return None
    test_df = df[(df["split"] == "test") & (df["region"] == region)]
    if test_df.empty:
        return None

    feats_by_model, values_by_model, truth, context_meta = build_batch_blend_inputs(test_df, variable)
    if not feats_by_model or len(truth) < 10:
        return None

    fallback_skill = {m: context_meta["historical_skill_expanding"].to_numpy() for m in feats_by_model}
    blended_raw, _, source = batch_blend(variable, feats_by_model, values_by_model, fallback_skill)
    blended_calibrated = np.array([apply_quantile_map(v, variable, region=region) for v in blended_raw])

    thresh = THRESHOLDS[variable]["heavy"] if variable == "precipitation" else None
    result = {
        "region": region, "variable": variable, "n_test_contexts": int(len(truth)),
        "weight_source": source, "split_boundaries": {"val_start": str(val_start), "test_start": str(test_start)},
    }

    if variable == "precipitation":
        skillblend_csi, _, _ = _csi_pod_far(blended_calibrated, truth, thresh)
        best_single = 0.0
        for m, preds in values_by_model.items():
            csi_m, _, _ = _csi_pod_far(preds, truth, thresh)
            if not np.isnan(csi_m):
                best_single = max(best_single, csi_m)
        skillblend_csi = 0.0 if np.isnan(skillblend_csi) else float(skillblend_csi)
        rel_improve = ((skillblend_csi - best_single) / best_single * 100.0) if best_single > 0 else 0.0
        result.update({
            "skillblend_csi": round(skillblend_csi, 4),
            "best_single_model_csi": round(float(best_single), 4),
            "relative_csi_improvement_pct": round(float(rel_improve), 2),
            "fss_metric_type": "not applicable at this threshold summary (see /verification/scorecard for FSS-proxy rows)",
        })
    else:
        rmse_blend = float(np.sqrt(np.mean((blended_calibrated - truth) ** 2)))
        best_single_rmse = min(
            float(np.sqrt(np.mean((preds - truth) ** 2))) for preds in values_by_model.values()
        )
        rel_improve = ((best_single_rmse - rmse_blend) / best_single_rmse * 100.0) if best_single_rmse > 0 else 0.0
        result.update({
            "skillblend_rmse": round(rmse_blend, 3),
            "best_single_model_rmse": round(best_single_rmse, 3),
            "relative_rmse_improvement_pct": round(rel_improve, 2),
        })

    out_path = ARTIFACTS_DIR / f"blend_eval_{region}_{variable}.json"
    out_path.write_text(json.dumps(result, indent=2))
    return result


def main():
    t0 = time.time()
    db = SessionLocal()
    val_start, test_start = get_global_split_boundaries(db)
    print(f"Evaluating on TEST split only (valid_time >= {test_start}). "
          f"SYNTHETIC PROTOTYPE RESULTS — not real forecast skill.")
    for region in PILOT_ZONES:
        r = evaluate(db, region, val_start, test_start, "precipitation")
        if r:
            print(f"{region} [precipitation]: SkillBlend CSI@50mm={r['skillblend_csi']} vs best single "
                  f"{r['best_single_model_csi']}  ({r['relative_csi_improvement_pct']:+.1f}% relative, "
                  f"weight_source={r['weight_source']}, n={r['n_test_contexts']})")
        else:
            print(f"{region} [precipitation]: not enough held-out test samples to evaluate.")
    print(f"[evaluation time: {time.time() - t0:.1f}s]")
    db.close()


if __name__ == "__main__":
    main()
