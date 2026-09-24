import numpy as np
from app.blending.calibration import fit_quantile_map, apply_quantile_map, CAL_DIR

def test_quantile_map_roundtrip_on_identity_distribution():
    rng = np.random.default_rng(0)
    blend_vals = rng.normal(50, 10, 500)
    truth_vals = blend_vals.copy()  # perfect model -> mapping should be ~identity
    fit_quantile_map(blend_vals, truth_vals, "test_variable_identity")
    mapped = apply_quantile_map(52.0, "test_variable_identity")
    assert abs(mapped - 52.0) < 3.0  # allow interpolation slack

def test_unseen_variable_passes_through_unchanged():
    val = apply_quantile_map(77.0, "no_such_variable_ever_fit")
    assert val == 77.0
