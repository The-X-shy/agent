"""Test parameterized PSF generator."""
import numpy as np
from optiresearch.adapters.parameterized_psf import (
    generate_parameterized_psf,
    compute_psf_metrics,
    optical_vars_to_dict,
)
from optiresearch.schemas.optimization import OpticalVariable


def test_generate_psf_returns_correct_shape():
    cube = generate_parameterized_psf(
        {"phase_mask_strength": 0.5, "doe_grating_period": 1.0, "surface_curvature": 0.5, "chromatic_shift": 0.3, "depth_variation": 0.5},
        depth_planes=5, wavelength_bands=31, psf_size=32,
    )
    assert cube.shape == (5, 31, 32, 32)
    assert np.all(np.isfinite(cube))
    assert np.all(cube >= 0)


def test_generate_psf_is_deterministic():
    vars_dict = {"phase_mask_strength": 0.5, "doe_grating_period": 1.0, "surface_curvature": 0.5, "chromatic_shift": 0.3, "depth_variation": 0.5}
    cube1 = generate_parameterized_psf(vars_dict, seed=42)
    cube2 = generate_parameterized_psf(vars_dict, seed=42)
    assert np.allclose(cube1, cube2)


def test_different_params_produce_different_psf():
    vars1 = {"phase_mask_strength": 0.0, "doe_grating_period": 0.5, "surface_curvature": 0.2, "chromatic_shift": 0.0, "depth_variation": 0.1}
    vars2 = {"phase_mask_strength": 1.0, "doe_grating_period": 2.0, "surface_curvature": 0.9, "chromatic_shift": 1.0, "depth_variation": 0.9}
    cube1 = generate_parameterized_psf(vars1, seed=42)
    cube2 = generate_parameterized_psf(vars2, seed=42)
    assert not np.allclose(cube1, cube2)


def test_compute_psf_metrics_returns_expected_keys():
    cube = generate_parameterized_psf({"phase_mask_strength": 0.5}, seed=42)
    metrics = compute_psf_metrics(cube)
    for key in ["depth_stability_score", "spectral_separability_score", "coding_strength", "band_condition_score"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_different_encoders_produce_distinct_metrics():
    """Different encoders should produce measurably different PSF optical metrics."""
    vars_dict = {"phase_mask_strength": 0.5, "depth_variation": 0.7, "chromatic_shift": 0.6}
    cube_conv = generate_parameterized_psf(vars_dict, encoder_type="conventional", seed=42)
    cube_cc_edof = generate_parameterized_psf(vars_dict, encoder_type="controlled_chromatic_edof", seed=42)

    metrics_conv = compute_psf_metrics(cube_conv)
    metrics_edof = compute_psf_metrics(cube_cc_edof)

    # At least one metric should differ significantly
    diffs = []
    for key in ["depth_stability_score", "spectral_separability_score", "coding_strength"]:
        diffs.append(abs(metrics_conv[key] - metrics_edof[key]))
    assert max(diffs) > 0.001, f"All metrics too similar: conv={metrics_conv}, edof={metrics_edof}"


def test_optical_vars_to_dict():
    variables = [
        OpticalVariable(name="phase_mask_strength", current_value=0.7),
        OpticalVariable(name="doe_grating_period", current_value=1.5),
    ]
    result = optical_vars_to_dict(variables)
    assert result["phase_mask_strength"] == 0.7
    assert result["doe_grating_period"] == 1.5
