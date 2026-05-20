"""Test DeepLensParameterizedPSFGenerator variable mapping (no real DeepLens needed)."""
from optiresearch.adapters.deeplens_parameterized_psf import DeepLensParameterizedPSFGenerator


def test_generator_initializes():
    gen = DeepLensParameterizedPSFGenerator()
    assert gen is not None
    assert isinstance(gen.deeplens_available, bool)


def test_supported_variables_returns_dict():
    gen = DeepLensParameterizedPSFGenerator()
    supported = gen.supported_variables()

    assert "surface_curvature" in supported
    assert supported["surface_curvature"]["supported"] is True
    assert supported["phase_mask_strength"]["supported"] is False
    assert supported["doe_grating_period"]["supported"] is False


def test_unsupported_variables_reports_phase_mask_and_doe():
    gen = DeepLensParameterizedPSFGenerator()
    optical_vars = {
        "phase_mask_strength": 0.7,
        "doe_grating_period": 1.5,
        "surface_curvature": 0.5,
        "chromatic_shift": 0.3,
        "depth_variation": 0.5,
    }
    unsupported = gen.unsupported_variables(optical_vars)

    unsupported_names = [u["variable"] for u in unsupported]
    assert "phase_mask_strength" in unsupported_names
    assert "doe_grating_period" in unsupported_names
    assert "surface_curvature" not in unsupported_names


def test_map_variables_marks_unsupported():
    gen = DeepLensParameterizedPSFGenerator()
    optical_vars = {
        "phase_mask_strength": 0.7,
        "doe_grating_period": 1.5,
        "surface_curvature": 0.5,
        "chromatic_shift": 0.3,
        "depth_variation": 0.5,
    }
    mapping = gen.map_variables_to_deeplens_config(optical_vars)

    assert "deeplens_config" in mapping
    assert mapping["differentiable"] is False
    assert mapping["native_parameter_update"] is False
    assert len(mapping["unsupported_variables"]) == 2

    cfg = mapping["deeplens_config"]
    assert "optical_parameters" in cfg
    assert "f_number" in cfg["optical_parameters"]


def test_map_variables_surface_curvature_affects_f_number():
    gen = DeepLensParameterizedPSFGenerator()

    low_curv = gen.map_variables_to_deeplens_config({"surface_curvature": 0.0})
    high_curv = gen.map_variables_to_deeplens_config({"surface_curvature": 1.0})

    low_fnum = low_curv["deeplens_config"]["optical_parameters"]["f_number"]
    high_fnum = high_curv["deeplens_config"]["optical_parameters"]["f_number"]

    # Higher curvature → lower f_number
    assert high_fnum < low_fnum


def test_generate_psf_cube_without_deeplens_returns_fallback(tmp_path):
    gen = DeepLensParameterizedPSFGenerator()
    output_dir = tmp_path / "psf_output"
    output_dir.mkdir()

    result = gen.generate_psf_cube(
        {"surface_curvature": 0.5, "chromatic_shift": 0.3, "depth_variation": 0.5},
        output_dir,
    )

    assert result["status"] in ("fallback", "succeeded", "partial")
    if result["fallback_used"]:
        assert result["psf_cube"] is None or result["fallback_reason"] is not None
