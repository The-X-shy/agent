"""Test real DeepLens parameterized PSF (opt-in only).

Requires:
  OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS=1
"""

import os

import pytest

from optiresearch.adapters.deeplens_parameterized_psf import DeepLensParameterizedPSFGenerator


@pytest.mark.skipif(
    os.getenv("OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS") != "1",
    reason="Real DeepLens test requires explicit opt-in.",
)
def test_real_deeplens_generator_initializes():
    gen = DeepLensParameterizedPSFGenerator()
    assert gen is not None


@pytest.mark.skipif(
    os.getenv("OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS") != "1",
    reason="Real DeepLens test requires explicit opt-in.",
)
def test_real_deeplens_supported_unsupported(tmp_path):
    gen = DeepLensParameterizedPSFGenerator()
    supported = gen.supported_variables()
    assert "surface_curvature" in supported
    assert "phase_mask_strength" in supported

    output_dir = tmp_path / "real_dl_psf"
    output_dir.mkdir()
    result = gen.generate_psf_cube(
        {"surface_curvature": 0.5, "chromatic_shift": 0.3, "depth_variation": 0.5},
        output_dir,
    )
    # Should return structured result (success or fallback)
    assert "status" in result
    assert "psf_cube" in result
    assert "fallback_used" in result
