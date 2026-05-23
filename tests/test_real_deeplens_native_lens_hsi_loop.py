"""Phase 33: Real DeepLens Native GeoLens HSI Loop (opt-in).

Requires: OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS=1
"""

import os
import pytest

from optiresearch.runtime.stable_native_lens_hsi_loop import (
    run_stable_native_lens_hsi_codesign,
)
from optiresearch.schemas.stable_native_lens_hsi import (
    StableNativeLensHSISpec, make_stable_lens_id,
)


@pytest.mark.skipif(
    not os.environ.get("OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS"),
    reason="Set OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS=1 for real DeepLens tests",
)
def test_real_deeplens_native_geolens_hsi():
    run_id = make_stable_lens_id("GeoLensCooke", "differentiable_linear")
    spec = StableNativeLensHSISpec(
        run_id=run_id,
        candidate="GeoLensCooke",
        reconstructor="differentiable_linear",
        max_steps=5,
        optical_lr=1e-6,
        recon_lr=1e-3,
        rollback_on_loss_increase=True,
        device="cpu",
        full_wave_optics=False,
        phase_to_fft_proxy_used=False,
    )
    result = run_stable_native_lens_hsi_codesign(spec)
    assert result.status == "succeeded", f"Failed: {result.error_code} {result.error_message}"
    assert result.full_wave_optics is False
    assert result.phase_to_fft_proxy_used is False
    assert result.deeplens_native_psf_path == "geolens.psf_geometric"
    assert result.evidence_level is not None
    assert result.reconstruction_loss_after is not None
    assert result.optical_gradient_norm_max is not None
