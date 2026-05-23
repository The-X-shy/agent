"""Phase 32: Real DeepLens GeoLens geometric deep probe (opt-in).

Requires: OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS=1
"""

import os
import pytest

from optiresearch.runtime.lightweight_experiments import (
    run_deeplens_geolens_geometric_deep_probe,
)


@pytest.mark.skipif(
    not os.environ.get("OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS"),
    reason="Set OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS=1 for real DeepLens tests",
)
def test_real_deeplens_geometric_deep_probe():
    result = run_deeplens_geolens_geometric_deep_probe(
        backend_id="deeplens_geolens_geometric",
    )
    assert result.status == "succeeded", (
        f"Deep probe failed: {result.errors}"
    )
    payload = result.result_payload or {}
    assert payload.get("differentiable") is True
    assert payload.get("optical_gradient_norm", 0) > 0
    assert payload.get("parameters_changed") is True
    assert payload.get("deeplens_native_psf_path") == "geolens.psf_geometric"
    assert payload.get("full_wave_optics") is False
    assert payload.get("phase_to_fft_proxy_used") is False
    assert payload.get("evidence_level") == "native_lens_simulation"
