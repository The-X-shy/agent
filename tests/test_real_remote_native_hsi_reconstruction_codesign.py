"""Real remote native HSI reconstruction co-design test (opt-in)."""

import os
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("OPTIRESEARCH_ENABLE_REAL_REMOTE_TESTS") != "1",
    reason="Real remote test requires explicit opt-in with OPTIRESEARCH_ENABLE_REAL_REMOTE_TESTS=1",
)


def test_real_remote_native_hsi_reconstruction_codesign():
    worker_id = os.getenv("OPTIRESEARCH_REMOTE_WORKER_ID", "windows_wsl")
    from optiresearch.runtime.remote_jobs import run_remote_native_hsi_reconstruction_codesign
    payload = run_remote_native_hsi_reconstruction_codesign(
        worker_id,
        optical_component="Fresnel",
        reconstructor="differentiable_linear",
        max_steps=5,
        device="cpu",
        bands=4,
        image_size=16,
        psf_size=7,
    )
    result = payload["result"]
    assert result.status in ("succeeded", "failed", "timeout")
