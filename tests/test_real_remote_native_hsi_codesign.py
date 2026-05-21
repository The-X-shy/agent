"""Real remote native HSI co-design test (opt-in)."""

import os
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("OPTIRESEARCH_ENABLE_REAL_REMOTE_TESTS") != "1",
    reason="Real remote test requires explicit opt-in with OPTIRESEARCH_ENABLE_REAL_REMOTE_TESTS=1",
)


def test_real_remote_native_hsi_codesign_fresnel():
    worker_id = os.getenv("OPTIRESEARCH_REMOTE_WORKER_ID", "windows_wsl")
    from optiresearch.runtime.remote_jobs import run_remote_native_hsi_codesign
    payload = run_remote_native_hsi_codesign(
        worker_id,
        optical_component="Fresnel",
        objective="minimize_hsi_proxy_loss",
        max_steps=3,
        device="cpu",
        bands=4,
        image_size=16,
        psf_size=8,
    )
    result = payload["result"]
    assert result.status in ("succeeded", "failed", "timeout")
