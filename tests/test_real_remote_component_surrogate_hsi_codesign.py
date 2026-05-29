"""Real remote component surrogate HSI co-design test (opt-in only)."""

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("OPTIRESEARCH_ENABLE_REAL_REMOTE_TESTS") != "1",
    reason="Real remote test requires OPTIRESEARCH_ENABLE_REAL_REMOTE_TESTS=1",
)

WORKER_ID = os.getenv("OPTIRESEARCH_REMOTE_WORKER_ID", "windows_wsl")


@pytest.mark.parametrize("component", ["fresnel", "binary2phase"])
def test_real_remote_component_surrogate_hsi_codesign(component):
    from optiresearch.runtime.remote_jobs import run_remote_component_surrogate_hsi_codesign

    try:
        payload = run_remote_component_surrogate_hsi_codesign(
            WORKER_ID,
            component=component,
            dataset="synthetic",
            steps=3,
            device="cpu",
        )
    except Exception as exc:
        if any(kw in str(exc).lower() for kw in ("ssh", "connection", "worker")):
            pytest.skip(f"Remote worker unavailable: {exc}")
        raise

    result = payload["result"]
    assert result.status in ("succeeded", "failed")
    output_dir = Path(result.local_output_dir or "")
    result_path = output_dir / result.job_id / "outputs"
    candidates = list(result_path.rglob("result.json")) if result_path.exists() else []
    if candidates:
        data = json.loads(candidates[0].read_text())
        assert data["component_type"] == component
        assert data["psf_requires_grad"] is True
        assert data["loss_requires_grad"] is True
        assert data["component_grad_norm_max"] > 0
        assert data["component_parameter_changed"] is True
        assert data["claim_ceiling"] == "component_surrogate_hsi_codesign"
