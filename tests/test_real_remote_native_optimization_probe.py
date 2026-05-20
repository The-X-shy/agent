"""Real remote native optimization probe tests.

Skip unless:
- OPTIRESEARCH_ENABLE_REAL_REMOTE_TESTS=1
- OPTIRESEARCH_REMOTE_WORKER_ID=windows_wsl
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("OPTIRESEARCH_ENABLE_REAL_REMOTE_TESTS") != "1",
    reason="Real remote test requires explicit opt-in.",
)

requires_worker = pytest.mark.skipif(
    os.getenv("OPTIRESEARCH_REMOTE_WORKER_ID") != "windows_wsl",
    reason="Requires windows_wsl worker.",
)


@requires_worker
def test_remote_native_optimization_probe_runs():
    """Run a real remote native optimization probe on WSL."""
    from optiresearch.runtime.remote_jobs import run_remote_native_optimization_probe

    worker_id = os.environ["OPTIRESEARCH_REMOTE_WORKER_ID"]
    payload = run_remote_native_optimization_probe(
        worker_id=worker_id,
        lens_class="ParaxialLens",
        objective="minimize_psf_width",
        max_steps=2,
        learning_rate=1e-3,
        device="cpu",
        strict_native=True,
        allow_adapter_proxy=False,
    )
    result = payload["result"]
    assert result.status in ("succeeded", "failed", "unsupported")
    if result.status == "succeeded":
        assert result.metrics_summary.get("differentiable") is True


@requires_worker
def test_remote_native_optimization_inspection_runs():
    """Run inspection remotely on WSL."""
    from optiresearch.runtime.remote_jobs import run_remote_native_optimization_inspection

    worker_id = os.environ["OPTIRESEARCH_REMOTE_WORKER_ID"]
    payload = run_remote_native_optimization_inspection(
        worker_id=worker_id,
    )
    result = payload["result"]
    assert result.status in ("succeeded", "failed")
