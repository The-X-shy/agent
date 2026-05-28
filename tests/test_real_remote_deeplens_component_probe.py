"""Real remote DeepLens component probe test (opt-in only).

Requires:
    OPTIRESEARCH_ENABLE_REAL_REMOTE_TESTS=1
    OPTIRESEARCH_REMOTE_WORKER_ID=windows_wsl
"""

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("OPTIRESEARCH_ENABLE_REAL_REMOTE_TESTS") != "1",
    reason="Real remote test requires explicit opt-in with OPTIRESEARCH_ENABLE_REAL_REMOTE_TESTS=1",
)

WORKER_ID = os.getenv("OPTIRESEARCH_REMOTE_WORKER_ID", "windows_wsl")


class TestRealRemoteDeeplensComponentProbe:
    def test_remote_fresnel_component_probe(self):
        from optiresearch.runtime.remote_jobs import run_remote_deeplens_component_probe

        try:
            payload = run_remote_deeplens_component_probe(
                WORKER_ID,
                component="fresnel",
                objective="parameter_sanity_check",
                max_steps=5,
                device="cpu",
            )
        except Exception as exc:
            if any(kw in str(exc).lower() for kw in ("ssh", "connection", "worker")):
                pytest.skip(f"Remote worker unavailable: {exc}")
            raise

        result = payload["result"]
        assert result.job_id
        assert result.status in ("succeeded", "failed")

        # Check output files exist on local copy.
        output_dir = Path(result.local_output_dir) if result.local_output_dir else None
        if output_dir and output_dir.exists():
            result_path = output_dir / result.job_id / "result.json"
            if result_path.exists():
                data = json.loads(result_path.read_text())
                assert "component" in data or "surface_class" in data

    def test_remote_binary2phase_component_probe(self):
        from optiresearch.runtime.remote_jobs import run_remote_deeplens_component_probe

        try:
            payload = run_remote_deeplens_component_probe(
                WORKER_ID,
                component="binary2phase",
                objective="parameter_sanity_check",
                max_steps=5,
                device="cpu",
            )
        except Exception as exc:
            if any(kw in str(exc).lower() for kw in ("ssh", "connection", "worker")):
                pytest.skip(f"Remote worker unavailable: {exc}")
            raise

        result = payload["result"]
        assert result.status in ("succeeded", "failed")

    def test_remote_diffractive_candidate_probe(self):
        from optiresearch.runtime.remote_jobs import run_remote_deeplens_component_probe

        try:
            payload = run_remote_deeplens_component_probe(
                WORKER_ID,
                component="diffractive",
                objective="parameter_sanity_check",
                max_steps=3,
                device="cpu",
            )
        except Exception as exc:
            if any(kw in str(exc).lower() for kw in ("ssh", "connection", "worker")):
                pytest.skip(f"Remote worker unavailable: {exc}")
            raise

        result = payload["result"]
        assert result.status in ("succeeded", "failed")

    def test_remote_discover_components(self):
        from optiresearch.runtime.remote_jobs import run_remote_deeplens_component_discovery

        try:
            payload = run_remote_deeplens_component_discovery(
                WORKER_ID,
                components="fresnel,binary2phase,diffractive",
                device="cpu",
            )
        except Exception as exc:
            if any(kw in str(exc).lower() for kw in ("ssh", "connection", "worker")):
                pytest.skip(f"Remote worker unavailable: {exc}")
            raise

        result = payload["result"]
        assert result.status in ("succeeded", "failed")
