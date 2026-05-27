"""Real remote DeepLens diagnostic tests (opt-in).

Requires:
  OPTIRESEARCH_ENABLE_REAL_REMOTE_TESTS=1
  OPTIRESEARCH_REMOTE_WORKER_ID=windows_wsl
  WSL connection with DeepLens + cooke.json available
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("OPTIRESEARCH_ENABLE_REAL_REMOTE_TESTS") != "1",
    reason="Real remote test requires explicit opt-in with OPTIRESEARCH_ENABLE_REAL_REMOTE_TESTS=1",
)


def _worker_id() -> str:
    return os.getenv("OPTIRESEARCH_REMOTE_WORKER_ID", "windows_wsl")


class TestRealRemoteLensResolution:
    def test_remote_resolve_lens_file_auto_cooke(self):
        from optiresearch.runtime.remote_jobs import run_remote_resolve_lens_file

        payload = run_remote_resolve_lens_file(
            _worker_id(),
            lens_file="auto:cooke",
            backend_id="deeplens_geolens_geometric",
        )
        result = payload["result"]
        assert result.status == "succeeded", f"Lens resolution failed: {result.error_code}"
        metrics = result.metrics_summary
        assert metrics.get("requested_lens_file") == "auto:cooke"
        resolved = metrics.get("resolved_path") or metrics.get("resolved_lens_file")
        assert resolved, "resolved_lens_file must not be empty"
        assert metrics.get("exists") is True
        assert metrics.get("source"), "lens_resolution_source must not be empty"
        assert isinstance(metrics.get("checked_paths", []), list)
        assert len(metrics.get("checked_paths", [])) > 0
        assert metrics.get("error_code") is None


class TestRealRemoteTrainableParameterInspection:
    def test_remote_trainable_parameter_inspection(self):
        from optiresearch.runtime.remote_jobs import run_remote_deeplens_trainable_parameter_inspection

        payload = run_remote_deeplens_trainable_parameter_inspection(
            _worker_id(),
            lens_file="auto:cooke",
            device="cpu",
        )
        result = payload["result"]
        assert result.status == "succeeded", f"Trainable parameter inspection failed: {result.error_code}"
        metrics = result.metrics_summary
        assert metrics.get("resolved_lens_file"), "resolved_lens_file must not be empty"
        assert "parameter_count" in metrics
        assert "trainable_count" in metrics
        error_code = result.error_code
        if error_code:
            assert "LENS_FILE_NOT_FOUND" not in error_code.upper(), f"Lens file should be resolved: {error_code}"


class TestRealRemoteAutogradAudit:
    def test_remote_autograd_audit(self):
        from optiresearch.runtime.remote_jobs import run_remote_deeplens_autograd_audit

        payload = run_remote_deeplens_autograd_audit(
            _worker_id(),
            lens_file="auto:cooke",
            device="cpu",
        )
        result = payload["result"]
        assert result.status == "succeeded", f"Autograd audit failed: {result.error_code}"
        metrics = result.metrics_summary
        assert metrics.get("resolved_lens_file"), "resolved_lens_file must not be empty"
        assert "graph_connected" in metrics, "graph_connected must be present"
        assert "psf_requires_grad" in metrics, "psf_requires_grad must be present"
        assert "loss_requires_grad" in metrics, "loss_requires_grad must be present"
        assert "candidate_update_changes_parameter" in metrics
        assert "detach_suspected" in metrics
        error_code = result.error_code
        if error_code:
            assert "LENS_FILE_NOT_FOUND" not in error_code.upper(), f"Lens file should be resolved: {error_code}"
