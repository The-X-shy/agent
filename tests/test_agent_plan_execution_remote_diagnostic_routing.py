"""Test remote diagnostic routing in AgentPlanExecutionLoop."""

import pytest

from optiresearch.runtime.agent_plan_execution_loop import (
    _is_diagnostic_design,
    _execute_remote_diagnostic_design,
)
from optiresearch.agents.experiment_design_generator import ExperimentDesignCandidate


def _design(design_id, task_type="diagnostic_probe", backend_id="deeplens_geolens_geometric"):
    return ExperimentDesignCandidate(
        design_id=design_id, objective="test", backend_id=backend_id,
        task_type=task_type, required_skills=[],
    )


class TestIsDiagnosticDesign:
    def test_autograd_graph_audit_design(self):
        assert _is_diagnostic_design(_design("autograd_graph_audit_design", "autograd_audit"))

    def test_verify_trainable_design(self):
        assert _is_diagnostic_design(_design("verify_trainable_parameters_design", "backend_probe"))

    def test_curriculum_probe_design(self):
        assert _is_diagnostic_design(_design("deeplens_curriculum_probe_design", "curriculum_probe"))

    def test_regularized_probe_design(self):
        assert _is_diagnostic_design(_design("deeplens_regularized_probe_design", "regularized_probe"))

    def test_component_first_design(self):
        assert _is_diagnostic_design(_design("component_level_geolens_probe_design", "backend_probe"))

    def test_surface_freeze_design(self):
        assert _is_diagnostic_design(_design("surface_freeze_unfreeze_probe_design", "backend_probe"))

    def test_non_diagnostic_design(self):
        d = _design("remote_native_geolens_validation", "native_lens_simulation_codesign")
        assert not _is_diagnostic_design(d)


class TestRemoteDiagnosticDesignMapping:
    def test_autograd_design_maps_to_diagnostic_evidence(self):
        """Verify the design_id routing produces diagnostic_evidence even without WSL connection."""
        d = _design("autograd_graph_audit_design", "autograd_audit")
        try:
            result = _execute_remote_diagnostic_design(d, "windows_wsl")
        except Exception as e:
            if "ssh" in str(e).lower() or "connection" in str(e).lower():
                pytest.skip(f"SSH connection unavailable: {e}")
            raise
        assert result["design_id"] == "autograd_graph_audit_design"
        assert result["execution_target"] == "remote_wsl"
        assert "remote_job_id" in result or result.get("status") in ("failed", "completed")

    def test_verify_trainable_design_result_structure(self):
        d = _design("verify_trainable_parameters_design", "backend_probe")
        try:
            result = _execute_remote_diagnostic_design(d, "windows_wsl")
        except Exception as e:
            if "ssh" in str(e).lower() or "connection" in str(e).lower():
                pytest.skip(f"SSH connection unavailable: {e}")
            raise
        assert result["design_id"] == "verify_trainable_parameters_design"
        assert result["execution_target"] == "remote_wsl"

    def test_curriculum_probe_result_structure(self):
        d = _design("deeplens_curriculum_probe_design", "curriculum_probe")
        try:
            result = _execute_remote_diagnostic_design(d, "windows_wsl")
        except Exception as e:
            if "ssh" in str(e).lower() or "connection" in str(e).lower():
                pytest.skip(f"SSH connection unavailable: {e}")
            raise
        assert result["design_id"] == "deeplens_curriculum_probe_design"

    def test_unknown_design_returns_unsupported(self):
        d = _design("unknown_design", "unknown")
        result = _execute_remote_diagnostic_design(d, "windows_wsl")
        assert result["status"] == "unsupported"
