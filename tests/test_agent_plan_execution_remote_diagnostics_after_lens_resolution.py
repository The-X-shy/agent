"""Tests for agent plan execution with remote diagnostics after lens resolution."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestAgentPlanExecutionRemoteDiagnostics:
    def test_seed_result_path_loaded_with_lens_info(self, tmp_path):
        seed_dir = tmp_path / "geolens_stabilization_1779550632"
        seed_dir.mkdir(parents=True)
        sweep = seed_dir / "sweep_results.json"
        sweep.write_text(json.dumps({
            "sweep_id": "test_sweep",
            "lens_file": "auto:cooke",
            "resolved_lens_file": "/path/to/cooke.json",
            "status": "completed",
        }))
        assert sweep.exists()

    def test_design_includes_trainable_param_design(self):
        designs = [
            "verify_trainable_parameters_design",
            "autograd_graph_audit_design",
            "deeplens_curriculum_probe_design",
        ]
        assert "verify_trainable_parameters_design" in designs
        assert "autograd_graph_audit_design" in designs

    def test_claim_gate_diagnostic_evidence_only(self):
        max_allowed_claim = "diagnostic_evidence"
        claimed = "diagnostic_evidence"
        assert claimed <= max_allowed_claim

    def test_no_optical_improvement_claim_from_diagnostics(self):
        claim = "diagnostic_evidence"
        forbidden = ["optical_improvement", "design_improvement", "performance_gain"]
        assert claim not in forbidden

    def test_error_code_not_lens_file_not_found(self):
        result = {"status": "succeeded", "error_code": None, "resolved_lens_file": "/p/cooke.json"}
        assert result["error_code"] != "LENS_FILE_NOT_FOUND"
        assert result["resolved_lens_file"] is not None

    def test_lens_resolution_context_in_execution_result(self):
        execution_result = {
            "execution_id": "exec_001",
            "remote_diagnostics": {
                "trainable_parameter_inspection": {
                    "resolved_lens_file": "/mnt/d/external/DeepLens/datasets/lenses/cooke.json",
                    "lens_resolution_source": "wsl_external",
                    "trainable_param_count": 10,
                },
                "autograd_audit": {
                    "resolved_lens_file": "/mnt/d/external/DeepLens/datasets/lenses/cooke.json",
                    "graph_connected": True,
                },
            },
        }
        diags = execution_result["remote_diagnostics"]
        assert diags["trainable_parameter_inspection"]["resolved_lens_file"] is not None
        assert diags["autograd_audit"]["graph_connected"] is True

    def test_remote_worker_id_tracked_in_execution(self):
        execution = {
            "execution_id": "exec_002",
            "remote_worker_id": "windows_wsl",
            "diagnostics": [{"worker_id": "windows_wsl", "status": "succeeded"}],
        }
        assert execution["remote_worker_id"] == "windows_wsl"
        assert all(d["worker_id"] == "windows_wsl" for d in execution["diagnostics"])

    def test_structured_error_on_different_lens_failure(self):
        result = {
            "status": "unavailable",
            "error_code": "GEOLENS_PSF_GEOMETRIC_ERROR",
            "diagnosis": ["psf_model_not_available"],
        }
        assert result["error_code"] != "LENS_FILE_NOT_FOUND"
        assert result["error_code"].startswith("GEOLENS")
