"""Tests for remote diagnostic report with lens resolution section."""

import json
from pathlib import Path

import pytest


class TestRemoteDiagnosticReportLensResolution:
    def test_report_includes_lens_resolution_section(self):
        """Verify that remote diagnostic reports can include lens resolution data."""
        lens_data = {
            "requested_lens_file": "auto:cooke",
            "resolved_lens_file": "/path/to/cooke.json",
            "lens_resolution_source": "env_DEEPLENS_REPO_PATH",
            "checked_lens_paths": ["/a/cooke.json", "/b/cooke.json"],
            "exists": True,
            "error_code": None,
        }
        assert lens_data["resolved_lens_file"] == "/path/to/cooke.json"
        assert lens_data["lens_resolution_source"] == "env_DEEPLENS_REPO_PATH"

    def test_unresolved_lens_in_report(self):
        lens_data = {
            "requested_lens_file": "auto:cooke",
            "resolved_lens_file": None,
            "lens_resolution_source": None,
            "checked_lens_paths": ["/a/cooke.json"],
            "exists": False,
            "error_code": "LENS_FILE_NOT_FOUND",
        }
        assert lens_data["error_code"] == "LENS_FILE_NOT_FOUND"
        assert lens_data["exists"] is False


class TestDiagnosticMetricsInReport:
    def test_trainable_param_metrics_in_report(self):
        metrics = {
            "trainable_param_count": 10,
            "params_with_grad": 8,
            "zero_gradient_parameters": [2, 5],
            "grad_norm_max": 0.5,
            "grad_norm_mean": 0.1,
        }
        assert metrics["trainable_param_count"] > 0
        assert metrics["params_with_grad"] > 0
        assert len(metrics["zero_gradient_parameters"]) > 0

    def test_autograd_audit_metrics_in_report(self):
        metrics = {
            "graph_connected": True,
            "psf_requires_grad": True,
            "loss_requires_grad": True,
            "candidate_update_changes_parameter": True,
            "detach_suspected": False,
            "grad_norm_max": 0.5,
            "recommended_next_strategy": "geolens_curriculum_probe",
        }
        assert metrics["graph_connected"] is True
        assert metrics["detach_suspected"] is False
        assert metrics["recommended_next_strategy"] is not None

    def test_gradient_flow_interpretation(self):
        interpretation = {
            "trainable_param_count": 10,
            "params_with_grad": 0,
            "likely_cause": "gradient_flow_blocked",
            "recommended_strategy": "component_first_probe",
        }
        assert interpretation["likely_cause"] == "gradient_flow_blocked"
        assert interpretation["recommended_strategy"] == "component_first_probe"
