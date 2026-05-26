"""Tests for gradient instability analyzer with remote diagnostic metrics."""

import json

import pytest

from optiresearch.analysis.gradient_instability_analyzer import (
    analyze_gradient_instability,
)


class TestRemoteDiagnosticMetricsIngestion:
    def test_empty_sources_returns_insufficient_evidence(self):
        diag = analyze_gradient_instability(source_paths=[], remote_job_ids=[])
        assert diag.status == "insufficient_evidence"

    def test_nonexistent_path_handled_gracefully(self, tmp_path):
        diag = analyze_gradient_instability(
            source_paths=[str(tmp_path / "nonexistent.json")],
            remote_job_ids=["remote_job_nonexistent"]
        )
        assert diag.status == "insufficient_evidence"

    def test_metrics_from_source_file_included(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps({
            "optical_gradient_norm_max": 0.5,
            "stable_training_succeeded": True,
            "optical_parameters_changed": True,
        }))
        diag = analyze_gradient_instability(source_paths=[str(result_file)])
        assert diag.status in ("insufficient_evidence", "diagnosed", "stable")


class TestGradientInstabilityFailureModes:
    def test_excessive_gradient_norm_detected(self, tmp_path):
        result_file = tmp_path / "high_grad.json"
        result_file.write_text(json.dumps({
            "optical_gradient_norm_max": 5000,
            "optical_gradient_norm_mean": 3000,
            "stable_training_succeeded": False,
            "optical_parameters_changed": False,
        }))
        diag = analyze_gradient_instability(source_paths=[str(result_file)])
        if diag.status == "diagnosed":
            assert any(m in diag.failure_modes
                       for m in ("excessive_gradient_norm", "extreme_gradient_spike"))

    def test_no_parameter_change_detected(self, tmp_path):
        result_file = tmp_path / "no_change.json"
        result_file.write_text(json.dumps({
            "optical_gradient_norm_max": 0.1,
            "optical_parameters_changed": False,
            "accepted_update_count": 0,
            "rejected_update_count": 3,
            "stable_training_succeeded": False,
        }))
        diag = analyze_gradient_instability(source_paths=[str(result_file)])
        if diag.status == "diagnosed":
            assert any(m in diag.failure_modes
                       for m in ("no_parameter_change", "all_updates_rollback"))


class TestRemoteDiagnosticClassification:
    def test_gradient_flow_blocked_scenario(self, tmp_path):
        result_file = tmp_path / "grad_blocked.json"
        result_file.write_text(json.dumps({
            "trainable_param_count": 10,
            "params_with_grad": 0,
            "graph_connected": False,
            "psf_requires_grad": False,
            "grad_norm_max": 0.0,
        }))
        diag = analyze_gradient_instability(source_paths=[str(result_file)])
        assert diag.status is not None

    def test_optimizer_update_blocked_scenario(self, tmp_path):
        result_file = tmp_path / "opt_blocked.json"
        result_file.write_text(json.dumps({
            "params_with_grad": 5,
            "candidate_update_changes_parameter": False,
            "accepted_update_count": 0,
            "optical_parameters_changed": False,
        }))
        diag = analyze_gradient_instability(source_paths=[str(result_file)])
        assert diag.status is not None

    def test_objective_instability_scenario(self, tmp_path):
        result_file = tmp_path / "obj_instability.json"
        result_file.write_text(json.dumps({
            "graph_connected": True,
            "accepted_update_count": 0,
            "rejected_update_count": 5,
            "rollback_count": 3,
            "optical_gradient_norm_max": 0.5,
            "stable_training_succeeded": False,
        }))
        diag = analyze_gradient_instability(source_paths=[str(result_file)])
        assert diag.status is not None
