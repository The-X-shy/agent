"""Test metric-first trajectory evaluation with patience and min iterations."""

import pytest
from optiresearch.agents.trajectory_evaluator import evaluate_trajectory


class _FakeIter:
    def __init__(self, iteration_id, status="succeeded", metrics=None, claim_dec=None):
        self.iteration_id = iteration_id
        self.execution_result = {"status": status, "result_payload": metrics or {}}
        self.claim_gate_decision = claim_dec or {}
        self.metrics_snapshot = {}


class _FakeSpec:
    def __init__(self, max_iterations=3):
        self.max_iterations = max_iterations


def test_improvement_detected_with_decreasing_loss():
    iterations = [
        _FakeIter(1, "succeeded", {"reconstruction_loss_after": 0.10}),
        _FakeIter(2, "succeeded", {"reconstruction_loss_after": 0.05}),
    ]
    spec = _FakeSpec()
    result = evaluate_trajectory(iterations, spec)
    assert result.improvement_detected
    assert result.best_iteration == 2
    assert result.best_metric_value == 0.05


def test_max_iterations_stops_naturally():
    iterations = [
        _FakeIter(1, "succeeded", {"reconstruction_loss_after": 0.05}),
        _FakeIter(2, "succeeded", {"reconstruction_loss_after": 0.04}),
        _FakeIter(3, "succeeded", {"reconstruction_loss_after": 0.03}),
    ]
    spec = _FakeSpec(max_iterations=3)
    result = evaluate_trajectory(iterations, spec)
    assert result.stop_reason == "max_iterations_reached"


def test_mse_decrease_counts_as_improvement():
    iterations = [
        _FakeIter(1, "succeeded", {"psnr_after": 25.0, "mse_after": 0.01}),
        _FakeIter(2, "succeeded", {"psnr_after": 28.0, "mse_after": 0.005}),
    ]
    spec = _FakeSpec()
    result = evaluate_trajectory(iterations, spec)
    assert result.improvement_detected


def test_claim_downgrade_does_not_affect_metric_improvement():
    iterations = [
        _FakeIter(1, "claim_downgraded",
                  {"reconstruction_loss_after": 0.10},
                  {"decision": "qualified", "max_allowed_claim": "native_full_reconstruction_proxy"}),
        _FakeIter(2, "succeeded",
                  {"reconstruction_loss_after": 0.05},
                  {"decision": "supported", "max_allowed_claim": "native_full_reconstruction_proxy"}),
    ]
    spec = _FakeSpec(max_iterations=3)
    result = evaluate_trajectory(iterations, spec)
    assert result.improvement_detected
    assert result.claim_downgraded_count == 1


def test_min_iterations_satisfied_flag():
    iterations = [
        _FakeIter(1, "succeeded", {"reconstruction_loss_after": 0.05}),
        _FakeIter(2, "succeeded", {"reconstruction_loss_after": 0.04}),
    ]
    spec = _FakeSpec()
    result = evaluate_trajectory(iterations, spec, min_iterations_before_stop=2)
    assert result.min_iterations_satisfied


def test_repeated_failure_stops_despite_min_iterations():
    iterations = [
        _FakeIter(1, "failed", {}),
        _FakeIter(2, "failed", {}),
    ]
    spec = _FakeSpec(max_iterations=5)
    result = evaluate_trajectory(iterations, spec, min_iterations_before_stop=3)
    assert result.stop_reason == "repeated_failure"
    assert result.repeated_failure_detected


def test_loss_after_preferred_over_mse():
    iterations = [
        _FakeIter(1, "succeeded", {"reconstruction_loss_after": 0.10, "mse_after": 0.01}),
        _FakeIter(2, "succeeded", {"reconstruction_loss_after": 0.05, "mse_after": 0.02}),
    ]
    spec = _FakeSpec()
    result = evaluate_trajectory(iterations, spec)
    assert result.metric_trajectory[0] == 0.10
    assert result.metric_trajectory[1] == 0.05


def test_empty_iterations_returns_no_iterations():
    spec = _FakeSpec()
    result = evaluate_trajectory([], spec)
    assert result.stop_reason == "no_iterations"
