"""Test that claim_downgraded status is not treated as execution failure."""

import pytest
from optiresearch.agents.trajectory_evaluator import evaluate_trajectory


class _FakeIter:
    def __init__(self, iteration_id, status, metrics=None, claim_dec=None):
        self.iteration_id = iteration_id
        self.execution_result = {"status": status, "result_payload": metrics or {}}
        self.claim_gate_decision = claim_dec or {}
        self.metrics_snapshot = {}


class _FakeSpec:
    def __init__(self, max_iterations=3):
        self.max_iterations = max_iterations


def test_claim_downgraded_not_counted_as_failure():
    iterations = [
        _FakeIter(1, "claim_downgraded", {"reconstruction_loss_after": 0.05}),
    ]
    spec = _FakeSpec(max_iterations=3)
    result = evaluate_trajectory(iterations, spec)
    assert not result.repeated_failure_detected


def test_claim_downgraded_with_metrics_enters_trajectory():
    iterations = [
        _FakeIter(1, "claim_downgraded", {"reconstruction_loss_after": 0.05}),
        _FakeIter(2, "succeeded", {"reconstruction_loss_after": 0.03}),
    ]
    spec = _FakeSpec(max_iterations=3)
    result = evaluate_trajectory(iterations, spec)
    assert len(result.metric_trajectory) == 2
    assert result.metric_trajectory[0] == 0.05
    assert result.improvement_detected


def test_no_improvement_stop_requires_min_iterations():
    iterations = [
        _FakeIter(1, "succeeded", {"reconstruction_loss_after": 0.05}),
    ]
    spec = _FakeSpec(max_iterations=3)
    result = evaluate_trajectory(iterations, spec, min_iterations_before_stop=2)
    assert result.stop_reason == ""


def test_no_improvement_stop_after_min_iterations_with_patience():
    iterations = [
        _FakeIter(1, "succeeded", {"reconstruction_loss_after": 0.05}),
        _FakeIter(2, "succeeded", {"reconstruction_loss_after": 0.06}),
        _FakeIter(3, "succeeded", {"reconstruction_loss_after": 0.07}),
    ]
    spec = _FakeSpec(max_iterations=10)
    result = evaluate_trajectory(
        iterations, spec,
        min_iterations_before_stop=2,
        no_improvement_patience=2,
    )
    assert result.stop_reason == "no_improvement"


def test_claim_downgraded_count_tracked():
    iterations = [
        _FakeIter(1, "claim_downgraded", {"reconstruction_loss_after": 0.05}),
        _FakeIter(2, "claim_downgraded", {"reconstruction_loss_after": 0.04}),
        _FakeIter(3, "succeeded", {"reconstruction_loss_after": 0.03}),
    ]
    spec = _FakeSpec(max_iterations=5)
    result = evaluate_trajectory(iterations, spec)
    assert result.claim_downgraded_count == 2


def test_improvement_resets_no_improvement_streak():
    iterations = [
        _FakeIter(1, "succeeded", {"reconstruction_loss_after": 0.05}),
        _FakeIter(2, "succeeded", {"reconstruction_loss_after": 0.06}),
        _FakeIter(3, "succeeded", {"reconstruction_loss_after": 0.04}),
        _FakeIter(4, "succeeded", {"reconstruction_loss_after": 0.03}),
    ]
    spec = _FakeSpec(max_iterations=10)
    result = evaluate_trajectory(
        iterations, spec,
        min_iterations_before_stop=2,
        no_improvement_patience=2,
    )
    assert result.stop_reason == ""
    assert result.improvement_detected
