"""Phase 25 trajectory evaluator tests."""

from optiresearch.agents.trajectory_evaluator import evaluate_trajectory
from optiresearch.schemas.autonomous_loop import (
    AutonomousLoopIteration,
    AutonomousLoopSpec,
)


def _make_iteration(it_id: int, status: str, payload: dict) -> AutonomousLoopIteration:
    return AutonomousLoopIteration(
        iteration_id=it_id,
        execution_result={"status": status, "result_payload": payload},
    )


def _make_iteration_with_claim(it_id: int, status: str, payload: dict, ceiling: str) -> AutonomousLoopIteration:
    return AutonomousLoopIteration(
        iteration_id=it_id,
        execution_result={"status": status, "result_payload": payload},
        claim_gate_decision={"max_allowed_claim": ceiling, "decision": "supported"},
    )


def test_no_iterations_returns_no_iterations():
    spec = AutonomousLoopSpec(objective="Test")
    result = evaluate_trajectory([], spec)
    assert result.stop_reason == "no_iterations"


def test_improvement_detected():
    spec = AutonomousLoopSpec(objective="Test")
    iterations = [
        _make_iteration(1, "succeeded", {"reconstruction_loss_after": 1.0}),
        _make_iteration(2, "succeeded", {"reconstruction_loss_after": 0.5}),
    ]
    result = evaluate_trajectory(iterations, spec)
    assert result.improvement_detected is True
    assert result.best_iteration == 2


def test_no_improvement():
    spec = AutonomousLoopSpec(objective="Test")
    iterations = [
        _make_iteration(1, "succeeded", {"reconstruction_loss_after": 1.0}),
        _make_iteration(2, "succeeded", {"reconstruction_loss_after": 1.1}),
    ]
    result = evaluate_trajectory(iterations, spec)
    assert result.improvement_detected is False


def test_repeated_failure_detected():
    spec = AutonomousLoopSpec(objective="Test", max_iterations=3)
    iterations = [
        _make_iteration(1, "failed", {}),
        _make_iteration(2, "failed", {}),
    ]
    result = evaluate_trajectory(iterations, spec)
    assert result.repeated_failure_detected is True
    assert result.stop_reason == "repeated_failure"


def test_single_failure_not_repeated():
    spec = AutonomousLoopSpec(objective="Test", max_iterations=3)
    iterations = [
        _make_iteration(1, "failed", {}),
        _make_iteration(2, "succeeded", {"reconstruction_loss_after": 0.5}),
    ]
    result = evaluate_trajectory(iterations, spec)
    assert result.repeated_failure_detected is False


def test_claim_ceiling_reached():
    spec = AutonomousLoopSpec(objective="Test", max_iterations=5)
    iterations = [
        _make_iteration_with_claim(1, "succeeded", {"loss_after": 0.5}, "native_lens_simulation"),
        _make_iteration_with_claim(2, "succeeded", {"loss_after": 0.4}, "native_lens_simulation"),
    ]
    result = evaluate_trajectory(iterations, spec)
    assert result.claim_ceiling_reached is True
    assert result.stop_reason == "claim_ceiling_reached"


def test_max_iterations_reached():
    spec = AutonomousLoopSpec(objective="Test", max_iterations=2)
    iterations = [
        _make_iteration(1, "succeeded", {"reconstruction_loss_after": 1.0}),
        _make_iteration(2, "succeeded", {"reconstruction_loss_after": 0.9}),
    ]
    result = evaluate_trajectory(iterations, spec)
    assert result.stop_reason == "max_iterations_reached"


def test_extracts_mse_after_fallback():
    spec = AutonomousLoopSpec(objective="Test")
    iterations = [
        _make_iteration(1, "succeeded", {"mse_after": 0.05}),
        _make_iteration(2, "succeeded", {"mse_after": 0.03}),
    ]
    result = evaluate_trajectory(iterations, spec)
    assert result.best_iteration == 2
    assert result.metric_trajectory == [0.05, 0.03]


def test_metric_trajectory_handles_missing_values():
    spec = AutonomousLoopSpec(objective="Test")
    iterations = [
        _make_iteration(1, "succeeded", {}),
        _make_iteration(2, "succeeded", {"reconstruction_loss_after": 0.5}),
    ]
    result = evaluate_trajectory(iterations, spec)
    assert result.metric_trajectory == [0.0, 0.5]


def test_skipped_iterations_count_as_failures():
    spec = AutonomousLoopSpec(objective="Test", max_iterations=5)
    iterations = [
        _make_iteration(1, "skipped", {}),
        _make_iteration(2, "unsupported", {}),
    ]
    result = evaluate_trajectory(iterations, spec)
    assert result.repeated_failure_detected is True
