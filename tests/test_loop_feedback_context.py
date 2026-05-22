"""Tests for loop feedback context builder."""

from optiresearch.agents.loop_feedback_context import (
    build_feedback_context,
    build_recent_results,
)


class _FakeIteration:
    """Minimal fake iteration for testing."""
    def __init__(self, iteration_id, strategy, execution, claim):
        self.iteration_id = iteration_id
        self.strategy_recommendation = strategy
        self.execution_result = execution
        self.claim_gate_decision = claim


def test_build_feedback_context_from_successful():
    it = _FakeIteration(
        iteration_id=1,
        strategy={"recommended_action": "retry_with_smaller_lr"},
        execution={
            "status": "succeeded",
            "backend_id": "phase_to_fft_proxy",
            "result_payload": {
                "reconstruction_loss_before": 0.5,
                "reconstruction_loss_after": 0.3,
                "rollback_count": 0,
                "accepted_update_count": 3,
            },
        },
        claim={"decision": "supported"},
    )
    ctx = build_feedback_context(it)
    assert ctx["iteration_id"] == 1
    assert ctx["previous_action"] == "retry_with_smaller_lr"
    assert ctx["previous_status"] == "succeeded"
    assert ctx["loss_before"] == 0.5
    assert ctx["loss_after"] == 0.3
    assert ctx["improvement_detected"] is True
    assert ctx["failure_mode"] == "improvement"


def test_build_feedback_context_detects_no_improvement():
    it = _FakeIteration(
        iteration_id=2,
        strategy={"recommended_action": "enable_rollback"},
        execution={
            "status": "succeeded",
            "backend_id": "phase_to_fft_proxy",
            "result_payload": {
                "reconstruction_loss_before": 0.3,
                "reconstruction_loss_after": 0.5,
                "rollback_count": 2,
            },
        },
        claim={"decision": "qualified"},
    )
    ctx = build_feedback_context(it)
    assert ctx["improvement_detected"] is False
    assert ctx["failure_mode"] == "rollback_protected"


def test_build_feedback_context_handles_empty():
    it = _FakeIteration(
        iteration_id=0,
        strategy={},
        execution={},
        claim={},
    )
    ctx = build_feedback_context(it)
    assert ctx["previous_status"] == ""
    assert ctx["loss_before"] is None
    assert ctx["loss_after"] is None
    assert ctx["improvement_detected"] is False


def test_build_feedback_context_detects_execution_failure():
    it = _FakeIteration(
        iteration_id=3,
        strategy={},
        execution={
            "status": "failed",
            "errors": [{"message": "DeepLens not found"}],
        },
        claim={},
    )
    ctx = build_feedback_context(it)
    assert ctx["failure_mode"] == "execution_failure"
    assert "DeepLens not found" in ctx["error_message"]


def test_build_recent_results():
    it1 = _FakeIteration(
        iteration_id=1,
        strategy={"recommended_action": "retry_with_smaller_lr"},
        execution={"status": "succeeded", "result_payload": {"reconstruction_loss_after": 0.3}},
        claim={"decision": "supported"},
    )
    it2 = _FakeIteration(
        iteration_id=2,
        strategy={"recommended_action": "stop_and_report"},
        execution={},  # no execution
        claim={},
    )
    results = build_recent_results([it1, it2])
    assert len(results) == 1  # only it1 has execution
    assert results[0]["iteration_id"] == 1
