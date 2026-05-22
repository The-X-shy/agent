"""Phase 25 autonomous loop schema tests."""

from optiresearch.schemas.autonomous_loop import (
    AutonomousLoopSpec,
    AutonomousLoopIteration,
    AutonomousLoopResult,
)


def test_spec_defaults():
    spec = AutonomousLoopSpec(objective="Test co-design")
    assert spec.objective == "Test co-design"
    assert spec.max_iterations == 3
    assert spec.execution_mode == "dry_run"
    assert spec.strict_claim_gate is True
    assert spec.allow_remote is False
    assert spec.allow_code_modification is False


def test_spec_remote_requires_explicit_opt_in():
    spec = AutonomousLoopSpec(
        objective="Test",
        execution_mode="remote_opt_in",
        allow_remote=True,
        remote_worker_id="wslbox",
    )
    assert spec.allow_remote is True
    assert spec.remote_worker_id == "wslbox"


def test_spec_serialization():
    spec = AutonomousLoopSpec(objective="Test", max_iterations=5)
    data = spec.model_dump()
    assert data["objective"] == "Test"
    assert data["max_iterations"] == 5
    assert data["execution_mode"] == "dry_run"


def test_iteration_defaults():
    it = AutonomousLoopIteration(iteration_id=1)
    assert it.iteration_id == 1
    assert it.strategy_recommendation == {}
    assert it.next_action == "continue"
    assert it.stop_reason == ""


def test_result_validates():
    result = AutonomousLoopResult(
        loop_id="test_loop",
        status="completed",
        objective="Test",
    )
    data = result.model_dump()
    assert data["loop_id"] == "test_loop"
    assert data["status"] == "completed"
    assert data["iterations"] == []


def test_result_with_iterations():
    it = AutonomousLoopIteration(iteration_id=1)
    result = AutonomousLoopResult(
        loop_id="test",
        status="completed",
        objective="Test objective",
        iterations=[it],
        final_supported_claims=["claim 1"],
        final_unsupported_claims=["claim 2"],
    )
    assert len(result.iterations) == 1
    assert result.final_supported_claims == ["claim 1"]
    assert len(result.final_unsupported_claims) == 1


def test_spec_allowed_backends_default():
    spec = AutonomousLoopSpec(objective="Test")
    assert "deeplens_geolens_geometric" in spec.allowed_backends
    assert "phase_to_fft_proxy" in spec.allowed_backends


def test_spec_stop_conditions_default():
    spec = AutonomousLoopSpec(objective="Test")
    assert "claim_supported" in spec.stop_conditions
    assert "no_improvement" in spec.stop_conditions
    assert "max_iterations_reached" in spec.stop_conditions
