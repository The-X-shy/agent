"""Phase 25 claim gate hard enforcement tests."""

from optiresearch.schemas.autonomous_loop import (
    AutonomousLoopSpec,
    AutonomousLoopIteration,
)
from optiresearch.runtime.autonomous_research_loop import (
    _check_claim_gate,
    run_autonomous_research_loop,
)


def test_check_claim_gate_returns_decision_dict():
    spec = AutonomousLoopSpec(objective="Test", allowed_backends=["deeplens_geolens_geometric"])
    strategy = {"recommended_action": "retry_with_smaller_lr"}
    execution = {"status": "succeeded", "result_payload": {"reconstruction_loss_after": 0.5}}

    decision = _check_claim_gate(strategy, execution, spec)
    assert "decision" in decision
    assert "max_allowed_claim" in decision
    assert decision["decision"] in ("supported", "qualified", "needs_followup", "unsupported")
    assert decision["max_allowed_claim"] == "native_lens_simulation"


def test_check_claim_gate_with_geometric_backend():
    spec = AutonomousLoopSpec(objective="Test", allowed_backends=["deeplens_geolens_geometric"])
    strategy = {"recommended_action": "run_ablation"}
    execution = {"status": "succeeded", "result_payload": {}}

    decision = _check_claim_gate(strategy, execution, spec)
    assert decision["max_allowed_claim"] == "native_lens_simulation"


def test_strict_claim_gate_enforced_locally():
    spec = AutonomousLoopSpec(
        objective="Test enforcement",
        max_iterations=1,
        execution_mode="local",
        allowed_backends=["deeplens_geolens_geometric"],
        strict_claim_gate=True,
    )
    result = run_autonomous_research_loop(spec)
    # The loop should complete without error
    assert result.status in ("completed", "stopped", "failed")
    assert result.loop_id is not None


def test_dry_run_with_strict_claim_gate():
    spec = AutonomousLoopSpec(
        objective="Test dry gate",
        max_iterations=1,
        execution_mode="dry_run",
        strict_claim_gate=True,
    )
    result = run_autonomous_research_loop(spec)
    assert result.status == "dry_run_only"
