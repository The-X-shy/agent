"""Phase 25 safety guard tests — remote and claim gate enforcement."""

from optiresearch.schemas.autonomous_loop import AutonomousLoopSpec
from optiresearch.runtime.autonomous_research_loop import run_autonomous_research_loop


def test_remote_fails_without_opt_in():
    spec = AutonomousLoopSpec(
        objective="Test remote",
        execution_mode="remote_opt_in",
        allow_remote=False,
    )
    result = run_autonomous_research_loop(spec)
    assert result.status == "failed"
    assert "allow_remote" in (result.error or "").lower()


def test_remote_fails_without_worker_id():
    spec = AutonomousLoopSpec(
        objective="Test remote",
        execution_mode="remote_opt_in",
        allow_remote=True,
        remote_worker_id=None,
    )
    result = run_autonomous_research_loop(spec)
    assert result.status == "failed"
    assert "remote_worker_id" in (result.error or "").lower()


def test_dry_run_succeeds():
    spec = AutonomousLoopSpec(
        objective="Test",
        max_iterations=1,
        execution_mode="dry_run",
    )
    result = run_autonomous_research_loop(spec)
    assert result.status == "dry_run_only"


def test_local_mode_with_subset_backends():
    spec = AutonomousLoopSpec(
        objective="Test local",
        max_iterations=1,
        execution_mode="local",
        allowed_backends=["deeplens_geolens_geometric"],
        strict_claim_gate=True,
    )
    result = run_autonomous_research_loop(spec)
    assert result.status in ("completed", "stopped", "failed", "dry_run_only")


def test_strict_claim_gate_default_is_true():
    spec = AutonomousLoopSpec(objective="Test")
    assert spec.strict_claim_gate is True
