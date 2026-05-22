"""Phase 26 autonomous loop with LLM planner tests."""

from optiresearch.schemas.autonomous_loop import AutonomousLoopSpec
from optiresearch.runtime.autonomous_research_loop import run_autonomous_research_loop


def test_llm_assisted_dry_run():
    spec = AutonomousLoopSpec(
        objective="test llm assisted",
        max_iterations=1,
        execution_mode="dry_run",
        planner_mode="llm_assisted",
        llm_provider="mock",
    )
    result = run_autonomous_research_loop(spec)
    assert result.status == "dry_run_only"
    assert len(result.iterations) == 1
    it = result.iterations[0]
    assert "recommended_action" in it.strategy_recommendation


def test_llm_assisted_planner_metadata():
    spec = AutonomousLoopSpec(
        objective="test planner metadata",
        max_iterations=1,
        execution_mode="dry_run",
        planner_mode="llm_assisted",
        llm_provider="mock",
    )
    result = run_autonomous_research_loop(spec)
    it = result.iterations[0]
    metadata = it.strategy_recommendation.get("metadata", {})
    assert metadata.get("planner") in ("llm", "fallback")


def test_rule_based_as_default():
    spec = AutonomousLoopSpec(
        objective="test rule based default",
        max_iterations=1,
        execution_mode="dry_run",
    )
    result = run_autonomous_research_loop(spec)
    assert result.status == "dry_run_only"


def test_llm_assisted_local_mode():
    spec = AutonomousLoopSpec(
        objective="test llm local",
        max_iterations=1,
        execution_mode="local",
        planner_mode="llm_assisted",
        llm_provider="mock",
        allowed_backends=["deeplens_geolens_geometric"],
        strict_claim_gate=True,
    )
    result = run_autonomous_research_loop(spec)
    assert result.status in ("completed", "stopped", "failed")
