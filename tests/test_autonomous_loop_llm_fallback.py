"""Phase 26 autonomous loop LLM fallback tests."""

from optiresearch.schemas.autonomous_loop import AutonomousLoopSpec
from optiresearch.runtime.autonomous_research_loop import run_autonomous_research_loop


def test_llm_first_fallback_to_rule():
    spec = AutonomousLoopSpec(
        objective="test llm first fallback",
        max_iterations=1,
        execution_mode="dry_run",
        planner_mode="llm_first_with_rule_fallback",
        llm_provider="mock",
    )
    result = run_autonomous_research_loop(spec)
    assert result.status == "dry_run_only"
    it = result.iterations[0]
    assert "recommended_action" in it.strategy_recommendation


def test_fallback_when_llm_fails():
    spec = AutonomousLoopSpec(
        objective="test disabled llm fallback",
        max_iterations=1,
        execution_mode="dry_run",
        planner_mode="llm_first_with_rule_fallback",
        llm_provider="mock",
    )
    result = run_autonomous_research_loop(spec)
    assert result.status == "dry_run_only"
    assert len(result.iterations) == 1


def test_spec_has_planner_mode_default():
    spec = AutonomousLoopSpec(objective="test")
    assert spec.planner_mode == "rule_based"
    assert spec.llm_provider == "mock"
