"""Phase 26 LLM planner fallback tests."""

from optiresearch.agents.llm_planner import LLMPlanner


def test_disabled_provider_falls_back():
    planner = LLMPlanner()
    result = planner.plan("test", provider_name="disabled")
    assert result.status == "fallback_used"
    assert result.fallback_strategy is not None
    assert "recommended_action" in result.fallback_strategy


def test_fallback_strategy_has_rationale():
    planner = LLMPlanner()
    result = planner.plan("test", provider_name="disabled")
    assert result.fallback_strategy["rationale"] != ""


def test_fallback_records_reason():
    planner = LLMPlanner()
    result = planner.plan("test", provider_name="disabled")
    assert "reason" in result.fallback_strategy
