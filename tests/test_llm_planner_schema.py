"""Phase 26 LLM planner schema tests."""

from optiresearch.schemas.llm_planner import (
    LLMPlannerContext,
    LLMPlannerProposal,
    LLMPlannerResult,
)


def test_context_defaults():
    ctx = LLMPlannerContext(objective="Test")
    assert ctx.objective == "Test"
    assert ctx.execution_mode == "dry_run"
    assert ctx.max_candidate_plans == 3


def test_proposal_defaults():
    p = LLMPlannerProposal()
    assert p.risk_level == "low"
    assert p.recommended_action == "stop_and_report"


def test_result_defaults():
    r = LLMPlannerResult(status="succeeded")
    assert r.status == "succeeded"
    assert r.provider == "mock"
    assert r.proposals == []


def test_result_with_proposal():
    p = LLMPlannerProposal(proposal_id="p1", recommended_action="retry_with_smaller_lr")
    r = LLMPlannerResult(status="succeeded", proposals=[p], selected_proposal=p)
    assert len(r.proposals) == 1
    assert r.selected_proposal.proposal_id == "p1"


def test_fallback_result():
    r = LLMPlannerResult(
        status="fallback_used",
        provider="deepseek",
        fallback_strategy={"reason": "no_key"},
    )
    assert r.status == "fallback_used"
    assert r.fallback_strategy["reason"] == "no_key"


def test_proposal_serialization():
    p = LLMPlannerProposal(
        proposal_id="test_1",
        recommended_action="enable_rollback",
        backend_id="deeplens_geolens_geometric",
        task_type="stable_lens_hsi_codesign",
        risk_level="low",
    )
    data = p.model_dump()
    assert data["proposal_id"] == "test_1"
    assert data["backend_id"] == "deeplens_geolens_geometric"
