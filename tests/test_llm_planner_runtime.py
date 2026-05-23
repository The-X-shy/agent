"""Phase 26 LLM planner runtime tests."""

from optiresearch.agents.llm_planner import LLMPlanner


def test_mock_planner_returns_proposals():
    planner = LLMPlanner()
    result = planner.plan(
        "test objective", provider_name="mock",
        allowed_task_types=["stable_lens_hsi_codesign", "backend_probe",
                           "native_lens_simulation_codesign"],
    )
    assert result.status == "succeeded"
    assert len(result.proposals) >= 2
    assert result.selected_proposal is not None


def test_mock_planner_selects_low_risk():
    planner = LLMPlanner()
    result = planner.plan("test", provider_name="mock")
    assert result.selected_proposal.risk_level == "low"


def test_mock_planner_has_valid_backend():
    planner = LLMPlanner()
    result = planner.plan("test", provider_name="mock", allowed_backends=["deeplens_geolens_geometric"])
    assert result.selected_proposal is not None
    assert result.selected_proposal.backend_id == "deeplens_geolens_geometric"


def test_mock_planner_respects_max_proposals():
    planner = LLMPlanner()
    result = planner.plan(
        "test", provider_name="mock", max_candidate_plans=1,
        allowed_task_types=["stable_lens_hsi_codesign", "backend_probe"],
    )
    assert len(result.proposals) == 1


def test_mock_planner_generates_trace():
    planner = LLMPlanner()
    result = planner.plan("test trace", provider_name="mock")
    assert result.planner_trace_path is not None


def test_build_context_includes_backends():
    planner = LLMPlanner()
    ctx = planner.build_context(
        objective="test",
        allowed_backends=["deeplens_geolens_geometric"],
        allowed_task_types=["stable_lens_hsi_codesign"],
        recent_results=[],
    )
    assert ctx["objective"] == "test"
    assert "backend_registry_summary" in ctx
    assert "research_memory" in ctx


def test_rank_proposals_low_first():
    from optiresearch.schemas.llm_planner import LLMPlannerProposal
    planner = LLMPlanner()
    proposals = [
        LLMPlannerProposal(proposal_id="high", risk_level="high"),
        LLMPlannerProposal(proposal_id="low", risk_level="low"),
        LLMPlannerProposal(proposal_id="med", risk_level="medium"),
    ]
    ranked = planner.rank_proposals(proposals)
    assert ranked[0].proposal_id == "low"
    assert ranked[1].proposal_id == "med"
    assert ranked[2].proposal_id == "high"
