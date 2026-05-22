"""Phase 26 claim gate on LLM proposals tests."""

from optiresearch.agents.llm_planner import LLMPlanner
from optiresearch.memory.claim_gate_v2 import ClaimGateV2
from optiresearch.schemas.llm_planner import LLMPlannerProposal


def test_mock_proposals_pass_claim_gate():
    planner = LLMPlanner()
    result = planner.plan("test", provider_name="mock")
    assert result.selected_proposal is not None
    gate = ClaimGateV2()
    decision = gate.check_claim(
        result.selected_proposal.proposed_claim,
        result.selected_proposal.backend_id,
    )
    assert decision.decision in ("supported", "qualified", "needs_followup")


def test_overclaim_on_geometric_rejected():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Full coherent wave-optics native HSI co-design",
        "deeplens_geolens_geometric",
    )
    assert decision.decision == "unsupported"


def test_overclaim_applies_safe_wording():
    p = LLMPlannerProposal(
        proposal_id="test",
        recommended_action="run_ablation",
        backend_id="deeplens_geolens_geometric",
        proposed_claim="Full wave-optics co-design",
    )
    planner = LLMPlanner()
    p = planner._apply_claim_gate(p)
    # Safe wording should be set
    assert len(p.safe_wording) > 0
