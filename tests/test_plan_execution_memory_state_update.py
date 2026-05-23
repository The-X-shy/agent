from optiresearch.agent_system.state_store import StateStore
from optiresearch.memory.claim_gate_v2 import ClaimGateDecision
from optiresearch.memory.research_memory_v2 import ResearchMemoryV2


def test_research_memory_records_plan_execution_outcome():
    mem = ResearchMemoryV2()

    entry_id = mem.record_plan_execution_outcome(
        execution_id="plan_exec_memory_test",
        selected_design="alt_param_diffractive_sweep",
        execution_result={
            "status": "completed",
            "design_id": "report_negative_result_doc",
            "evidence_level": "report_only",
        },
        attempted_designs=[
            {"design_id": "alt_param_diffractive_sweep", "status": "unsupported"},
            {"design_id": "report_negative_result_doc", "status": "completed"},
        ],
        skipped_higher_ranked_designs=[
            {"design_id": "real_data_request_req", "skipped_reason": "needs_user_data"}
        ],
        claim_decision={"decision": "supported", "max_allowed_claim": "report_only"},
    )

    entry = mem.query(content_contains="plan_exec_memory_test")[0]
    assert entry.memory_id == entry_id
    assert entry.memory_type == "ExperimentOutcome"
    assert entry.metadata["selected_design"] == "alt_param_diffractive_sweep"
    assert entry.metadata["attempted_designs"][1]["design_id"] == "report_negative_result_doc"
    assert entry.metadata["skipped_higher_ranked_designs"][0]["skipped_reason"] == "needs_user_data"


def test_state_store_records_plan_execution_outcome(tmp_path):
    store = StateStore(workspace_root=tmp_path)
    decision = ClaimGateDecision(
        decision="supported",
        max_allowed_claim="report_only",
        violation_reason=None,
        violation_type=None,
        safe_wording="The negative result is documented",
    )

    store.record_plan_execution_outcome(
        execution_id="plan_exec_state_test",
        selected_design="alt_param_diffractive_sweep",
        execution_result={
            "status": "completed",
            "design_id": "report_negative_result_doc",
            "evidence_level": "report_only",
        },
        claim_decision=decision,
        attempted_designs=[
            {"design_id": "alt_param_diffractive_sweep", "status": "unsupported"},
            {"design_id": "report_negative_result_doc", "status": "completed"},
        ],
        skipped_higher_ranked_designs=[
            {"design_id": "real_data_request_req", "skipped_reason": "needs_user_data"}
        ],
    )
    store.snapshot()

    state = store.state
    assert state.last_executed_design == "report_negative_result_doc"
    assert state.last_execution_status == "completed"
    assert state.last_claim_decision["decision"] == "supported"
    assert "alt_param_diffractive_sweep" in state.pending_actions
    assert "The negative result is documented" in state.known_supported_claims
    assert state.snapshot_count == 1
