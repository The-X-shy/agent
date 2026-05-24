"""Test report includes evidence alignment sections."""

from optiresearch.reports.agent_plan_execution_report import _markdown


def test_report_contains_evidence_alignment_section():
    r = {
        "execution_id": "test_align",
        "objective": "test",
        "status": "completed",
        "mode": "local",
        "executed_or_dry_run": "executed",
        "classified_failure": "test", "failure_category": "test",
        "candidate_strategies_count": 2, "candidate_strategies": [],
        "candidate_designs_count": 2,
        "candidate_designs": [
            {
                "design_id": "objective_redesign_simpler_metric_mse_only",
                "backend_id": "deeplens_geolens_geometric",
                "task_type": "stable_lens_hsi_codesign",
                "expected_evidence_level": "lightweight_scientific_execution",
                "actual_handler_evidence_level": "lightweight_scientific_execution",
                "evidence_alignment_status": "downgraded_to_handler_capability",
                "evidence_downgrade_reason": "Strategy target native_lens_simulation downgraded to handler capability lightweight_scientific_execution",
            },
            {
                "design_id": "backend_switch_waveoptics_coherent",
                "backend_id": "deeplens_coherent_asm",
                "task_type": "native_waveoptics_codesign",
                "expected_evidence_level": "native_waveoptics_simulation",
                "actual_handler_evidence_level": "structured_unsupported",
                "evidence_alignment_status": "downgraded_to_handler_capability",
                "evidence_downgrade_reason": "Strategy target native_waveoptics_simulation downgraded",
            },
        ],
        "plan_scores": [],
        "selected_design": None, "selected_design_rank": None,
        "executable_selection_reason": "",
        "selected_designs": [],
        "skipped_higher_ranked_designs": [],
        "attempted_designs": [],
        "execution_result": {},
        "execution_results": [],
        "claim_gate_decision": {},
        "claim_gate_decisions": [],
        "memory_updates": [], "memory_updated": False,
        "state_snapshots_count": 0, "state_snapshot_refs": [],
        "event_count": 0, "event_log_path": "", "report_path": "",
        "final_recommendation": "",
        "errors": [],
    }
    md = _markdown("test_align", r)
    assert "## 7b. Evidence Alignment" in md
    assert "Downgraded to Handler Capability" in md
    assert "objective_redesign_simpler_metric_mse_only" in md


def test_report_contains_handler_comparison():
    r = {
        "execution_id": "test_comp",
        "objective": "test",
        "status": "completed",
        "mode": "local",
        "executed_or_dry_run": "executed",
        "classified_failure": "test", "failure_category": "test",
        "candidate_strategies_count": 0, "candidate_strategies": [],
        "candidate_designs_count": 2, "candidate_designs": [],
        "plan_scores": [],
        "selected_design": None, "selected_design_rank": None,
        "executable_selection_reason": "",
        "selected_designs": [],
        "skipped_higher_ranked_designs": [],
        "attempted_designs": [
            {
                "design_id": "objective_redesign_simpler_metric_mse_only",
                "status": "completed",
                "evidence_level": "lightweight_scientific_execution",
                "handler_id": "objective_redesign_simpler_metric",
                "metrics": {
                    "mse_after": 0.03, "psnr_after": 15.0, "improvement_detected": True,
                },
                "errors": [],
            },
            {
                "design_id": "param_reduction_sweep",
                "status": "completed",
                "evidence_level": "lightweight_scientific_execution",
                "handler_id": "param_reduction_sweep",
                "metrics": {
                    "mse_after": 0.04, "psnr_after": 14.0, "improvement_detected": True,
                },
                "errors": [],
            },
        ],
        "execution_result": {},
        "execution_results": [],
        "claim_gate_decision": {},
        "claim_gate_decisions": [],
        "memory_updates": [], "memory_updated": False,
        "state_snapshots_count": 0, "state_snapshot_refs": [],
        "event_count": 0, "event_log_path": "", "report_path": "",
        "final_recommendation": "",
        "errors": [],
    }
    md = _markdown("test_comp", r)
    assert "## 7c. Scientific Handler Comparison" in md
    assert "objective_redesign_simpler_metric_mse_only" in md
    assert "param_reduction_sweep" in md
