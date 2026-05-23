"""Test report generation includes scientific execution metrics section."""

import json
from pathlib import Path

from optiresearch.reports.agent_plan_execution_report import (
    export_agent_plan_execution_report,
    _markdown,
)


def test_report_includes_scientific_execution_metrics():
    """Report should contain 7a. Scientific Execution Metrics when evidence is lightweight."""
    r = {
        "execution_id": "test_sci_001",
        "objective": "test",
        "status": "completed",
        "mode": "local",
        "executed_or_dry_run": "executed",
        "classified_failure": "unstable_native_geolens_update",
        "failure_category": "gradient_instability",
        "candidate_strategies_count": 1,
        "candidate_strategies": [],
        "candidate_designs_count": 1,
        "candidate_designs": [],
        "plan_scores": [],
        "selected_design": "objective_redesign_simpler_metric_mse_only",
        "selected_design_rank": 1,
        "executable_selection_reason": "test",
        "selected_designs": [],
        "skipped_higher_ranked_designs": [],
        "attempted_designs": [
            {
                "design_id": "objective_redesign_simpler_metric_mse_only",
                "status": "completed",
                "evidence_level": "lightweight_scientific_execution",
                "errors": [],
            },
        ],
        "execution_result": {
            "status": "completed",
            "design_id": "objective_redesign_simpler_metric_mse_only",
            "task_type": "stable_lens_hsi_codesign",
            "backend_id": "phase_to_fft_proxy",
            "evidence_level": "lightweight_scientific_execution",
            "metrics": {
                "reconstruction_loss_before": 0.1234,
                "reconstruction_loss_after": 0.0567,
                "best_reconstruction_loss": 0.0456,
                "mse_before": 0.1234,
                "mse_after": 0.0567,
                "psnr_before": 9.09,
                "psnr_after": 12.46,
                "improvement_detected": True,
                "metrics_valid": True,
                "execution_time_sec": 2.5,
                "synthetic_data": True,
                "physical_backend": False,
                "mse_only_objective": True,
            },
            "artifacts": [],
            "errors": [],
            "caveats": [
                "MSE-only synthetic HSI experiment",
                "Synthetic HSI data",
            ],
        },
        "execution_results": [],
        "claim_gate_decision": {
            "decision": "supported",
            "max_allowed_claim": "lightweight_scientific_execution",
        },
        "claim_gate_decisions": [],
        "memory_updates": [],
        "memory_updated": True,
        "state_snapshots_count": 1,
        "state_snapshot_refs": [],
        "event_count": 10,
        "event_log_path": "workspace/agent_plan_executions/test_sci_001/events.json",
        "report_path": "workspace/agent_plan_executions/test_sci_001/plan_execution_report.md",
        "final_recommendation": "test",
        "errors": [],
    }

    md = _markdown("test_sci_001", r)
    assert "## 7a. Scientific Execution Metrics" in md
    assert "0.1234" in md
    assert "0.0567" in md
    assert "9.09" in md
    assert "12.46" in md
    assert "Improvement Detected" in md
    assert "Metrics Valid" in md
    assert "Synthetic Data" in md
    assert "Physical Backend" in md
    assert "MSE-only Objective" in md


def test_report_omits_scientific_section_for_other_evidence():
    """Report should NOT include 7a section for non-scientific evidence."""
    r = {
        "execution_id": "test_002",
        "objective": "test",
        "status": "completed",
        "mode": "local",
        "executed_or_dry_run": "executed",
        "classified_failure": "test",
        "failure_category": "test",
        "candidate_strategies_count": 1,
        "candidate_strategies": [],
        "candidate_designs_count": 1,
        "candidate_designs": [],
        "plan_scores": [],
        "selected_design": None,
        "selected_design_rank": None,
        "executable_selection_reason": "",
        "selected_designs": [],
        "skipped_higher_ranked_designs": [],
        "attempted_designs": [],
        "execution_result": {
            "status": "completed",
            "design_id": "report_negative_result_doc",
            "evidence_level": "report_only",
            "metrics": {"report_generated": True},
            "artifacts": [],
            "errors": [],
        },
        "execution_results": [],
        "claim_gate_decision": {},
        "claim_gate_decisions": [],
        "memory_updates": [],
        "memory_updated": False,
        "state_snapshots_count": 0,
        "state_snapshot_refs": [],
        "event_count": 0,
        "event_log_path": "",
        "report_path": "",
        "final_recommendation": "",
        "errors": [],
    }

    md = _markdown("test_002", r)
    assert "## 7a. Scientific Execution Metrics" not in md
    assert "## 7. Local Execution Result" in md


def test_scientific_report_contains_caveats():
    r = {
        "execution_id": "test_caveats",
        "objective": "test",
        "status": "completed",
        "mode": "local",
        "executed_or_dry_run": "executed",
        "classified_failure": "test",
        "failure_category": "test",
        "candidate_strategies_count": 0,
        "candidate_strategies": [],
        "candidate_designs_count": 0,
        "candidate_designs": [],
        "plan_scores": [],
        "selected_design": None,
        "selected_design_rank": None,
        "executable_selection_reason": "",
        "selected_designs": [],
        "skipped_higher_ranked_designs": [],
        "attempted_designs": [],
        "execution_result": {
            "status": "completed",
            "design_id": "test",
            "evidence_level": "lightweight_scientific_execution",
            "metrics": {
                "reconstruction_loss_before": 0.1,
                "reconstruction_loss_after": 0.05,
                "best_reconstruction_loss": 0.04,
                "mse_before": 0.1,
                "mse_after": 0.05,
                "psnr_before": 10.0,
                "psnr_after": 13.0,
                "improvement_detected": True,
                "metrics_valid": True,
                "execution_time_sec": 1.0,
                "synthetic_data": True,
                "physical_backend": False,
                "mse_only_objective": True,
            },
            "artifacts": [],
            "errors": [],
        },
        "execution_results": [],
        "claim_gate_decision": {},
        "claim_gate_decisions": [],
        "memory_updates": [],
        "memory_updated": False,
        "state_snapshots_count": 0,
        "state_snapshot_refs": [],
        "event_count": 0,
        "event_log_path": "",
        "report_path": "",
        "final_recommendation": "",
        "errors": [],
    }

    md = _markdown("test_caveats", r)
    assert "### Caveats" in md
    assert "not native DeepLens simulation" in md
