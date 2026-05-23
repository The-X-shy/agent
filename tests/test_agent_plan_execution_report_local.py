import json
from pathlib import Path

from optiresearch.reports.agent_plan_execution_report import (
    export_agent_plan_execution_report,
)


def test_local_report_contains_selection_attempt_claim_memory_and_state_sections(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = Path("workspace/agent_plan_executions/plan_exec_report_test")
    run_dir.mkdir(parents=True)
    (run_dir / "execution_result.json").write_text(
        json.dumps(
            {
                "execution_id": "plan_exec_report_test",
                "objective": "recover from native GeoLens optical update instability",
                "status": "completed",
                "mode": "local",
                "selected_design": "alt_param_diffractive_sweep",
                "selected_design_rank": 2,
                "skipped_higher_ranked_designs": [
                    {
                        "design_id": "real_data_request_req",
                        "rank": 1,
                        "skipped_reason": "needs_user_data",
                    }
                ],
                "attempted_designs": [
                    {"design_id": "alt_param_diffractive_sweep", "status": "unsupported"},
                    {"design_id": "report_negative_result_doc", "status": "completed"},
                ],
                "execution_result": {
                    "status": "completed",
                    "design_id": "report_negative_result_doc",
                    "evidence_level": "report_only",
                    "metrics": {"report_generated": True},
                    "errors": [],
                },
                "claim_gate_decision": {
                    "decision": "supported",
                    "max_allowed_claim": "report_only",
                },
                "memory_updated": True,
                "state_snapshots_count": 1,
                "final_recommendation": "Report-only fallback completed.",
            }
        ),
        encoding="utf-8",
    )

    path = export_agent_plan_execution_report("plan_exec_report_test")
    text = path.read_text(encoding="utf-8")

    assert "## 5. Executable Selection" in text
    assert "Skipped Higher-Ranked Designs" in text
    assert "Attempted Designs" in text
    assert "## 7. Local Execution Result" in text
    assert "## 8. ClaimGate Outcome" in text
    assert "## 10. Memory / State Updates" in text
    assert "Report-only fallback completed." in text
