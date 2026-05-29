"""Agent plan report tests for component surrogate HSI co-design section."""

import json
from pathlib import Path

from optiresearch.reports.agent_plan_execution_report import export_agent_plan_execution_report


def test_agent_plan_report_includes_component_surrogate_hsi_section(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    execution_id = "exec_component_surrogate"
    run_dir = Path("workspace/agent_plan_executions") / execution_id
    run_dir.mkdir(parents=True)
    result = {
        "objective": "recover",
        "status": "completed",
        "mode": "local",
        "executed_or_dry_run": "executed",
        "candidate_strategies": [],
        "candidate_designs": [],
        "plan_scores": [],
        "selected_design": "component_surrogate_fresnel_hsi_codesign_design",
        "selected_design_rank": 1,
        "selected_designs": [],
        "attempted_designs": [
            {
                "design_id": "component_surrogate_fresnel_hsi_codesign_design",
                "status": "completed",
                "evidence_level": "component_surrogate_hsi_codesign",
                "metrics": {"component_grad_norm_max": 0.1},
            }
        ],
        "execution_result": {
            "status": "completed",
            "design_id": "component_surrogate_fresnel_hsi_codesign_design",
            "task_type": "component_surrogate_hsi_codesign",
            "backend_id": "component_surrogate_psf",
            "evidence_level": "component_surrogate_hsi_codesign",
            "metrics": {
                "component_type": "fresnel",
                "reconstruction_loss_before": 1.0,
                "reconstruction_loss_after": 0.9,
                "mse_before": 1.0,
                "mse_after": 0.9,
                "psnr_before": 10.0,
                "psnr_after": 11.0,
                "sam_before": 0.5,
                "sam_after": 0.4,
                "component_grad_norm_max": 0.1,
                "component_parameter_changed": True,
                "psf_requires_grad": True,
                "loss_requires_grad": True,
            },
            "artifacts": ["workspace/component_surrogate_hsi/run/result.json"],
            "errors": [],
            "handler_claim_ceiling": "component_surrogate_hsi_codesign",
            "design_backend_claim_ceiling": "component_surrogate_hsi_codesign",
            "dataset_claim_ceiling": "lightweight_scientific_execution",
            "execution_fidelity_claim_ceiling": "component_surrogate_hsi_codesign",
        },
        "claim_gate_decision": {
            "decision": "supported",
            "max_allowed_claim": "component_surrogate_hsi_codesign",
            "final_claim_ceiling": "component_surrogate_hsi_codesign",
            "safe_wording": "Component surrogate HSI co-design completed",
        },
    }
    (run_dir / "execution_result.json").write_text(json.dumps(result), encoding="utf-8")

    path = export_agent_plan_execution_report(execution_id)

    content = path.read_text()
    assert "Component Surrogate HSI Co-design" in content
    assert "component_grad_norm_max" in content
    assert "What Not To Claim" in content
