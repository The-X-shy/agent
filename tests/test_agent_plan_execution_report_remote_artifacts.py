import json
from pathlib import Path

from optiresearch.reports.agent_plan_execution_report import export_agent_plan_execution_report


def test_agent_plan_execution_report_contains_remote_artifacts_and_event_sequence(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    run_dir = Path("workspace/agent_plan_executions/remote_report_test")
    run_dir.mkdir(parents=True)
    (run_dir / "execution_result.json").write_text(
        json.dumps(
            {
                "execution_id": "remote_report_test",
                "objective": "validate remote handler",
                "status": "completed",
                "mode": "remote_opt_in",
                "executed_or_dry_run": "executed",
                "selected_design": "remote_native_geolens_validation",
                "attempted_designs": [
                    {
                        "design_id": "remote_native_geolens_validation",
                        "status": "completed",
                        "evidence_level": "native_lens_simulation",
                        "errors": [],
                    }
                ],
                "execution_result": {
                    "status": "completed",
                    "design_id": "remote_native_geolens_validation",
                    "handler_id": "remote_native_geolens_validation",
                    "backend_id": "deeplens_geolens_geometric",
                    "evidence_level": "native_lens_simulation",
                    "execution_target": "remote_wsl",
                    "remote_worker_id": "windows_wsl",
                    "remote_job_id": "remote_job_abc",
                    "remote_validation_passed": True,
                    "run_id": "run_abc",
                    "execution_fidelity": "deeplens_native_geometric",
                    "proxy_fallback_used": False,
                    "deeplens_native_psf_path": "geolens.psf_geometric",
                    "full_wave_optics": False,
                    "phase_to_fft_proxy_used": False,
                    "metrics": {"accepted_update_count": 1},
                    "artifacts": ["workspace/remote_jobs/remote_job_abc/result.json"],
                    "artifact_return_path": "workspace/remote_jobs/remote_job_abc",
                    "errors": [],
                    "caveats": [],
                },
                "claim_gate_decision": {
                    "decision": "supported",
                    "max_allowed_claim": "native_lens_simulation",
                    "final_claim_ceiling": "native_lens_simulation",
                    "ceiling_source": "handler",
                },
                "memory_updated": True,
                "memory_updates": ["plan_outcome_remote_report_test"],
                "state_snapshots_count": 1,
                "event_log_path": "workspace/agent_plan_executions/remote_report_test/events.json",
                "final_recommendation": "Remote execution completed.",
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "events.json").write_text(
        json.dumps(
            [
                {"event_type": "remote_execution_requested", "payload": {}},
                {"event_type": "remote_execution_started", "payload": {}},
                {"event_type": "remote_execution_completed", "payload": {}},
                {"event_type": "remote_validation_passed", "payload": {}},
                {"event_type": "artifact_ingested", "payload": {}},
            ]
        ),
        encoding="utf-8",
    )

    path = export_agent_plan_execution_report("remote_report_test")
    text = path.read_text(encoding="utf-8")

    assert "Remote Job" in text
    assert "Remote Result Ingestion" in text
    assert "Artifact Return Path" in text
    assert "Remote Claim Ceiling" in text
    assert "Remote Event Sequence" in text
    assert "remote_job_abc" in text
    assert "geolens.psf_geometric" in text
    assert "remote_validation_passed" in text
