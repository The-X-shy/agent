from pathlib import Path

from optiresearch.agent_system.event_bus import get_event_bus
from optiresearch.remote.worker_registry import RemoteWorkerRegistry
from optiresearch.runtime import agent_plan_execution_loop as loop
from optiresearch.schemas.agent_plan_execution import AgentPlanExecutionSpec
from optiresearch.schemas.remote import RemoteWorkerSpec


def test_agent_plan_execution_remote_result_updates_claim_state_memory_and_events(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPTIRESEARCH_REMOTE_WORKER_ROOT", str(tmp_path / "remote_workers"))
    get_event_bus().clear()
    RemoteWorkerRegistry().add_worker(
        RemoteWorkerSpec(
            worker_id="windows_wsl",
            host="wslbox",
            username="ysl",
            remote_project_dir="/mnt/d/agent",
            remote_workspace_dir="/mnt/d/agent/workspace",
            python_executable="/mnt/d/agent/run_agent_python.sh",
            backend_tags=["windows_wsl", "deeplens_available", "geolens_psf_geometric"],
            capabilities={
                "allowed_commands": ["run-deeplens-native-geolens-hsi-codesign"],
                "artifact_return_path": "/mnt/d/agent/workspace/remote_jobs",
            },
        )
    )

    def fake_execute_remote_design(design, worker_id):
        return {
            "status": "completed",
            "outcome": "remote_execution_completed",
            "design_id": design.design_id,
            "handler_id": "remote_native_geolens_validation",
            "task_type": design.task_type,
            "backend_id": "deeplens_geolens_geometric",
            "evidence_level": "native_lens_simulation",
            "execution_target": "remote_wsl",
            "remote_worker_id": worker_id,
            "remote_job_id": "remote_job_abc",
            "remote_validation_passed": True,
            "run_id": "run_abc",
            "execution_fidelity": "deeplens_native_geometric",
            "proxy_fallback_used": False,
            "deeplens_native_psf_path": "geolens.psf_geometric",
            "full_wave_optics": False,
            "phase_to_fft_proxy_used": False,
            "metrics": {"accepted_update_count": 1, "improvement_detected": True},
            "artifacts": ["workspace/remote_jobs/remote_job_abc/result.json"],
            "artifact_return_path": "workspace/remote_jobs/remote_job_abc",
            "errors": [],
            "caveats": [],
        }

    monkeypatch.setattr(loop, "_execute_remote_design", fake_execute_remote_design)

    result = loop.run_agent_plan_execution(
        AgentPlanExecutionSpec(
            execution_id="remote_plan_test",
            objective="validate native GeoLens HSI path on WSL through remote-aware handler",
            mode="remote_opt_in",
            allow_remote=True,
            remote_worker_id="windows_wsl",
            execute_top_k=1,
        )
    )

    assert result.status == "completed"
    # selected_design may vary based on scoring — remote handler still selectable
    assert result.selected_design
    assert result.execution_result["execution_target"] == "remote_wsl"
    assert result.execution_result["remote_validation_passed"] is True
    assert result.execution_result["artifact_return_path"].endswith("remote_job_abc")
    assert result.claim_gate_decision["final_claim_ceiling"] == "native_lens_simulation"
    assert result.memory_updated is True

    state_path = Path("workspace/agent_state/current_state.json")
    state_text = state_path.read_text(encoding="utf-8")
    assert "remote_job_abc" in state_text
    assert "windows_wsl" in state_text

    event_types = [event.event_type for event in get_event_bus().list_events()]
    assert event_types.index("remote_execution_requested") < event_types.index("remote_execution_started")
    assert "remote_execution_completed" in event_types
    assert "remote_validation_passed" in event_types
