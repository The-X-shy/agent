import json

from optiresearch.agent_system.event_bus import get_event_bus
from optiresearch.schemas.remote import RemoteJobResult
from optiresearch.skills.runtime_v2 import SkillRuntimeV2


def test_skill_runtime_remote_native_geolens_parses_result_and_publishes_events(
    tmp_path, monkeypatch
):
    get_event_bus().clear()
    local_dir = tmp_path / "remote_jobs" / "remote_job_abc"
    local_dir.mkdir(parents=True)
    (local_dir / "result.json").write_text(
        json.dumps(
            {
                "reconstruction_loss_before": 0.5,
                "reconstruction_loss_after": 0.4,
                "improvement_detected": True,
                "accepted_update_count": 1,
            }
        ),
        encoding="utf-8",
    )
    manifest = {"artifacts": [{"path": "result.json", "artifact_type": "manifest"}]}
    remote_result = RemoteJobResult(
        job_id="remote_job_abc",
        status="succeeded",
        remote_run_id="run_abc",
        started_at="2026-05-24T00:00:00Z",
        finished_at="2026-05-24T00:00:01Z",
        command=["/mnt/d/agent/run_agent_python.sh", "-m", "optiresearch.cli", "run-deeplens-native-geolens-hsi-codesign"],
        stdout_path=str(local_dir / "stdout.txt"),
        stderr_path=str(local_dir / "stderr.txt"),
        remote_output_dir="/mnt/d/agent/workspace/remote_jobs/remote_job_abc",
        local_output_dir=str(local_dir),
        artifact_manifest=manifest,
        metrics_summary={
            "status": "succeeded",
            "remote_run_id": "run_abc",
            "evidence_level": "native_lens_simulation",
            "execution_fidelity": "deeplens_native_geometric",
            "proxy_fallback_used": False,
            "fallback_used": False,
            "deeplens_native_psf_path": "geolens.psf_geometric",
            "full_wave_optics": False,
            "phase_to_fft_proxy_used": False,
        },
        error_code=None,
        caveats=[],
    )

    def fake_remote_job(**kwargs):
        return {"result": remote_result, "ingestion": {"artifact_ids": ["artifact_1"]}}

    import optiresearch.runtime.remote_jobs as remote_jobs

    monkeypatch.setattr(
        remote_jobs,
        "run_remote_deeplens_native_geolens_hsi_codesign",
        fake_remote_job,
    )

    skill = SkillRuntimeV2().execute_skill(
        "remote_execution",
        {
            "allow_remote": True,
            "worker_id": "windows_wsl",
            "design_id": "remote_native_geolens_validation",
            "handler_id": "remote_native_geolens_validation",
            "spec_payload": {"candidate": "auto:cooke"},
        },
    )

    output = skill.output
    assert skill.status == "succeeded"
    assert output["remote_validation_passed"] is True
    assert output["remote_handler_result"]["remote_job_id"] == "remote_job_abc"
    assert output["remote_handler_result"]["proxy_fallback_used"] is False
    assert output["remote_handler_result"]["phase_to_fft_proxy_used"] is False
    assert output["artifacts"]

    event_types = [event.event_type for event in get_event_bus().list_events()]
    assert "remote_execution_requested" in event_types
    assert "remote_execution_started" in event_types
    assert "remote_execution_completed" in event_types
    assert "remote_validation_passed" in event_types
    assert "artifact_ingested" in event_types
