import pytest

from optiresearch.schemas.remote import RemoteJobResult, RemoteJobSpec, RemoteWorkerSpec


def test_remote_worker_defaults_and_dump():
    worker = RemoteWorkerSpec(
        worker_id="windows_wsl",
        host="wslbox",
        username="ysl",
        remote_project_dir="/mnt/d/agent",
        remote_workspace_dir="/mnt/d/agent/workspace",
        python_executable="/mnt/d/agent/run_agent_python.sh",
        backend_tags=["wsl", "deeplens"],
        capabilities={"torch": True},
    )

    payload = worker.model_dump()

    assert payload["port"] == 22
    assert payload["ssh_key_path"] is None
    assert payload["max_runtime_seconds"] == 3600
    assert payload["environment_name"] is None


def test_remote_job_rejects_unknown_type():
    with pytest.raises(ValueError):
        RemoteJobSpec(
            job_id="job_bad",
            job_type="shell",
            objective="bad",
            cli_args={},
            input_artifacts=[],
            expected_outputs=[],
            timeout_seconds=60,
            evidence_policy={},
        )


def test_remote_job_result_structured_error_shape():
    result = RemoteJobResult(
        job_id="job_1",
        status="failed",
        remote_run_id=None,
        started_at="2026-05-20T00:00:00Z",
        finished_at="2026-05-20T00:00:01Z",
        command=["/mnt/d/agent/run_agent_python.sh", "-m", "optiresearch.cli", "run-codesign-loop"],
        stdout_path="workspace/remote_jobs/job_1/logs/stdout.txt",
        stderr_path="workspace/remote_jobs/job_1/logs/stderr.txt",
        remote_output_dir="/mnt/d/agent/workspace/remote_jobs/job_1",
        local_output_dir="workspace/remote_jobs/job_1",
        artifact_manifest={},
        metrics_summary={},
        error_code="DEEPLENS_UNAVAILABLE",
        caveats=["strict mode returned structured error"],
    )

    dumped = result.model_dump()
    assert dumped["status"] == "failed"
    assert dumped["error_code"] == "DEEPLENS_UNAVAILABLE"
    assert dumped["command"][0].endswith("run_agent_python.sh")
