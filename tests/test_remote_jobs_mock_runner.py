from optiresearch.runtime.remote_jobs import _extract_metrics, run_remote_deeplens_source_smoke
from optiresearch.schemas.remote import RemoteJobResult, RemoteWorkerSpec


class MockRunner:
    def __init__(self):
        self.jobs = []

    def run_remote_job(self, worker, job):
        self.jobs.append((worker, job))
        return RemoteJobResult(
            job_id=job.job_id,
            status="succeeded",
            remote_run_id="remote_smoke",
            started_at="2026-05-20T00:00:00Z",
            finished_at="2026-05-20T00:00:01Z",
            command=[worker.python_executable, "-m", "optiresearch.cli", "run-deeplens-source-smoke"],
            stdout_path="stdout.txt",
            stderr_path="stderr.txt",
            remote_output_dir=f"{worker.remote_workspace_dir}/remote_jobs/{job.job_id}",
            local_output_dir=f"workspace/remote_jobs/{job.job_id}",
            artifact_manifest={"artifacts": []},
            metrics_summary={"available": True},
            error_code=None,
            caveats=[],
        )


def test_run_remote_smoke_builds_job_and_uses_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REMOTE_WORKER_ROOT", str(tmp_path / "remote_workers"))
    worker = RemoteWorkerSpec(
        worker_id="windows_wsl",
        host="wslbox",
        username="ysl",
        remote_project_dir="/mnt/d/agent",
        remote_workspace_dir="/mnt/d/agent/workspace",
        python_executable="/mnt/d/agent/run_agent_python.sh",
        backend_tags=["wsl"],
        capabilities={},
    )
    from optiresearch.remote.worker_registry import RemoteWorkerRegistry

    RemoteWorkerRegistry().add_worker(worker)
    runner = MockRunner()

    payload = run_remote_deeplens_source_smoke("windows_wsl", runner=runner, ingest=False)

    assert payload["result"].status == "succeeded"
    assert runner.jobs[0][1].job_type == "deeplens_source_smoke"
    assert runner.jobs[0][1].cli_args == {}


def test_extract_metrics_ignores_json_arrays(tmp_path):
    path = tmp_path / "history.json"
    path.write_text('[{"score": 1.0}]', encoding="utf-8")

    assert _extract_metrics(path) == {}
