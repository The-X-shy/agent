"""Tests for remote native waveoptics HSI CLI argument plumbing."""

from __future__ import annotations

from optiresearch.remote.command_allowlist import validate_remote_command
from optiresearch.remote.ssh_runner import build_job_command
from optiresearch.runtime.remote_jobs import run_remote_native_waveoptics_hsi_codesign
from optiresearch.schemas.remote import RemoteJobResult, RemoteJobSpec, RemoteWorkerSpec


class _MockRunner:
    def __init__(self):
        self.jobs: list[RemoteJobSpec] = []

    def run_remote_job(self, worker: RemoteWorkerSpec, job: RemoteJobSpec) -> RemoteJobResult:
        self.jobs.append(job)
        return RemoteJobResult(
            job_id=job.job_id,
            status="succeeded",
            remote_run_id="remote_wave_hsi_test",
            started_at="2026-05-22T00:00:00Z",
            finished_at="2026-05-22T00:00:01Z",
            command=[],
            stdout_path="stdout.txt",
            stderr_path="stderr.txt",
            remote_output_dir="/mnt/d/agent/workspace/remote_jobs/test",
            local_output_dir="workspace/remote_jobs/test",
            error_code=None,
            metrics_summary={},
            caveats=[],
        )


def _worker() -> RemoteWorkerSpec:
    return RemoteWorkerSpec(
        worker_id="windows_wsl",
        host="localhost",
        username="user",
        remote_project_dir="/mnt/d/agent",
        remote_workspace_dir="/mnt/d/agent/workspace",
        python_executable="/mnt/d/agent/run_agent_python.sh",
    )


def test_remote_native_waveoptics_hsi_job_keeps_dataset_arg():
    runner = _MockRunner()

    run_remote_native_waveoptics_hsi_codesign(
        "windows_wsl",
        "GeoLensCooke",
        "differentiable_linear",
        dataset="synthetic",
        max_steps=3,
        device="cpu",
        runner=runner,
        ingest=False,
    )

    assert runner.jobs
    assert runner.jobs[0].cli_args["dataset"] == "synthetic"


def test_remote_native_waveoptics_hsi_command_allows_dataset_and_remote_job_id():
    job = RemoteJobSpec(
        job_id="remote_job_1436c05c2c4d6359",
        job_type="native_waveoptics_hsi_codesign",
        objective="Native waveoptics HSI",
        cli_args={
            "candidate": "GeoLensCooke",
            "reconstructor": "differentiable_linear",
            "dataset": "synthetic",
            "max_steps": 3,
            "device": "cpu",
        },
        timeout_seconds=3600,
    )

    command = build_job_command(_worker(), job)

    assert "--dataset" in command
    assert command[command.index("--dataset") + 1] == "synthetic"
    assert "--remote-job-id" in command
    assert validate_remote_command(command)["cli_command"] == "run-native-waveoptics-hsi-codesign"


def test_cli_accepts_dataset_argument(monkeypatch, capsys):
    from optiresearch import cli
    from optiresearch.schemas.remote import RemoteJobResult

    calls = {}

    def fake_remote(worker_id, candidate, reconstructor, **kwargs):
        calls.update(
            {
                "worker_id": worker_id,
                "candidate": candidate,
                "reconstructor": reconstructor,
                **kwargs,
            }
        )
        return {
            "result": RemoteJobResult(
                job_id="remote_job_1436c05c2c4d6359",
                status="succeeded",
                remote_run_id="remote_wave_hsi_test",
                started_at="2026-05-22T00:00:00Z",
                finished_at="2026-05-22T00:00:01Z",
                command=[],
                stdout_path="stdout.txt",
                stderr_path="stderr.txt",
                remote_output_dir="/mnt/d/agent/workspace/remote_jobs/remote_job_1436c05c2c4d6359",
                local_output_dir="workspace/remote_jobs/remote_job_1436c05c2c4d6359",
                error_code=None,
                metrics_summary={},
                caveats=[],
            )
        }

    monkeypatch.setattr(cli, "run_remote_native_waveoptics_hsi_codesign", fake_remote)

    cli.main(
        [
            "run-remote-native-waveoptics-hsi-codesign",
            "--worker-id",
            "windows_wsl",
            "--candidate",
            "GeoLensCooke",
            "--reconstructor",
            "differentiable_linear",
            "--dataset",
            "synthetic",
            "--max-steps",
            "3",
            "--device",
            "cpu",
        ]
    )

    assert calls["dataset"] == "synthetic"
    assert calls["max_steps"] == 3
    assert "remote_job_1436c05c2c4d6359" in capsys.readouterr().out
