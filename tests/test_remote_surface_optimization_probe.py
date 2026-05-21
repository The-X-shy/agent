"""Tests for Phase 19B remote surface optimization probe plumbing."""

from __future__ import annotations

from optiresearch.remote.command_allowlist import validate_remote_command
from optiresearch.remote.ssh_runner import build_job_command
from optiresearch.runtime.remote_jobs import run_remote_deeplens_surface_optimization_probe
from optiresearch.schemas.remote import RemoteJobResult, RemoteJobSpec, RemoteWorkerSpec
from optiresearch.remote.worker_registry import RemoteWorkerRegistry


class MockRunner:
    def __init__(self):
        self.jobs: list[RemoteJobSpec] = []

    def run_remote_job(self, worker: RemoteWorkerSpec, job: RemoteJobSpec) -> RemoteJobResult:
        self.jobs.append(job)
        return RemoteJobResult(
            job_id=job.job_id,
            status="succeeded",
            remote_run_id=f"remote_{job.job_id}",
            started_at="2026-05-21T00:00:00Z",
            finished_at="2026-05-21T00:01:00Z",
            command=["python", "-m", "optiresearch.cli", "run-deeplens-surface-optimization-probe"],
            stdout_path="/tmp/stdout.txt",
            stderr_path="/tmp/stderr.txt",
            remote_output_dir="/tmp/output",
            local_output_dir="/tmp/local",
            error_code=None,
            metrics_summary={"differentiable": True, "gradient_norm": 0.2},
            caveats=[],
        )


def _worker() -> RemoteWorkerSpec:
    return RemoteWorkerSpec(
        worker_id="windows_wsl",
        host="192.168.1.100",
        username="testuser",
        remote_project_dir="/mnt/d/agent",
        remote_workspace_dir="/mnt/d/agent/workspace",
        python_executable="/mnt/d/agent/run_agent_python.sh",
    )


def test_allowlist_accepts_surface_and_lensfile_probe_commands():
    surface_command = [
        "/mnt/d/agent/run_agent_python.sh",
        "-m",
        "optiresearch.cli",
        "run-deeplens-surface-optimization-probe",
        "--surface",
        "Fresnel",
        "--objective",
        "minimize_phase_variance",
        "--max-steps",
        "3",
        "--learning-rate",
        "0.001",
        "--device",
        "cpu",
        "--remote-job-id",
        "remote_job_1",
    ]
    lensfile_command = [
        "/mnt/d/agent/run_agent_python.sh",
        "-m",
        "optiresearch.cli",
        "run-deeplens-lensfile-optimization-probe",
        "--lens-class",
        "GeoLens",
        "--max-files",
        "5",
        "--max-steps",
        "2",
        "--learning-rate",
        "0.001",
        "--device",
        "cpu",
        "--remote-job-id",
        "remote_job_2",
    ]

    assert validate_remote_command(surface_command)["cli_command"] == "run-deeplens-surface-optimization-probe"
    assert validate_remote_command(lensfile_command)["cli_command"] == "run-deeplens-lensfile-optimization-probe"


def test_remote_surface_probe_builds_job_without_real_wsl(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REMOTE_WORKER_ROOT", str(tmp_path / "workers"))
    RemoteWorkerRegistry().add_worker(_worker())
    runner = MockRunner()

    payload = run_remote_deeplens_surface_optimization_probe(
        "windows_wsl",
        surface="Fresnel",
        objective="minimize_phase_variance",
        max_steps=3,
        runner=runner,
        ingest=False,
    )

    assert payload["result"].status == "succeeded"
    assert runner.jobs[0].job_type == "deeplens_surface_optimization_probe"
    assert runner.jobs[0].cli_args["surface"] == "Fresnel"


def test_build_job_command_maps_native_surface_and_lensfile_jobs():
    worker = _worker()
    surface_job = RemoteJobSpec(
        job_id="job_surface",
        job_type="deeplens_surface_optimization_probe",
        objective="Surface probe",
        cli_args={"surface": "Fresnel", "objective": "minimize_phase_variance", "max_steps": 3},
        timeout_seconds=1800,
    )
    lensfile_job = RemoteJobSpec(
        job_id="job_lensfile",
        job_type="deeplens_lensfile_optimization_probe",
        objective="Lens file probe",
        cli_args={"lens_class": "GeoLens", "max_files": 5, "max_steps": 2},
        timeout_seconds=1800,
    )

    assert build_job_command(worker, surface_job)[3] == "run-deeplens-surface-optimization-probe"
    assert build_job_command(worker, lensfile_job)[3] == "run-deeplens-lensfile-optimization-probe"
