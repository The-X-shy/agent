"""Mock tests for remote native optimization probe job."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from optiresearch.schemas.remote import RemoteJobResult, RemoteJobSpec, RemoteWorkerSpec
from optiresearch.remote.worker_registry import RemoteWorkerRegistry


class MockRunner:
    def __init__(self):
        self.jobs: list[tuple[RemoteWorkerSpec, RemoteJobSpec]] = []

    def run_remote_job(self, worker: RemoteWorkerSpec, job: RemoteJobSpec) -> RemoteJobResult:
        self.jobs.append((worker, job))
        return RemoteJobResult(
            job_id=job.job_id,
            status="succeeded",
            remote_run_id=f"remote_{job.job_id}",
            started_at="2026-05-21T00:00:00Z",
            finished_at="2026-05-21T00:01:00Z",
            command=["python", "-m", "optiresearch.cli", "run-native-optimization-probe"],
            stdout_path="/tmp/stdout.txt",
            stderr_path="/tmp/stderr.txt",
            remote_output_dir="/tmp/output",
            local_output_dir="/tmp/local",
            error_code=None,
            metrics_summary={"differentiable": True, "gradient_norm": 0.15},
            caveats=[],
        )


def test_remote_native_probe_builds_job(tmp_path, monkeypatch):
    """Remote native optimization probe should build correct RemoteJobSpec."""
    monkeypatch.setenv("OPTIRESEARCH_REMOTE_WORKER_ROOT", str(tmp_path / "remote_workers"))
    worker = RemoteWorkerSpec(
        worker_id="windows_wsl",
        host="192.168.1.100",
        username="testuser",
        remote_project_dir="/mnt/d/agent",
        remote_workspace_dir="/mnt/d/agent/workspace",
        python_executable="/mnt/d/agent/run_agent_python.sh",
    )
    RemoteWorkerRegistry().add_worker(worker)

    from optiresearch.runtime.remote_jobs import _job as make_remote_job
    job = make_remote_job(
        "native_optimization_probe",
        objective="Test native probe",
        cli_args={
            "lens_class": "ParaxialLens",
            "objective": "minimize_psf_width",
            "max_steps": 2,
            "learning_rate": 1e-3,
            "device": "cpu",
            "strict_native": True,
            "allow_adapter_proxy": False,
        },
        timeout_seconds=1800,
        expected_outputs=["probe_result.json", "psf_before.npz", "psf_after.npz"],
    )
    assert job.job_type == "native_optimization_probe"
    assert job.cli_args["lens_class"] == "ParaxialLens"


def test_mock_runner_produces_structured_result(tmp_path, monkeypatch):
    """Mock runner should produce a valid RemoteJobResult for native probe."""
    monkeypatch.setenv("OPTIRESEARCH_REMOTE_WORKER_ROOT", str(tmp_path / "remote_workers"))
    worker = RemoteWorkerSpec(
        worker_id="windows_wsl",
        host="192.168.1.100",
        username="testuser",
        remote_project_dir="/mnt/d/agent",
        remote_workspace_dir="/mnt/d/agent/workspace",
        python_executable="/mnt/d/agent/run_agent_python.sh",
    )
    RemoteWorkerRegistry().add_worker(worker)

    runner = MockRunner()

    from optiresearch.runtime.remote_jobs import execute_remote_job
    from optiresearch.schemas.remote import RemoteJobSpec
    job = RemoteJobSpec(
        job_id="remote_probe_test_1",
        job_type="native_optimization_probe",
        objective="Test native probe",
        timeout_seconds=1800,
    )
    payload = execute_remote_job("windows_wsl", job, runner=runner, ingest=False)
    assert payload["result"].status == "succeeded"
    assert payload["result"].job_id == "remote_probe_test_1"


def test_native_probe_remote_job_type_valid():
    """native_optimization_probe should be a valid RemoteJobType."""
    from optiresearch.schemas.remote import RemoteJobSpec
    job = RemoteJobSpec(
        job_id="test_job_type",
        job_type="native_optimization_probe",
        objective="Test",
        timeout_seconds=1800,
    )
    assert job.job_type == "native_optimization_probe"
