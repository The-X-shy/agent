import subprocess

from optiresearch.remote.ssh_runner import SSHRemoteRunner
from optiresearch.schemas.remote import RemoteWorkerSpec


def _worker() -> RemoteWorkerSpec:
    return RemoteWorkerSpec(
        worker_id="windows_wsl",
        host="wslbox",
        username="ysl",
        remote_project_dir="/mnt/d/agent",
        remote_workspace_dir="/mnt/d/agent/workspace",
        python_executable="/mnt/d/agent/run_agent_python.sh",
        backend_tags=[],
        capabilities={},
    )


def test_runner_returns_structured_error_on_nonzero(tmp_path):
    def fake_run(cmd, capture_output, text, timeout):
        class Completed:
            returncode = 2
            stdout = ""
            stderr = "DeepLens unavailable"

        return Completed()

    runner = SSHRemoteRunner(worker=_worker(), workspace_root=tmp_path / "workspace", process_runner=fake_run)
    result = runner.run_command(
        ["/mnt/d/agent/run_agent_python.sh", "-m", "optiresearch.cli", "check-deeplens"],
        cwd="/mnt/d/agent",
        timeout=10,
        job_id="job_err",
    )

    assert result["status"] == "failed"
    assert result["error_code"] == "REMOTE_COMMAND_FAILED"
    assert result["returncode"] == 2


def test_runner_returns_structured_error_on_timeout(tmp_path):
    def fake_run(cmd, capture_output, text, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    runner = SSHRemoteRunner(worker=_worker(), workspace_root=tmp_path / "workspace", process_runner=fake_run)
    result = runner.run_command(
        ["/mnt/d/agent/run_agent_python.sh", "-m", "optiresearch.cli", "check-deeplens"],
        cwd="/mnt/d/agent",
        timeout=1,
        job_id="job_timeout",
    )

    assert result["status"] == "failed"
    assert result["error_code"] == "REMOTE_COMMAND_TIMEOUT"
