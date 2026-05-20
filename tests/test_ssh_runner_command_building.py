from pathlib import Path

from optiresearch.remote.ssh_runner import SSHRemoteRunner
from optiresearch.schemas.remote import RemoteWorkerSpec


def test_ssh_runner_builds_quoted_remote_command(tmp_path):
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
    runner = SSHRemoteRunner(worker=worker, workspace_root=tmp_path / "workspace")

    ssh_command = runner.build_ssh_command(
        [worker.python_executable, "-m", "optiresearch.cli", "check-deeplens"],
        cwd="/mnt/d/agent",
    )

    assert ssh_command[:3] == ["ssh", "-p", "22"]
    assert "ysl@wslbox" in ssh_command
    remote = ssh_command[-1]
    assert "cd /mnt/d/agent" in remote
    assert "/mnt/d/agent/run_agent_python.sh -m optiresearch.cli check-deeplens" in remote


def test_ssh_runner_logs_stdout_and_stderr(tmp_path):
    worker = RemoteWorkerSpec(
        worker_id="windows_wsl",
        host="wslbox",
        username="ysl",
        remote_project_dir="/mnt/d/agent",
        remote_workspace_dir="/mnt/d/agent/workspace",
        python_executable="/mnt/d/agent/run_agent_python.sh",
        backend_tags=[],
        capabilities={},
    )

    def fake_run(cmd, capture_output, text, timeout):
        class Completed:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return Completed()

    runner = SSHRemoteRunner(worker=worker, workspace_root=tmp_path / "workspace", process_runner=fake_run)
    result = runner.run_command(
        [worker.python_executable, "-m", "optiresearch.cli", "check-deeplens"],
        cwd="/mnt/d/agent",
        timeout=10,
        job_id="job_1",
    )

    assert result["status"] == "succeeded"
    assert Path(result["stdout_path"]).read_text(encoding="utf-8") == "ok"
