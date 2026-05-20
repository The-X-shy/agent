import pytest

from optiresearch.remote.command_allowlist import CommandValidationError, validate_remote_command


def test_allowlist_accepts_deeplens_source_smoke_wrapper_command():
    command = [
        "/mnt/d/agent/run_agent_python.sh",
        "-m",
        "optiresearch.cli",
        "run-deeplens-source-smoke",
        "--remote-job-id",
        "remote_job_123",
    ]

    assert validate_remote_command(command)["allowed"] is True


@pytest.mark.parametrize(
    "command",
    [
        ["sudo", "apt", "install", "x"],
        ["rm", "-rf", "/mnt/d/agent"],
        ["python", "-c", "print('bad')"],
        ["python", "-m", "optiresearch.cli", "run-deeplens-source-smoke", ";", "whoami"],
        ["python", "-m", "optiresearch.cli", "run-deeplens-source-smoke", "&&", "whoami"],
        ["python", "-m", "optiresearch.cli", "run-deeplens-source-smoke", "$(whoami)"],
        ["python", "-m", "optiresearch.cli", "run-deeplens-source-smoke", "--unknown", "x"],
    ],
)
def test_allowlist_rejects_dangerous_or_unknown_commands(command):
    with pytest.raises(CommandValidationError):
        validate_remote_command(command)


def test_allowlist_accepts_strict_codesign_controlled_args():
    command = [
        "/mnt/d/agent/run_agent_python.sh",
        "-m",
        "optiresearch.cli",
        "run-codesign-loop",
        "--objective",
        "Run strict DeepLens-backed co-design on WSL D drive worker",
        "--psf-source",
        "deeplens_parameterized",
        "--backend",
        "deeplens",
        "--fallback-policy",
        "fail",
        "--max-iterations",
        "2",
        "--strict-deeplens",
        "--remote-job-id",
        "remote_codesign_1",
    ]

    assert validate_remote_command(command)["cli_command"] == "run-codesign-loop"
