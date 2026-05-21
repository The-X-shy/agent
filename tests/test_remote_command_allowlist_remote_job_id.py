"""Tests for internal remote job id allowlist validation."""

from __future__ import annotations

import pytest

from optiresearch.remote.command_allowlist import CommandValidationError, validate_remote_command


def _command(cli_command: str, remote_job_id: str) -> list[str]:
    return [
        "/mnt/d/agent/run_agent_python.sh",
        "-m",
        "optiresearch.cli",
        cli_command,
        "--remote-job-id",
        remote_job_id,
    ]


def test_remote_job_id_accepts_current_internal_format():
    command = _command("run-deeplens-source-smoke", "remote_job_1436c05c2c4d6359")

    assert validate_remote_command(command)["allowed"] is True


@pytest.mark.parametrize(
    "remote_job_id",
    [
        "../../tmp",
        "abc",
        "remote_job_$(rm -rf /)",
        "remote_job_1436c05c2c4d6359;rm -rf /",
    ],
)
def test_remote_job_id_rejects_unsafe_values(remote_job_id):
    with pytest.raises(CommandValidationError):
        validate_remote_command(_command("run-deeplens-source-smoke", remote_job_id))


def test_remote_job_id_rejects_command_without_remote_job_id_allowance():
    command = _command("check-deeplens", "remote_job_1436c05c2c4d6359")

    with pytest.raises(CommandValidationError):
        validate_remote_command(command)
