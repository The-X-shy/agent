"""Allowlist tests for remote component surrogate HSI co-design."""

import pytest

from optiresearch.remote.command_allowlist import (
    ALLOWED_CLI_COMMANDS,
    CommandValidationError,
    validate_remote_command,
)


def test_component_surrogate_hsi_command_is_allowlisted():
    assert "run-component-surrogate-hsi-codesign" in ALLOWED_CLI_COMMANDS


@pytest.mark.parametrize("option", ["--component", "--dataset", "--steps", "--device", "--remote-job-id"])
def test_component_surrogate_hsi_allowed_options(option):
    assert option in ALLOWED_CLI_COMMANDS["run-component-surrogate-hsi-codesign"]


def test_component_surrogate_hsi_valid_command_passes_validation():
    command = [
        "python",
        "-m",
        "optiresearch.cli",
        "run-component-surrogate-hsi-codesign",
        "--component",
        "fresnel",
        "--dataset",
        "synthetic",
        "--steps",
        "3",
        "--device",
        "cpu",
        "--remote-job-id",
        "remote_job_0000000000000001",
    ]

    result = validate_remote_command(command)

    assert result["allowed"] is True
    assert result["cli_command"] == "run-component-surrogate-hsi-codesign"


def test_component_surrogate_hsi_unknown_option_is_rejected():
    command = [
        "python",
        "-m",
        "optiresearch.cli",
        "run-component-surrogate-hsi-codesign",
        "--component",
        "fresnel",
        "--shell",
        "echo bad",
    ]
    with pytest.raises(CommandValidationError):
        validate_remote_command(command)


def test_component_surrogate_hsi_shell_metacharacter_is_rejected():
    command = [
        "python",
        "-m",
        "optiresearch.cli",
        "run-component-surrogate-hsi-codesign",
        "--component",
        "fresnel; rm -rf /",
    ]
    with pytest.raises(CommandValidationError):
        validate_remote_command(command)
