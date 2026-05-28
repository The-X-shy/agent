"""Tests for allowlisted component probe CLI commands."""

import pytest

from optiresearch.remote.command_allowlist import (
    ALLOWED_CLI_COMMANDS,
    CommandValidationError,
    validate_remote_command,
)


class TestComponentProbeAllowlist:
    def test_component_probe_command_is_allowlisted(self):
        assert "run-deeplens-component-probe" in ALLOWED_CLI_COMMANDS

    def test_discover_components_command_is_allowlisted(self):
        assert "discover-deeplens-components" in ALLOWED_CLI_COMMANDS

    @pytest.mark.parametrize("option", [
        "--component", "--objective", "--max-steps", "--learning-rate",
        "--device", "--remote-job-id",
    ])
    def test_component_probe_allowed_options(self, option):
        assert option in ALLOWED_CLI_COMMANDS["run-deeplens-component-probe"]

    @pytest.mark.parametrize("option", [
        "--components", "--device", "--remote-job-id",
    ])
    def test_discover_components_allowed_options(self, option):
        assert option in ALLOWED_CLI_COMMANDS["discover-deeplens-components"]

    def test_component_probe_valid_command_passes_validation(self):
        command = [
            "python", "-m", "optiresearch.cli",
            "run-deeplens-component-probe",
            "--component", "fresnel",
            "--device", "cpu",
            "--remote-job-id", "remote_job_0000000000000001",
        ]
        result = validate_remote_command(command)
        assert result["allowed"] is True
        assert result["cli_command"] == "run-deeplens-component-probe"

    def test_discover_components_valid_command_passes_validation(self):
        command = [
            "python", "-m", "optiresearch.cli",
            "discover-deeplens-components",
            "--components", "fresnel,binary2phase",
            "--device", "cpu",
            "--remote-job-id", "remote_job_0000000000000001",
        ]
        result = validate_remote_command(command)
        assert result["allowed"] is True
        assert result["cli_command"] == "discover-deeplens-components"

    def test_unknown_option_is_rejected(self):
        command = [
            "python", "-m", "optiresearch.cli",
            "run-deeplens-component-probe",
            "--component", "fresnel",
            "--unknown-flag", "value",
        ]
        with pytest.raises(CommandValidationError):
            validate_remote_command(command)

    def test_shell_metacharacter_is_rejected(self):
        command = [
            "python", "-m", "optiresearch.cli",
            "run-deeplens-component-probe",
            "--component", "fresnel; rm -rf /",
        ]
        with pytest.raises(CommandValidationError):
            validate_remote_command(command)

    def test_invalid_remote_job_id_is_rejected(self):
        command = [
            "python", "-m", "optiresearch.cli",
            "run-deeplens-component-probe",
            "--component", "fresnel",
            "--remote-job-id", "not_a_valid_id",
        ]
        with pytest.raises(CommandValidationError):
            validate_remote_command(command)
