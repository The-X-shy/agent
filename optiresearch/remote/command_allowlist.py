"""Allowlist for remote worker commands.

Remote job commands are built as argument lists by OptiResearch. This module
rejects shell fragments and only accepts known CLI subcommands with known
options.

``--remote-job-id`` is an internal remote execution parameter appended by the
remote runner. It is not a general user extension point, and values must match
OptiResearch's generated ``remote_job_<16 hex chars>`` format.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class CommandValidationError(ValueError):
    """Raised when a remote command is not approved."""


ALLOWED_CLI_COMMANDS: dict[str, set[str]] = {
    "check-deeplens": set(),
    "probe-deeplens-source": {"--remote-job-id"},
    "inspect-deeplens-source": {"--remote-job-id"},
    "run-deeplens-source-smoke": {"--remote-job-id"},
    "run-hsi-reconstruction": {
        "--objective",
        "--backend",
        "--encoder",
        "--workspace-id",
        "--realization",
        "--forward-mode",
        "--reconstructor",
        "--dataset",
        "--dataset-path",
        "--dataset-pattern",
        "--tiny-cnn-epochs",
        "--tiny-cnn-hidden",
        "--device",
        "--use-optical-feature-maps",
        "--remote-job-id",
    },
    "run-hsi-matrix": {
        "--datasets",
        "--backends",
        "--encoders",
        "--reconstructors",
        "--forward-modes",
        "--objective",
        "--workspace-id",
        "--dataset-path",
        "--tiny-cnn-epochs",
        "--tiny-cnn-hidden",
        "--device",
        "--use-optical-feature-maps",
        "--remote-job-id",
    },
    "run-codesign-loop": {
        "--objective",
        "--llm-provider",
        "--max-iterations",
        "--backend",
        "--encoder",
        "--reconstructor",
        "--forward-mode",
        "--dataset",
        "--psf-source",
        "--fallback-policy",
        "--strict-deeplens",
        "--remote-job-id",
    },
    "run-autonomous-loop": {
        "--objective",
        "--llm-provider",
        "--max-iterations",
        "--backend",
        "--dataset",
        "--allowed-reconstructors",
        "--allowed-encoders",
        "--execution-mode",
        "--worker-id",
        "--remote-job-id",
    },
    "inspect-deeplens-native-optimization": {
        "--remote-job-id",
    },
    "run-native-optimization-probe": {
        "--lens-class",
        "--objective",
        "--max-steps",
        "--learning-rate",
        "--device",
        "--strict-native",
        "--allow-adapter-proxy",
        "--remote-job-id",
    },
    "run-deeplens-surface-optimization-probe": {
        "--surface",
        "--objective",
        "--max-steps",
        "--learning-rate",
        "--device",
        "--remote-job-id",
    },
    "run-deeplens-lensfile-optimization-probe": {
        "--lens-class",
        "--max-files",
        "--max-steps",
        "--learning-rate",
        "--device",
        "--remote-job-id",
    },
    "run-native-hsi-codesign": {
        "--optical-component",
        "--objective",
        "--max-steps",
        "--learning-rate",
        "--device",
        "--bands",
        "--image-size",
        "--psf-size",
        "--remote-job-id",
    },
    "run-deeplens-waveoptics-probe": {
        "--candidate", "--objective", "--psf-size", "--max-steps",
        "--learning-rate", "--device", "--remote-job-id",
    },
    "run-stable-native-lens-hsi-codesign": {
        "--candidate", "--reconstructor", "--dataset", "--max-steps",
        "--optical-lr", "--recon-lr", "--optical-grad-clip",
        "--rollback-on-loss-increase", "--device", "--remote-job-id",
    },
    "run-stable-native-lens-hsi-ablation": {
        "--candidate", "--reconstructor", "--dataset", "--device",
        "--remote-job-id",
    },
    "run-native-waveoptics-hsi-codesign": {
        "--candidate", "--reconstructor", "--max-steps",
        "--optical-lr", "--recon-lr", "--device",
        "--bands", "--image-size", "--psf-size",
        "--dataset",
        "--remote-job-id",
    },
    "run-native-hsi-reconstruction-codesign": {
        "--optical-component",
        "--reconstructor",
        "--dataset",
        "--max-steps",
        "--optical-lr",
        "--recon-lr",
        "--device",
        "--bands",
        "--image-size",
        "--psf-size",
        "--remote-job-id",
    },
}

FLAG_OPTIONS = {"--strict-deeplens", "--use-optical-feature-maps", "--strict-native", "--allow-adapter-proxy", "--rollback-on-loss-increase"}
DENIED_EXECUTABLES = {"sudo", "rm", "curl", "chmod"}
SHELL_META_TOKENS = {";", "&&", "||", "|", ">", "<"}
SHELL_META_CHARS = {";", "|", ">", "<", "`"}
REMOTE_JOB_ID_RE = re.compile(r"^remote_job_[a-f0-9]{16}$")
REMOTE_JOB_ID_ALLOWED_COMMANDS = {
    command
    for command, options in ALLOWED_CLI_COMMANDS.items()
    if "--remote-job-id" in options
}


def validate_remote_command(command: list[str]) -> dict[str, Any]:
    """Validate and describe an allowlisted remote command."""

    if not isinstance(command, list) or not command or not all(isinstance(arg, str) for arg in command):
        raise CommandValidationError("remote command must be a non-empty list[str]")

    _reject_shell_fragments(command)
    executable_name = Path(command[0]).name
    if executable_name in DENIED_EXECUTABLES:
        raise CommandValidationError(f"denied executable: {executable_name}")
    if len(command) >= 2 and command[1] == "-c" and executable_name.startswith("python"):
        raise CommandValidationError("python -c is not allowlisted")
    if "-c" in command and executable_name.startswith("python"):
        raise CommandValidationError("python -c is not allowlisted")

    if len(command) < 4 or command[1:3] != ["-m", "optiresearch.cli"]:
        raise CommandValidationError("remote command must run python -m optiresearch.cli <command>")

    cli_command = command[3]
    if cli_command not in ALLOWED_CLI_COMMANDS:
        raise CommandValidationError(f"CLI command is not allowlisted: {cli_command}")

    allowed_options = ALLOWED_CLI_COMMANDS[cli_command]
    _validate_options(command[4:], allowed_options, cli_command)
    return {
        "allowed": True,
        "executable": command[0],
        "cli_command": cli_command,
        "options": command[4:],
    }


def _reject_shell_fragments(command: list[str]) -> None:
    for arg in command:
        if arg in SHELL_META_TOKENS:
            raise CommandValidationError(f"shell metacharacter is not allowed: {arg}")
        if "$(" in arg or ")" in arg and "$(" in " ".join(command):
            raise CommandValidationError("command substitution is not allowed")
        if any(char in arg for char in SHELL_META_CHARS):
            raise CommandValidationError(f"shell metacharacter is not allowed in argument: {arg}")
        if arg == "777":
            previous = command[command.index(arg) - 1] if command.index(arg) > 0 else ""
            if previous == "chmod":
                raise CommandValidationError("chmod 777 is not allowed")


def _validate_options(args: list[str], allowed_options: set[str], cli_command: str) -> None:
    i = 0
    while i < len(args):
        item = args[i]
        if not item.startswith("--"):
            raise CommandValidationError(f"user-supplied command fragment is not allowed: {item}")
        if item not in allowed_options:
            raise CommandValidationError(f"option is not allowlisted: {item}")
        if item in FLAG_OPTIONS:
            i += 1
            continue
        if i + 1 >= len(args):
            raise CommandValidationError(f"missing value for option: {item}")
        value = args[i + 1]
        if value.startswith("--"):
            raise CommandValidationError(f"missing value for option: {item}")
        if item == "--remote-job-id":
            _validate_remote_job_id(cli_command, value)
        i += 2


def _validate_remote_job_id(cli_command: str, value: str) -> None:
    """Validate the internal remote execution id appended by the runner."""

    if cli_command not in REMOTE_JOB_ID_ALLOWED_COMMANDS:
        raise CommandValidationError(f"--remote-job-id is not allowed for command: {cli_command}")
    if not REMOTE_JOB_ID_RE.fullmatch(value):
        raise CommandValidationError("invalid --remote-job-id value")
