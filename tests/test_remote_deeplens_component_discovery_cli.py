"""Tests for remote component discovery CLI integration."""

import json
import subprocess
import sys


def _run_cli(args: list[str]):
    return subprocess.run(
        [sys.executable, "-m", "optiresearch.cli", *args],
        capture_output=True, text=True, timeout=30,
    )


class TestRemoteComponentDiscoveryCLI:
    def test_remote_discover_help(self):
        result = _run_cli(["run-remote-discover-deeplens-components", "--help"])
        assert result.returncode == 0
        assert "--worker-id" in result.stdout

    def test_remote_component_probe_help(self):
        result = _run_cli(["run-remote-deeplens-component-probe", "--help"])
        assert result.returncode == 0
        assert "--component" in result.stdout
        assert "--worker-id" in result.stdout

    def test_remote_component_probe_requires_worker_id(self):
        """Without --worker-id, remote probe should fail."""
        result = _run_cli([
            "run-remote-deeplens-component-probe",
            "--component", "fresnel",
        ])
        assert result.returncode != 0

    def test_remote_discover_requires_worker_id(self):
        result = _run_cli([
            "run-remote-discover-deeplens-components",
        ])
        assert result.returncode != 0

    def test_component_probe_choices_validated(self):
        """--component must be one of fresnel/binary2phase/diffractive."""
        result = _run_cli([
            "run-deeplens-component-probe",
            "--component", "invalid_component",
        ])
        assert result.returncode != 0
