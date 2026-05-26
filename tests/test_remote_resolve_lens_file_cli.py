"""Tests for remote resolve-lens-file CLI command structure."""

import pytest


class TestResolveLensFileCLI:
    def test_cli_command_registered(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "optiresearch.cli", "--help"],
            capture_output=True, text=True,
        )
        assert "resolve-lens-file" in result.stdout

    def test_remote_resolve_lens_file_registered(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "optiresearch.cli", "--help"],
            capture_output=True, text=True,
        )
        assert "run-remote-resolve-lens-file" in result.stdout

    def test_remote_diagnostic_commands_registered(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "optiresearch.cli", "--help"],
            capture_output=True, text=True,
        )
        for cmd in [
            "run-remote-deeplens-trainable-parameter-inspection",
            "run-remote-deeplens-autograd-audit",
        ]:
            assert cmd in result.stdout, f"{cmd} not in CLI help"

    def test_local_resolve_produces_valid_json(self):
        import subprocess
        import sys
        import json
        result = subprocess.run(
            [sys.executable, "-m", "optiresearch.cli", "resolve-lens-file",
             "--lens-file", "auto:cooke",
             "--backend-id", "deeplens_geolens_geometric"],
            capture_output=True, text=True,
        )
        parsed = json.loads(result.stdout)
        assert "requested_lens_file" in parsed
        assert "resolved_path" in parsed
        assert "source" in parsed
        assert "checked_paths" in parsed
