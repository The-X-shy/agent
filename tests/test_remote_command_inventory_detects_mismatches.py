"""Tests that inventory detects mismatches."""
from __future__ import annotations

from optiresearch.system.remote_command_inventory import build_remote_command_inventory


def test_detects_missing_from_allowlist():
    inv = build_remote_command_inventory()
    missing = inv["missing_from_allowlist"]
    # Known gaps should be in the missing_from_allowlist
    assert "run-remote-native-geolens-benchmark-failure-analysis" in missing
    assert "run-remote-resume-native-geolens-benchmark" in missing


def test_cli_commands_summary():
    inv = build_remote_command_inventory()
    assert isinstance(inv["cli_commands"], list)
    assert any("run-remote-deeplens" in c for c in inv["cli_commands"])


def test_known_gaps_in_inventory():
    inv = build_remote_command_inventory()
    assert "rec_benchmark_failure_analysis" in inv["known_gaps"]
    assert "rec_resume_benchmark" in inv["known_gaps"]
