"""Tests for remote command inventory."""
from __future__ import annotations

from optiresearch.system.remote_command_inventory import build_remote_command_inventory, get_canonical_command_name


def test_build_remote_command_inventory():
    inv = build_remote_command_inventory()
    assert inv["inventory_version"] == "0.1"
    assert inv["total_cli_commands"] > 0
    assert inv["total_allowlist_entries"] > 0
    assert inv["total_remote_jobs_functions"] > 0
    assert inv["total_contracts"] >= 8


def test_canonical_mapping_strips_prefix():
    assert get_canonical_command_name("run-remote-deeplens-trainable-parameter-inspection") == "run-deeplens-trainable-parameter-inspection"
    assert get_canonical_command_name("run-remote-stabilized-native-geolens-hsi") == "run-stabilized-native-geolens-hsi"


def test_known_gaps_return_none():
    assert get_canonical_command_name("run-remote-native-geolens-benchmark-failure-analysis") is None
    assert get_canonical_command_name("run-remote-resume-native-geolens-benchmark") is None


def test_mapped_contracts_have_expected_keys():
    inv = build_remote_command_inventory()
    for mc in inv["mapped_contracts"]:
        assert "contract_command_name" in mc
        assert "mapped_allowlist_name" in mc
        assert "is_known_gap" in mc
