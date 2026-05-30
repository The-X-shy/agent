"""Tests for alias/compatibility mapping in remote contracts."""
from __future__ import annotations

from optiresearch.system.remote_command_inventory import get_canonical_command_name


def test_component_probe_maps_to_existing_allowlist():
    canonical = get_canonical_command_name("run-remote-deeplens-component-probe")
    assert canonical == "run-deeplens-component-probe"


def test_curriculum_probe_maps_to_existing_allowlist():
    canonical = get_canonical_command_name("run-remote-deeplens-curriculum-probe")
    assert canonical == "run-deeplens-curriculum-probe"


def test_regularized_probe_maps_to_existing_allowlist():
    canonical = get_canonical_command_name("run-remote-deeplens-regularized-probe")
    assert canonical == "run-deeplens-regularized-probe"


def test_all_non_gap_contracts_have_valid_allowlist_mapping():
    from optiresearch.remote.command_allowlist import ALLOWED_CLI_COMMANDS
    from optiresearch.system.remote_command_inventory import CONTRACT_TO_ALLOWLIST_COMMAND, KNOWN_GAP_CONTRACT_IDS
    from tests.test_remote_execution_contracts_core_commands import get_all_remote_contracts

    contracts = get_all_remote_contracts()
    for cid, c in contracts.items():
        if cid in KNOWN_GAP_CONTRACT_IDS:
            continue
        canonical = get_canonical_command_name(c.command_name)
        assert canonical is not None, f"Contract {cid} has no canonical mapping"
        assert canonical in ALLOWED_CLI_COMMANDS, f"Contract {cid} canonical '{canonical}' not in allowlist"
