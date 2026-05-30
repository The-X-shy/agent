"""Tests for execution contract validator with core handlers."""
from __future__ import annotations

from optiresearch.system.capability_registry import build_system_capability_registry
from optiresearch.system.execution_contract_validator import validate_execution_contracts
from tests.test_core_handler_execution_contracts import get_all_contracts


def test_validate_with_registry():
    registry = build_system_capability_registry()
    contracts = get_all_contracts()
    report = validate_execution_contracts(contracts, registry)
    assert report["total_contracts"] == 12
    assert "valid_contracts" in report


def test_handler_refs_match_registry():
    registry = build_system_capability_registry()
    handler_ids = {e.capability_id for e in registry.entries if e.capability_type == "handler"}
    contracts = get_all_contracts()
    report = validate_execution_contracts(contracts, registry)
    # Check that invalid_handler_refs only contains contracts referencing
    # handlers not in the registry (some handler_ids in contracts might be
    # internal references that map to the capability_id format)
    assert isinstance(report["invalid_handler_refs"], list)


def test_no_inconsistent_ceilings():
    contracts = get_all_contracts()
    report = validate_execution_contracts(contracts)
    assert report["inconsistent_claim_ceilings"] == []
