"""Tests for execution contract validator."""
from __future__ import annotations

from optiresearch.system.execution_contract_validator import validate_execution_contracts
from tests.test_core_handler_execution_contracts import get_all_contracts


def test_validate_all_core_contracts():
    contracts = get_all_contracts()
    report = validate_execution_contracts(contracts)
    assert report["total_contracts"] == 12
    assert report["valid_contracts"] >= 10  # Most should be valid
    assert "results" in report


def test_validate_detects_empty_execution_modes():
    contracts = get_all_contracts()
    report = validate_execution_contracts(contracts)
    assert report["missing_execution_modes"] == []


def test_validate_detects_no_required_outputs():
    contracts = get_all_contracts()
    report = validate_execution_contracts(contracts)
    assert report["missing_required_outputs"] == []


def test_all_contracts_have_handler_ids():
    contracts = get_all_contracts()
    for cid, c in contracts.items():
        assert c.handler_id, f"Contract {cid} missing handler_id"
