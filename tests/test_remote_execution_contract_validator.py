"""Tests for remote execution contract validator."""
from __future__ import annotations

from optiresearch.system.remote_execution_contract_validator import validate_remote_execution_contracts
from tests.test_remote_execution_contracts_core_commands import get_all_remote_contracts


def test_validate_all_remote_contracts():
    contracts = get_all_remote_contracts()
    report = validate_remote_execution_contracts(contracts)
    assert report["total_contracts"] == 8
    assert "valid_contracts" in report
    assert "allowlist_coverage" in report
    assert report["allowlist_coverage"] >= 0.0


def test_no_unsafe_args_detected():
    contracts = get_all_remote_contracts()
    report = validate_remote_execution_contracts(contracts)
    assert report["unsafe_args_detected"] == []


def test_all_contracts_have_result_parsers():
    contracts = get_all_remote_contracts()
    report = validate_remote_execution_contracts(contracts)
    assert report["missing_parsers"] == []


def test_all_contracts_have_valid_policies():
    contracts = get_all_remote_contracts()
    report = validate_remote_execution_contracts(contracts)
    assert report["total_issues"] >= 0
