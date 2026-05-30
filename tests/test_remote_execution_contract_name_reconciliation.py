"""Tests for remote contract name reconciliation via canonical mapping."""
from __future__ import annotations

from optiresearch.system.remote_execution_contract_validator import validate_remote_execution_contracts
from tests.test_remote_execution_contracts_core_commands import get_all_remote_contracts


def test_all_contracts_resolve_to_allowlist():
    contracts = get_all_remote_contracts()
    report = validate_remote_execution_contracts(contracts)
    assert report["allowlist_coverage"] == 1.0
    assert report["missing_allowlist"] == []


def test_no_unsafe_args():
    contracts = get_all_remote_contracts()
    report = validate_remote_execution_contracts(contracts)
    assert report["unsafe_args_detected"] == []


def test_all_active_contracts_have_result_parser():
    contracts = get_all_remote_contracts()
    report = validate_remote_execution_contracts(contracts)
    assert report["result_parser_coverage"] == 1.0
