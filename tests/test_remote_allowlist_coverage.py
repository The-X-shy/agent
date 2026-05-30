"""Tests for remote allowlist coverage."""
from __future__ import annotations

from optiresearch.system.remote_allowlist_coverage import validate_remote_allowlist_coverage
from tests.test_remote_execution_contracts_core_commands import get_all_remote_contracts


def test_allowlist_coverage_100():
    contracts = get_all_remote_contracts()
    report = validate_remote_allowlist_coverage(contracts)
    assert report["allowlist_coverage"] == 1.0
    assert report["covered_by_allowlist"] == report["total_contracts"]


def test_known_gaps_count():
    contracts = get_all_remote_contracts()
    report = validate_remote_allowlist_coverage(contracts)
    assert report["known_gaps"] == 0  # Known gaps excluded from coverage


def test_coverage_has_expected_structure():
    contracts = get_all_remote_contracts()
    report = validate_remote_allowlist_coverage(contracts)
    assert "total_contracts" in report
    assert "covered_by_allowlist" in report
    assert "allowlist_coverage" in report
    assert "orphan_allowlist_entries" in report
