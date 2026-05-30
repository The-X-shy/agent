"""Tests for expanded artifact contract coverage."""
from __future__ import annotations

from optiresearch.system.artifact_contract_validator import validate_artifact_contracts_against_registry
from tests.test_core_artifact_contracts import get_all_artifact_contracts


def test_validate_artifact_contracts_against_registry():
    contracts = get_all_artifact_contracts()
    report = validate_artifact_contracts_against_registry(contracts)
    assert report["artifact_contract_count"] == 9
    assert "invalid_handler_ids" in report
    assert "evidence_role_coverage" in report


def test_contracts_have_evidence_roles():
    contracts = get_all_artifact_contracts()
    report = validate_artifact_contracts_against_registry(contracts)
    roles = report["evidence_role_coverage"]
    assert len(roles) > 0


def test_missing_artifact_policies_defined():
    contracts = get_all_artifact_contracts()
    policies = {c.missing_artifact_policy for c in contracts.values()}
    assert "needs_followup" in policies
    assert "partial_evidence" in policies
    assert "structured_warning" in policies
