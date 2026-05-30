"""Tests for artifact contract validator with missing artifacts."""
from __future__ import annotations

from tests.test_core_artifact_contracts import get_artifact_contract
from optiresearch.system.artifact_contract_validator import validate_artifact_contract_for_run


def test_missing_policy_needs_followup(tmp_path):
    contract = get_artifact_contract("ac_native_geolens_stability")
    result = validate_artifact_contract_for_run(tmp_path, contract)
    assert result["status"] == "needs_followup"
    assert result["missing_artifact_policy"] == "needs_followup"


def test_missing_policy_structured_warning(tmp_path):
    contract = get_artifact_contract("ac_diagnostic")
    result = validate_artifact_contract_for_run(tmp_path, contract)
    assert result["status"] == "structured_warning"
    assert result["missing_artifact_policy"] == "structured_warning"


def test_missing_policy_partial_evidence(tmp_path):
    contract = get_artifact_contract("ac_agent_plan")
    result = validate_artifact_contract_for_run(tmp_path, contract)
    assert result["status"] == "partial_evidence"
