"""Tests for artifact contract validator."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from optiresearch.system.artifact_contract_validator import validate_artifact_contract_for_run
from tests.test_core_artifact_contracts import get_all_artifact_contracts


def test_validate_with_all_required_artifacts(tmp_path):
    contract = get_all_artifact_contracts()["ac_diagnostic"]
    for artifact in contract.required_artifacts:
        (tmp_path / artifact).write_text("{}", encoding="utf-8")
    result = validate_artifact_contract_for_run(tmp_path, contract)
    assert result["status"] == "passed"
    assert result["required_missing"] == 0


def test_validate_detects_missing_artifacts(tmp_path):
    contract = get_all_artifact_contracts()["ac_diagnostic"]
    # Don't create any artifacts
    result = validate_artifact_contract_for_run(tmp_path, contract)
    assert result["status"] != "passed"
    assert result["required_missing"] > 0


def test_validate_detects_partial_artifacts(tmp_path):
    contract = get_all_artifact_contracts()["ac_native_geolens_stability"]
    (tmp_path / "result.json").write_text("{}", encoding="utf-8")
    result = validate_artifact_contract_for_run(tmp_path, contract)
    assert result["status"] != "passed"
    assert "result.json" in result["present"]
    assert "report.md" in result["missing"]


def test_validate_all_core_contracts_defined():
    contracts = get_all_artifact_contracts()
    assert len(contracts) == 9
    for cid, c in contracts.items():
        assert c.contract_id == cid
        assert len(c.required_artifacts) > 0
