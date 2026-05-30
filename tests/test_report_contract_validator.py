"""Tests for report contract validator."""
from __future__ import annotations

from pathlib import Path

from optiresearch.system.report_contract_validator import validate_report_contract
from tests.test_core_report_contracts import get_all_report_contracts


def test_validate_report_missing_file():
    contract = get_all_report_contracts()["rc_agent_plan"]
    result = validate_report_contract("/nonexistent/path/report.md", contract)
    assert result["status"] == "report_missing"


def test_validate_report_with_all_sections(tmp_path):
    contract = get_all_report_contracts()["rc_native_geolens_benchmark"]
    content = "\n".join(
        f"## {s}" for s in contract.required_sections
    ) + "\n## Blocked Claims\n## Evidence Level\n## Safe Wording\n"
    report_path = tmp_path / "report.md"
    report_path.write_text(content, encoding="utf-8")
    result = validate_report_contract(report_path, contract)
    assert result["sections_missing"] == 0


def test_validate_report_with_missing_sections(tmp_path):
    contract = get_all_report_contracts()["rc_native_geolens_benchmark"]
    report_path = tmp_path / "report.md"
    report_path.write_text("## Benchmark Summary\n", encoding="utf-8")
    result = validate_report_contract(report_path, contract)
    assert result["sections_missing"] > 0


def test_all_core_contracts_defined():
    contracts = get_all_report_contracts()
    assert len(contracts) == 8
    for cid, c in contracts.items():
        assert c.report_contract_id == cid
        assert len(c.required_sections) > 0
