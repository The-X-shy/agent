"""Tests for report contract validator benchmark boundaries."""
from __future__ import annotations

from optiresearch.system.report_contract_validator import validate_report_contract
from tests.test_core_report_contracts import get_report_contract


def test_benchmark_contract_requires_completed_only_section(tmp_path):
    contract = get_report_contract("rc_native_geolens_benchmark")
    assert any("completed-only" in s.lower() or "completed only" in s.lower()
               for s in contract.required_sections), \
        "Benchmark contract should require completed-only section"


def test_benchmark_contract_requires_full_grid_section(tmp_path):
    contract = get_report_contract("rc_native_geolens_benchmark")
    assert any("full-grid" in s.lower() or "full grid" in s.lower()
               for s in contract.required_sections), \
        "Benchmark contract should require full-grid section"


def test_benchmark_contract_distinguishes_boundaries(tmp_path):
    contract = get_report_contract("rc_native_geolens_benchmark")
    content = "\n".join(
        f"## {s}" for s in contract.required_sections
    ) + "\n## Blocked Claims\n## Evidence Level\n## Safe Wording\n"
    content += "\nArtifacts: benchmark_summary.json, benchmark_results.csv, report.md\n"
    content += "Linked claims: reproducible_synthetic_stability\n"
    report_path = tmp_path / "benchmark.md"
    report_path.write_text(content, encoding="utf-8")
    result = validate_report_contract(report_path, contract)
    assert result["status"] == "passed"


def test_stability_contract_requires_claim_boundary_section():
    contract = get_report_contract("rc_native_geolens_stability")
    assert contract.blocked_claims_section_required is True
    assert contract.evidence_level_section_required is True
