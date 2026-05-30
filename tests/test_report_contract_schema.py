"""Tests for ReportContract schema."""
from __future__ import annotations

import pytest

from optiresearch.schemas.report_contract import ReportContract


def test_report_contract_minimal():
    contract = ReportContract(report_contract_id="rc_system_capability")
    assert contract.report_contract_id == "rc_system_capability"
    assert contract.safe_wording_required is False


def test_report_contract_full():
    contract = ReportContract(
        report_contract_id="rc_native_geolens_benchmark",
        report_type="native_geolens_benchmark_report",
        exporter_cli="export-native-geolens-benchmark-report",
        required_sections=[
            "Benchmark Summary",
            "Completed Configurations",
            "Improvement Rates (Completed Only)",
            "Full-Grid Improvement Rates",
            "Claim Boundary",
            "Blocked Claims",
            "Evidence Level",
            "Safe Wording",
        ],
        optional_sections=["PSF Statistics", "Stability Trace"],
        required_tables=["improvement_rates", "metric_statistics"],
        required_fields=["completed_count", "unsupported_count", "failed_count"],
        linked_artifacts=["benchmark_summary.json"],
        linked_claims=["reproducible_synthetic_stability"],
        safe_wording_required=True,
        blocked_claims_section_required=True,
        evidence_level_section_required=True,
    )
    assert len(contract.required_sections) == 8
    assert contract.blocked_claims_section_required is True
    assert contract.evidence_level_section_required is True


def test_report_contract_rejects_extra():
    with pytest.raises(ValueError):
        ReportContract(report_contract_id="test", extra="nope")
