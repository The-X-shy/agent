"""Tests for contract coverage dashboard."""
from __future__ import annotations

from optiresearch.system.contract_coverage import generate_contract_coverage


def test_generate_contract_coverage():
    dashboard = generate_contract_coverage()
    assert dashboard["dashboard_version"] == "0.1"
    assert "handler_contract_coverage" in dashboard
    assert "overall_system_readiness_score" in dashboard
    assert 0.0 <= dashboard["overall_system_readiness_score"] <= 1.0


def test_coverage_has_handler_count():
    dashboard = generate_contract_coverage()
    assert dashboard["handler_count"] > 0
    assert dashboard["skill_count"] > 0
    assert dashboard["design_count"] > 0


def test_coverage_has_contract_counts():
    dashboard = generate_contract_coverage()
    assert dashboard["execution_contract_count"] >= 0
    assert dashboard["remote_contract_count"] >= 0
    assert dashboard["artifact_contract_count"] >= 0
    assert dashboard["report_contract_count"] >= 0


def test_test_coverage_proxy():
    dashboard = generate_contract_coverage()
    proxy = dashboard["test_coverage_proxy"]
    assert "coverage_ratio" in proxy
    assert "details" in proxy
    assert 0.0 <= proxy["coverage_ratio"] <= 1.0


def test_doc_coverage_proxy():
    dashboard = generate_contract_coverage()
    proxy = dashboard["doc_coverage_proxy"]
    assert "coverage_ratio" in proxy
    assert "details" in proxy
    assert 0.0 <= proxy["coverage_ratio"] <= 1.0
