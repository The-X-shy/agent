"""Tests for core remote artifact contracts."""
from __future__ import annotations

from tests.test_core_artifact_contracts import get_artifact_contract


def test_diagnostic_contract_has_valid_handler():
    c = get_artifact_contract("ac_diagnostic")
    assert c.handler_id == "deeplens_trainable_parameter_inspection"


def test_component_probe_contract_has_valid_handler():
    c = get_artifact_contract("ac_component_probe")
    assert c.handler_id == "deeplens_component_first_probe"


def test_remote_job_contract_has_valid_handler():
    c = get_artifact_contract("ac_remote_job")
    assert c.handler_id == "remote_native_geolens_validation"


def test_agent_plan_contract_has_valid_handler():
    c = get_artifact_contract("ac_agent_plan")
    assert c.handler_id == "report_negative_result_doc"


def test_all_contracts_use_metrics_summary():
    contracts = ["ac_diagnostic", "ac_component_probe", "ac_native_geolens_stability",
                 "ac_native_geolens_benchmark", "ac_remote_job", "ac_agent_plan"]
    for cid in contracts:
        c = get_artifact_contract(cid)
        assert "metrics_summary.json" in c.required_artifacts, \
            f"Contract {cid} should require metrics_summary.json"
