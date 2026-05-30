"""Tests for ExecutionContract schema."""
from __future__ import annotations

import pytest

from optiresearch.schemas.execution_contract import ExecutionContract


def test_execution_contract_minimal():
    contract = ExecutionContract(
        contract_id="ec_deeplens_native_geolens",
        handler_id="deeplens_native_geolens_hsi_codesign",
    )
    assert contract.contract_id == "ec_deeplens_native_geolens"
    assert contract.handler_id == "deeplens_native_geolens_hsi_codesign"
    assert contract.execution_modes == []


def test_execution_contract_full():
    contract = ExecutionContract(
        contract_id="ec_native_geolens_hsi",
        handler_id="deeplens_native_geolens_hsi_codesign",
        skill_id="deeplens_native_geolens_hsi_codesign",
        design_ids=["geolens_curriculum_probe"],
        backend_ids=["deeplens_geolens_geometric"],
        execution_modes=["local", "remote_opt_in"],
        required_inputs=["lens_file", "dataset_spec"],
        required_outputs=["result.json", "metrics.json"],
        required_metrics=["mse", "psnr", "sam"],
        status_values=["succeeded", "unsupported", "failed"],
        evidence_level_mapping={"local": "native_lens_simulation", "remote_opt_in": "native_lens_simulation"},
        claim_ceiling_mapping={"local": "native_lens_simulation", "remote_opt_in": "native_lens_simulation"},
        failure_modes=["gradient_instability", "rollback_triggered"],
        retry_policy={"max_retries": 3, "backoff_sec": 60},
        timeout_policy={"local": 1200, "remote_opt_in": 3600},
        artifact_contract_id="ac_native_geolens",
        report_contract_id="rc_native_geolens",
    )
    assert contract.required_metrics == ["mse", "psnr", "sam"]
    assert contract.retry_policy["max_retries"] == 3


def test_execution_contract_rejects_extra_fields():
    with pytest.raises(ValueError):
        ExecutionContract(
            contract_id="test",
            handler_id="test",
            extra_field="no",
        )
