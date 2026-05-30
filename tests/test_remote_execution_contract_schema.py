"""Tests for RemoteExecutionContract schema."""
from __future__ import annotations

import pytest

from optiresearch.schemas.remote_execution_contract import RemoteExecutionContract


def test_remote_execution_contract_minimal():
    contract = RemoteExecutionContract(
        remote_contract_id="rec_native_geolens_benchmark",
        command_name="run-remote-native-geolens-stability-benchmark",
    )
    assert contract.remote_contract_id == "rec_native_geolens_benchmark"
    assert contract.allowlist_entry_required is True
    assert contract.timeout_sec == 600


def test_remote_execution_contract_full():
    contract = RemoteExecutionContract(
        remote_contract_id="rec_deeplens_param_inspection",
        command_name="run-remote-deeplens-trainable-parameter-inspection",
        handler_id="deeplens_trainable_parameter_inspection",
        allowed_args=["--lens-file", "--dataset", "--max-steps", "--steps", "--remote-job-id"],
        forbidden_args=["--allow-adapter-proxy"],
        required_worker_capabilities=["deeplens_available", "windows_wsl"],
        required_env_vars=["DEEPLENS_PATH"],
        propagated_env_vars=["OPTIRESEARCH_DB_PATH"],
        timeout_sec=1800,
        output_dir_policy="required",
        artifact_return_policy="required",
        allowlist_entry_required=True,
        workspace_write_policy="restricted",
        remote_job_id_required=True,
        result_parser="result.json",
        failure_parser="error_log.txt",
        retry_policy={"max_retries": 2, "backoff_sec": 120},
    )
    assert contract.required_worker_capabilities == ["deeplens_available", "windows_wsl"]
    assert contract.retry_policy["max_retries"] == 2
    assert "--allow-adapter-proxy" in contract.forbidden_args


def test_remote_execution_contract_rejects_extra():
    with pytest.raises(ValueError):
        RemoteExecutionContract(
            remote_contract_id="test",
            command_name="test",
            bad_field=True,
        )
