"""Tests that remote execution contracts block arbitrary args."""
from __future__ import annotations

from optiresearch.schemas.remote_execution_contract import RemoteExecutionContract
from optiresearch.system.remote_execution_contract_validator import validate_remote_execution_contracts


def test_contract_with_shell_metachar_in_allowed_args():
    bad = RemoteExecutionContract(
        remote_contract_id="rec_bad",
        command_name="test-cmd",
        allowed_args=["--safe", "; rm -rf /"],
    )
    report = validate_remote_execution_contracts({"rec_bad": bad})
    assert len(report["unsafe_args_detected"]) > 0


def test_contract_with_overlapping_allowed_forbidden():
    bad = RemoteExecutionContract(
        remote_contract_id="rec_overlap",
        command_name="test-cmd",
        allowed_args=["--arg1", "--arg2"],
        forbidden_args=["--arg1"],
    )
    report = validate_remote_execution_contracts({"rec_overlap": bad})
    assert report["valid_contracts"] == 0


def test_contract_with_invalid_timeout():
    bad = RemoteExecutionContract(
        remote_contract_id="rec_timeout",
        command_name="test-cmd",
        timeout_sec=-1,
    )
    report = validate_remote_execution_contracts({"rec_timeout": bad})
    # Should have at least one issue about timeout
    assert report["total_issues"] > 0
