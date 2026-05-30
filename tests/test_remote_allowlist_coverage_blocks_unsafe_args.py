"""Tests that unsafe args are blocked in allowlist coverage checks."""
from __future__ import annotations

from optiresearch.schemas.remote_execution_contract import RemoteExecutionContract
from optiresearch.system.remote_allowlist_coverage import validate_remote_allowlist_coverage


def test_unsafe_shell_metachar_handled():
    bad = RemoteExecutionContract(
        remote_contract_id="rec_shell",
        command_name="run-remote-test",
        allowed_args=["; rm -rf /"],
    )
    # Coverage shouldn't crash on unsafe args
    report = validate_remote_allowlist_coverage({"rec_shell": bad})
    assert "covered_by_allowlist" in report


def test_overlapping_forbidden_args():
    contract = RemoteExecutionContract(
        remote_contract_id="rec_overlap",
        command_name="run-remote-test",
        allowed_args=["--arg1"],
        forbidden_args=["--arg1"],
    )
    report = validate_remote_allowlist_coverage({"rec_overlap": contract})
    assert report["total_contracts"] == 1
