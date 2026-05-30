"""Validate remote execution contracts (Phase 68)."""
from __future__ import annotations

from typing import Any

from optiresearch.schemas.remote_execution_contract import RemoteExecutionContract


def validate_remote_execution_contracts(
    contracts: dict[str, RemoteExecutionContract],
) -> dict[str, Any]:
    """Check remote execution contracts against allowlist and safety rules."""
    from optiresearch.remote.command_allowlist import ALLOWED_CLI_COMMANDS

    issues: list[str] = []
    results: list[dict[str, Any]] = []
    missing_allowlist: list[str] = []
    unsafe_args: list[str] = []
    missing_parsers: list[str] = []

    for cid, c in contracts.items():
        entry_result = {
            "remote_contract_id": cid,
            "command_name": c.command_name,
            "valid": True,
            "issues": [],
        }

        # Rule 1: command must be in allowlist
        if c.command_name not in ALLOWED_CLI_COMMANDS:
            entry_result["valid"] = False
            entry_result["issues"].append(f"command '{c.command_name}' not in allowlist")
            missing_allowlist.append(cid)

        # Rule 2: allowed_args subset of allowlist
        else:
            allowlisted_flags = ALLOWED_CLI_COMMANDS[c.command_name]
            for arg in c.allowed_args:
                if arg not in allowlisted_flags:
                    entry_result["issues"].append(f"arg '{arg}' not in allowlist for '{c.command_name}'")

        # Rule 3: forbidden_args must not overlap allowed_args
        overlap = set(c.allowed_args) & set(c.forbidden_args)
        if overlap:
            entry_result["valid"] = False
            entry_result["issues"].append(f"Args both allowed and forbidden: {overlap}")

        # Rule 4: no arbitrary shell execution patterns
        dangerous_args = [a for a in c.allowed_args if ";" in a or "|" in a or "$(" in a]
        if dangerous_args:
            entry_result["valid"] = False
            entry_result["issues"].append(f"Dangerous args detected: {dangerous_args}")
            unsafe_args.append(cid)

        # Rule 5: result parser / failure parser specified
        if not c.result_parser:
            entry_result["issues"].append("No result_parser specified")
            missing_parsers.append(cid)

        # Rule 6: timeout reasonable
        if c.timeout_sec <= 0:
            entry_result["issues"].append(f"Invalid timeout: {c.timeout_sec}")
        if c.timeout_sec > 86400:
            entry_result["issues"].append("Timeout exceeds 24 hours")

        # Rule 7: valid policies
        valid_policies = {"required", "optional", "none"}
        if c.output_dir_policy not in valid_policies:
            entry_result["issues"].append(f"Invalid output_dir_policy: {c.output_dir_policy}")
        if c.artifact_return_policy not in valid_policies:
            entry_result["issues"].append(f"Invalid artifact_return_policy: {c.artifact_return_policy}")

        results.append(entry_result)

    valid = [r for r in results if r["valid"]]
    total_issues = sum(len(r["issues"]) for r in results)

    return {
        "validation_status": "passed" if not missing_allowlist and not unsafe_args and total_issues == 0 else "issues_found",
        "total_contracts": len(contracts),
        "valid_contracts": len(valid),
        "invalid_contracts": len(results) - len(valid),
        "total_issues": total_issues,
        "missing_allowlist": missing_allowlist,
        "unsafe_args_detected": unsafe_args,
        "missing_parsers": missing_parsers,
        "allowlist_coverage": len([c for c in contracts.values() if c.command_name in ALLOWED_CLI_COMMANDS]) / max(len(contracts), 1),
        "results": results,
    }
