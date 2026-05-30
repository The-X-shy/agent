"""Validate remote execution contracts with canonical mapping (Phase 69)."""
from __future__ import annotations

from typing import Any

from optiresearch.schemas.remote_execution_contract import RemoteExecutionContract


def validate_remote_execution_contracts(
    contracts: dict[str, RemoteExecutionContract],
    inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check remote execution contracts against allowlist and safety rules.

    Uses canonical mapping from remote_command_inventory to translate
    orchestrator-side command names (run-remote-X) to worker-side
    allowlist entries (run-X).
    """
    from optiresearch.remote.command_allowlist import ALLOWED_CLI_COMMANDS
    from optiresearch.system.remote_command_inventory import (
        get_canonical_command_name, KNOWN_GAP_CONTRACT_IDS,
    )

    results: list[dict[str, Any]] = []
    missing_allowlist: list[str] = []
    known_gaps: list[str] = []
    unsafe_args: list[str] = []
    missing_parsers: list[str] = []
    orphan_handler_ids: list[str] = []

    # Resolve handler IDs from registry if available
    valid_handler_ids: set[str] = set()
    try:
        from optiresearch.skills.handler_capability_registry import get_handler_capability_registry
        reg = get_handler_capability_registry()
        valid_handler_ids = {h.handler_id for h in reg.list_enabled()}
    except Exception:
        pass

    with_allowlist = 0
    for cid, c in contracts.items():
        entry_result = {
            "remote_contract_id": cid,
            "command_name": c.command_name,
            "valid": True,
            "issues": [],
        }

        # Known gaps: skip allowlist and existence checks
        is_gap = c.is_known_gap or cid in KNOWN_GAP_CONTRACT_IDS
        if is_gap:
            known_gaps.append(cid)
            entry_result["is_known_gap"] = True
            results.append(entry_result)
            continue

        # Resolve canonical (allowlist-side) command name
        canonical = get_canonical_command_name(c.command_name)
        entry_result["canonical_command_name"] = canonical

        # Rule 1: canonical command must be in allowlist
        if canonical is None:
            entry_result["valid"] = False
            entry_result["issues"].append(f"command '{c.command_name}' has no canonical mapping")
            missing_allowlist.append(cid)
        elif canonical not in ALLOWED_CLI_COMMANDS:
            entry_result["valid"] = False
            entry_result["issues"].append(f"canonical command '{canonical}' not in allowlist")
            missing_allowlist.append(cid)
        else:
            with_allowlist += 1
            # Rule 2: allowed_args subset of allowlist
            allowlisted_flags = ALLOWED_CLI_COMMANDS[canonical]
            for arg in c.allowed_args:
                if arg not in allowlisted_flags:
                    entry_result["issues"].append(f"arg '{arg}' not in allowlist for '{canonical}'")

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

        # Rule 5: result parser specified
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

        # Rule 8: handler_id must be valid in registry (if registry is available)
        if valid_handler_ids and c.handler_id and c.handler_id not in valid_handler_ids:
            entry_result["issues"].append(f"handler_id '{c.handler_id}' not in handler capability registry")
            orphan_handler_ids.append(cid)

        results.append(entry_result)

    active_contracts = len(contracts) - len(known_gaps)
    valid = [r for r in results if r["valid"]]
    total_issues = sum(len(r["issues"]) for r in results)

    return {
        "validation_status": "passed" if not missing_allowlist and not unsafe_args and total_issues == 0 else "issues_found",
        "total_contracts": len(contracts),
        "active_contracts": active_contracts,
        "known_gaps": known_gaps,
        "valid_contracts": len(valid),
        "invalid_contracts": len(results) - len(valid) - len(known_gaps),
        "total_issues": total_issues,
        "missing_allowlist": missing_allowlist,
        "unsafe_args_detected": unsafe_args,
        "missing_parsers": missing_parsers,
        "orphan_handler_ids": orphan_handler_ids,
        "allowlist_coverage": with_allowlist / max(active_contracts, 1),
        "result_parser_coverage": (active_contracts - len(missing_parsers)) / max(active_contracts, 1),
        "results": results,
    }
