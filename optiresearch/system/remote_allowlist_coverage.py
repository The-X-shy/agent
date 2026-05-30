"""Remote allowlist coverage validator (Phase 69)."""
from __future__ import annotations

from typing import Any


def validate_remote_allowlist_coverage(
    contracts: dict[str, Any],
    inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Cross-reference remote contracts against allowlist using canonical mapping."""
    from optiresearch.remote.command_allowlist import ALLOWED_CLI_COMMANDS
    from optiresearch.system.remote_command_inventory import (
        get_canonical_command_name, KNOWN_GAP_CONTRACT_IDS, build_remote_command_inventory,
    )

    if inventory is None:
        inventory = build_remote_command_inventory()

    covered: list[dict] = []
    uncovered: list[dict] = []
    known_gaps: list[str] = []
    orphan_allowlist: list[str] = []
    arg_mismatches: list[str] = []

    # Track which allowlist entries have contracts
    contracted_allowlist_names: set[str] = set()

    for cid, c in contracts.items():
        is_gap = getattr(c, "is_known_gap", False) or cid in KNOWN_GAP_CONTRACT_IDS
        if is_gap:
            known_gaps.append(cid)
            continue

        canonical = get_canonical_command_name(c.command_name)
        entry = {
            "contract_id": cid,
            "command_name": c.command_name,
            "canonical_name": canonical,
            "handler_id": c.handler_id,
            "covered": False,
            "issues": [],
        }

        if canonical is None:
            entry["issues"].append("No canonical mapping")
        elif canonical not in ALLOWED_CLI_COMMANDS:
            entry["issues"].append(f"Canonical '{canonical}' not in allowlist")
        else:
            entry["covered"] = True
            contracted_allowlist_names.add(canonical)
            allowlisted_flags = ALLOWED_CLI_COMMANDS[canonical]
            for arg in c.allowed_args:
                if arg not in allowlisted_flags:
                    entry["issues"].append(f"arg '{arg}' not in allowlist")
                    arg_mismatches.append(f"{cid}:{arg}")

        if entry["covered"]:
            covered.append(entry)
        else:
            uncovered.append(entry)

    # Find allowlist entries with no contract
    for al_name in ALLOWED_CLI_COMMANDS:
        if al_name not in contracted_allowlist_names and al_name.startswith("run-"):
            # Check if any contract maps to this via canonical name
            has_contract = False
            for c in contracts.values():
                if get_canonical_command_name(c.command_name) == al_name:
                    has_contract = True
                    break
            if not has_contract:
                orphan_allowlist.append(al_name)

    total = len(covered) + len(uncovered)
    allowlist_coverage = len(covered) / max(total, 1)

    return {
        "total_contracts": total,
        "covered_by_allowlist": len(covered),
        "uncovered": len(uncovered),
        "known_gaps": len(known_gaps),
        "allowlist_coverage": allowlist_coverage,
        "orphan_allowlist_entries": orphan_allowlist,
        "argument_mismatch_count": len(arg_mismatches),
        "arg_mismatches": arg_mismatches,
        "covered": covered,
        "uncovered": uncovered,
        "known_gap_ids": known_gaps,
    }
