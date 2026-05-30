"""Validate execution contracts against the system capability registry (Phase 68)."""
from __future__ import annotations

from typing import Any

from optiresearch.schemas.execution_contract import ExecutionContract
from optiresearch.schemas.system_capability import SystemCapabilityRegistry


def validate_execution_contracts(
    contracts: dict[str, ExecutionContract],
    registry: SystemCapabilityRegistry | None = None,
) -> dict[str, Any]:
    """Check execution contracts against registry entries. Returns validation report."""
    results: list[dict[str, Any]] = []
    handler_ids = set()
    skill_ids = set()
    design_ids = set()

    if registry:
        handler_ids = {e.capability_id for e in registry.entries if e.capability_type == "handler"}
        skill_ids = {e.capability_id for e in registry.entries if e.capability_type == "skill"}
        design_ids = {e.capability_id for e in registry.entries if e.capability_type == "design"}

    missing_contracts: list[str] = []
    invalid_handler_refs: list[str] = []
    invalid_skill_refs: list[str] = []
    invalid_design_refs: list[str] = []
    inconsistent_ceilings: list[str] = []
    missing_execution_modes: list[str] = []
    missing_required_outputs: list[str] = []

    for cid, c in contracts.items():
        entry_result = {
            "contract_id": cid,
            "handler_id": c.handler_id,
            "valid": True,
            "issues": [],
        }

        # Rule 1: handler_id must be valid
        if handler_ids and c.handler_id not in handler_ids:
            entry_result["valid"] = False
            entry_result["issues"].append(f"handler_id '{c.handler_id}' not in registry")
            invalid_handler_refs.append(cid)

        # Rule 2: skill_id must be valid (if set)
        if skill_ids and c.skill_id and c.skill_id not in skill_ids:
            entry_result["issues"].append(f"skill_id '{c.skill_id}' not in registry")
            invalid_skill_refs.append(cid)

        # Rule 3: design_ids must be valid
        if design_ids and c.design_ids:
            for did in c.design_ids:
                if did not in design_ids:
                    entry_result["issues"].append(f"design_id '{did}' not in registry")
                    invalid_design_refs.append(cid)

        # Rule 4: evidence_level must not exceed claim_ceiling
        from optiresearch.memory.claim_gate_v2 import _evidence_rank
        for mode, ev in c.evidence_level_mapping.items():
            ceil = c.claim_ceiling_mapping.get(mode, "")
            if _evidence_rank(ev) > _evidence_rank(ceil) > 0:
                entry_result["issues"].append(f"evidence_level '{ev}' exceeds claim_ceiling '{ceil}' for mode '{mode}'")
                inconsistent_ceilings.append(cid)

        # Rule 5: execution_modes required
        if not c.execution_modes:
            entry_result["issues"].append("no execution_modes specified")
            missing_execution_modes.append(cid)

        # Rule 6: required_outputs should be non-empty
        if not c.required_outputs:
            entry_result["issues"].append("no required_outputs specified")
            missing_required_outputs.append(cid)

        results.append(entry_result)

    valid_contracts = [r for r in results if r["valid"]]
    invalid_contracts = [r for r in results if not r["valid"]]
    total_issues = sum(len(r["issues"]) for r in results)

    return {
        "validation_status": "passed" if not invalid_contracts else "issues_found",
        "total_contracts": len(contracts),
        "valid_contracts": len(valid_contracts),
        "invalid_contracts": len(invalid_contracts),
        "total_issues": total_issues,
        "missing_contracts": missing_contracts,
        "invalid_handler_refs": invalid_handler_refs,
        "invalid_skill_refs": invalid_skill_refs,
        "invalid_design_refs": invalid_design_refs,
        "inconsistent_claim_ceilings": inconsistent_ceilings,
        "missing_execution_modes": missing_execution_modes,
        "missing_required_outputs": missing_required_outputs,
        "results": results,
    }
