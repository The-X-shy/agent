"""Contract coverage dashboard for Phase 69 — reconciled metrics."""
from __future__ import annotations

from typing import Any


def generate_contract_coverage() -> dict[str, Any]:
    """Compute contract coverage across all dimensions with Phase 69 upgrades."""
    from optiresearch.system.capability_registry import build_system_capability_registry

    registry = build_system_capability_registry()
    handler_count = sum(1 for e in registry.entries if e.capability_type == "handler" and e.enabled)
    design_count = sum(1 for e in registry.entries if e.capability_type == "design")
    skill_count = sum(1 for e in registry.entries if e.capability_type == "skill")

    # Count remote-supporting handlers
    remote_handler_count = sum(
        1 for e in registry.entries
        if e.capability_type == "handler" and e.enabled and e.supports_remote
    )

    try:
        from tests.test_core_handler_execution_contracts import get_all_contracts as get_exec
        exec_count = len(get_exec())
    except Exception:
        exec_count = 0

    try:
        from tests.test_core_artifact_contracts import get_all_artifact_contracts
        artifact_contracts = get_all_artifact_contracts()
        artifact_count = len(artifact_contracts)
    except Exception:
        artifact_count = 0

    try:
        from tests.test_remote_execution_contracts_core_commands import (
            get_all_remote_contracts, get_known_gaps,
        )
        remote_contracts = get_all_remote_contracts()
        gaps = get_known_gaps()
        remote_count = len(remote_contracts)
        known_gap_count = len(gaps)
    except Exception:
        remote_count = 0
        known_gap_count = 0

    try:
        from tests.test_core_report_contracts import get_all_report_contracts
        report_contracts = get_all_report_contracts()
        report_count = len(report_contracts)
    except Exception:
        report_count = 0

    claim_policy_count = sum(1 for e in registry.entries if e.capability_type == "claim_policy")

    # Phase 69: Allowlist coverage using canonical mapping
    allowlist_cov = _compute_allowlist_coverage()
    remote_allowlist_coverage = allowlist_cov.get("allowlist_coverage", 0.0)

    # Phase 69: Artifact handler_id validity
    artifact_registry = _validate_artifact_handler_ids()
    artifact_handler_id_valid = artifact_registry.get("valid_contracts_to_check", 0)

    # Coverage ratios
    handler_contract_coverage = exec_count / max(handler_count, 1)
    design_mapping_coverage = _design_mapping_coverage(registry)
    remote_contract_coverage = remote_count / max(remote_handler_count, 1)
    artifact_contract_coverage = artifact_count / max(handler_count, 1)
    report_contract_coverage = report_count / max(8, 1)
    claim_policy_coverage = claim_policy_count / max(16, 1)

    # Phase 69 new metrics
    remote_result_parser_coverage = _compute_result_parser_coverage(remote_contracts, gaps)
    artifact_evidence_role_coverage = _compute_evidence_role_coverage(artifact_contracts)
    canonical_mapping_coverage = allowlist_cov.get("canonical_mapping_ratio", 0.0)

    # Test and doc proxies
    test_proxy = _compute_test_coverage_proxy(registry)
    doc_proxy = _compute_doc_coverage_proxy(registry)

    # Overall readiness score with Phase 69 penalization
    base_scores = [
        handler_contract_coverage,
        design_mapping_coverage,
        remote_contract_coverage,
        artifact_contract_coverage,
        report_contract_coverage,
        claim_policy_coverage,
        remote_allowlist_coverage,
        artifact_handler_id_valid / max(artifact_count, 1),
        test_proxy["coverage_ratio"],
        doc_proxy["coverage_ratio"],
    ]
    base_score = sum(base_scores) / len(base_scores)

    # Hard penalties
    penalties = 0.0
    invalid_handler_count = artifact_registry.get("invalid_handler_ids_count", 0)
    if invalid_handler_count > 0:
        penalties += min(0.10, invalid_handler_count * 0.02)
    if remote_allowlist_coverage < 0.85:
        penalties += 0.05
    if artifact_handler_id_valid < artifact_count:
        penalties += 0.05

    overall_score = max(0.0, min(1.0, base_score - penalties))

    return {
        "dashboard_version": "0.2",
        "handler_contract_coverage": round(handler_contract_coverage, 3),
        "design_mapping_coverage": round(design_mapping_coverage, 3),
        "remote_contract_coverage": round(remote_contract_coverage, 3),
        "artifact_contract_coverage": round(artifact_contract_coverage, 3),
        "report_contract_coverage": round(report_contract_coverage, 3),
        "claim_policy_coverage": round(claim_policy_coverage, 3),
        "remote_allowlist_coverage": round(remote_allowlist_coverage, 3),
        "remote_result_parser_coverage": round(remote_result_parser_coverage, 3),
        "artifact_evidence_role_coverage": round(artifact_evidence_role_coverage, 3),
        "canonical_mapping_coverage": round(canonical_mapping_coverage, 3),
        "test_coverage_proxy": test_proxy,
        "doc_coverage_proxy": doc_proxy,
        "overall_system_readiness_score": round(overall_score, 3),
        "penalties_applied": round(penalties, 3),
        "handler_count": handler_count,
        "remote_handler_count": remote_handler_count,
        "skill_count": skill_count,
        "design_count": design_count,
        "execution_contract_count": exec_count,
        "remote_contract_count": remote_count,
        "known_gap_contracts": known_gap_count,
        "artifact_contract_count": artifact_count,
        "report_contract_count": report_count,
        "invalid_handler_id_count": invalid_handler_count,
    }


def _compute_allowlist_coverage() -> dict[str, Any]:
    try:
        from tests.test_remote_execution_contracts_core_commands import (
            get_all_remote_contracts, get_known_gaps,
        )
        from optiresearch.remote.command_allowlist import ALLOWED_CLI_COMMANDS
        from optiresearch.system.remote_command_inventory import get_canonical_command_name

        contracts = get_all_remote_contracts()
        gaps = get_known_gaps()
        active = len(contracts)
        covered = 0
        canonical_count = 0

        for cid, c in contracts.items():
            canonical = get_canonical_command_name(c.command_name)
            if canonical is not None:
                canonical_count += 1
                if canonical in ALLOWED_CLI_COMMANDS:
                    covered += 1

        return {
            "allowlist_coverage": covered / max(active, 1),
            "canonical_mapping_ratio": canonical_count / max(active, 1),
            "active_contracts": active,
            "covered": covered,
            "known_gaps": len(gaps),
        }
    except Exception:
        return {"allowlist_coverage": 0.0, "canonical_mapping_ratio": 0.0}


def _validate_artifact_handler_ids() -> dict[str, Any]:
    try:
        from tests.test_core_artifact_contracts import get_all_artifact_contracts
        from optiresearch.skills.handler_capability_registry import get_handler_capability_registry
        reg = get_handler_capability_registry()
        valid_ids = {h.handler_id for h in reg.list_enabled()}
        contracts = get_all_artifact_contracts()
        invalid_ids = []
        to_check = 0
        for cid, c in contracts.items():
            if c.handler_id == "system_capability":
                continue
            to_check += 1
            if c.handler_id and c.handler_id not in valid_ids:
                invalid_ids.append(cid)
        return {
            "valid_contracts_to_check": to_check - len(invalid_ids),
            "total_contracts_to_check": to_check,
            "invalid_handler_ids_count": len(invalid_ids),
            "invalid_handler_ids": invalid_ids,
        }
    except Exception:
        return {"valid_contracts_to_check": 0, "invalid_handler_ids_count": 0}


def _compute_result_parser_coverage(remote_contracts: dict, gaps: dict) -> float:
    active = len(remote_contracts)
    with_parser = sum(1 for c in remote_contracts.values() if c.result_parser)
    return with_parser / max(active, 1)


def _compute_evidence_role_coverage(artifact_contracts: dict) -> float:
    total_artifacts = 0
    artifacts_with_roles = 0
    for c in artifact_contracts.values():
        total_artifacts += len(c.required_artifacts) + len(c.optional_artifacts)
        artifacts_with_roles += len(c.artifact_roles)
    return artifacts_with_roles / max(total_artifacts, 1)


def _design_mapping_coverage(registry) -> float:
    designs = [e for e in registry.entries if e.capability_type == "design"]
    if not designs:
        return 1.0
    handlers = {e.capability_id for e in registry.entries if e.capability_type == "handler"}
    mapped = 0
    for d in designs:
        for h in handlers:
            if h in d.capability_id or d.capability_id in h:
                mapped += 1
                break
    return mapped / len(designs)


def _compute_test_coverage_proxy(registry) -> dict[str, Any]:
    from pathlib import Path
    test_dir = Path("tests")
    test_files = set()
    if test_dir.exists():
        test_files = {f.stem for f in test_dir.glob("test_*.py")}
    handlers = [e for e in registry.entries if e.capability_type == "handler" and e.enabled]
    covered = 0
    details = {}
    for h in handlers:
        has_test = any(
            h.capability_id.replace("_", "") in tf.replace("test_", "").replace("_", "")
            or h.capability_id.split("_")[0] in tf
            for tf in test_files
        )
        details[h.capability_id] = has_test
        if has_test:
            covered += 1
    return {
        "handlers_with_tests": covered,
        "total_handlers": len(handlers),
        "coverage_ratio": round(covered / max(len(handlers), 1), 3),
        "details": details,
    }


def _compute_doc_coverage_proxy(registry) -> dict[str, Any]:
    from pathlib import Path
    docs_dir = Path("docs")
    doc_text = ""
    if docs_dir.exists():
        for doc_file in docs_dir.glob("*.md"):
            try:
                doc_text += doc_file.read_text(encoding="utf-8")[:2000].lower()
            except Exception:
                pass
    handlers = [e for e in registry.entries if e.capability_type == "handler" and e.enabled]
    covered = 0
    details = {}
    for h in handlers:
        has_doc = h.capability_id.lower() in doc_text or h.capability_id.replace("_", " ") in doc_text
        details[h.capability_id] = has_doc
        if has_doc:
            covered += 1
    return {
        "handlers_with_docs": covered,
        "total_handlers": len(handlers),
        "coverage_ratio": round(covered / max(len(handlers), 1), 3),
        "details": details,
    }
