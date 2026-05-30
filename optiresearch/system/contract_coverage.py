"""Contract coverage dashboard for Phase 68."""
from __future__ import annotations

from typing import Any


def generate_contract_coverage() -> dict[str, Any]:
    """Compute contract coverage across all dimensions."""
    from optiresearch.system.capability_registry import build_system_capability_registry

    registry = build_system_capability_registry()
    handler_count = sum(1 for e in registry.entries if e.capability_type == "handler" and e.enabled)
    design_count = sum(1 for e in registry.entries if e.capability_type == "design")
    skill_count = sum(1 for e in registry.entries if e.capability_type == "skill")

    # Count contracts defined in test files
    try:
        from tests.test_core_handler_execution_contracts import get_all_contracts as get_exec
        exec_contracts = get_exec()
        exec_count = len(exec_contracts)
    except Exception:
        exec_count = 0

    try:
        from tests.test_core_artifact_contracts import get_all_artifact_contracts
        artifact_contracts = get_all_artifact_contracts()
        artifact_count = len(artifact_contracts)
    except Exception:
        artifact_count = 0

    try:
        from tests.test_remote_execution_contracts_core_commands import get_all_remote_contracts
        remote_contracts = get_all_remote_contracts()
        remote_count = len(remote_contracts)
    except Exception:
        remote_count = 0

    try:
        from tests.test_core_report_contracts import get_all_report_contracts
        report_contracts = get_all_report_contracts()
        report_count = len(report_contracts)
    except Exception:
        report_count = 0

    claim_policy_count = sum(1 for e in registry.entries if e.capability_type == "claim_policy")

    # Coverage ratios
    handler_contract_coverage = exec_count / max(handler_count, 1)
    design_mapping_coverage = _design_mapping_coverage(registry)
    remote_contract_coverage = remote_count / max(handler_count, 1)
    artifact_contract_coverage = artifact_count / max(handler_count, 1)
    report_contract_coverage = report_count / max(8, 1)
    claim_policy_coverage = claim_policy_count / max(16, 1)

    # Test coverage proxy: check if test files exist for each handler
    test_proxy = _compute_test_coverage_proxy(registry)

    # Doc coverage proxy: check if docs mention each handler
    doc_proxy = _compute_doc_coverage_proxy(registry)

    # Overall readiness score
    scores = [
        handler_contract_coverage,
        design_mapping_coverage,
        remote_contract_coverage,
        artifact_contract_coverage,
        report_contract_coverage,
        claim_policy_coverage,
        test_proxy["coverage_ratio"],
        doc_proxy["coverage_ratio"],
    ]
    overall_score = sum(scores) / len(scores)

    return {
        "dashboard_version": "0.1",
        "handler_contract_coverage": round(handler_contract_coverage, 3),
        "design_mapping_coverage": round(design_mapping_coverage, 3),
        "remote_contract_coverage": round(remote_contract_coverage, 3),
        "artifact_contract_coverage": round(artifact_contract_coverage, 3),
        "report_contract_coverage": round(report_contract_coverage, 3),
        "claim_policy_coverage": round(claim_policy_coverage, 3),
        "test_coverage_proxy": test_proxy,
        "doc_coverage_proxy": doc_proxy,
        "overall_system_readiness_score": round(overall_score, 3),
        "handler_count": handler_count,
        "skill_count": skill_count,
        "design_count": design_count,
        "execution_contract_count": exec_count,
        "remote_contract_count": remote_count,
        "artifact_contract_count": artifact_count,
        "report_contract_count": report_count,
    }


def _design_mapping_coverage(registry) -> float:
    designs = [e for e in registry.entries if e.capability_type == "design"]
    if not designs:
        return 1.0
    handlers = {e.capability_id for e in registry.entries if e.capability_type == "handler"}
    mapped = 0
    for d in designs:
        # Check if design name contains a handler-like prefix
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
