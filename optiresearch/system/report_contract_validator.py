"""Validate report contracts against report output files (Phase 68)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from optiresearch.schemas.report_contract import ReportContract


def validate_report_contract(
    report_path: str | Path,
    contract: ReportContract,
) -> dict[str, Any]:
    """Check that a report file satisfies a report contract."""
    rp = Path(report_path)
    issues: list[str] = []
    present_sections: list[str] = []
    missing_sections: list[str] = []

    if not rp.exists():
        return {
            "report_path": str(rp),
            "contract_id": contract.report_contract_id,
            "status": "report_missing",
            "issues": [f"Report file does not exist: {rp}"],
        }

    content = rp.read_text(encoding="utf-8")
    content_lower = content.lower()

    # Check required sections (by looking for markdown headings)
    for section in contract.required_sections:
        section_lower = section.lower()
        if f"# {section_lower}" in content_lower or f"## {section_lower}" in content_lower:
            present_sections.append(section)
        else:
            missing_sections.append(section)
            issues.append(f"Missing required section: {section}")

    # Check blocked claims section if required
    if contract.blocked_claims_section_required:
        if "blocked claim" not in content_lower and "unsupported claim" not in content_lower:
            issues.append("Blocked claims section required but not found")

    # Check evidence level section if required
    if contract.evidence_level_section_required:
        if "evidence level" not in content_lower:
            issues.append("Evidence level section required but not found")

    # Check safe wording if required
    if contract.safe_wording_required:
        if "safe wording" not in content_lower and "claim wording" not in content_lower:
            issues.append("Safe wording section required but not found")

    # Check artifact links
    for artifact in contract.linked_artifacts:
        if artifact.lower() not in content_lower:
            issues.append(f"Linked artifact '{artifact}' not referenced in report")

    status = "passed" if not missing_sections and not issues else "issues_found"

    return {
        "report_path": str(rp),
        "contract_id": contract.report_contract_id,
        "status": status,
        "required_sections_total": len(contract.required_sections),
        "sections_present": len(present_sections),
        "sections_missing": len(missing_sections),
        "present_sections": present_sections,
        "missing_sections": missing_sections,
        "issues": issues,
    }
