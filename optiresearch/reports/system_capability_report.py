"""System Capability Report for Phase 68."""
from __future__ import annotations

from pathlib import Path


def export_system_capability_report(output_dir: str = "workspace/system_capability") -> Path:
    """Generate a comprehensive system capability report with 13 sections."""
    from optiresearch.system.capability_registry import build_system_capability_registry
    from collections import Counter

    registry = build_system_capability_registry()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lines = _build_report_lines(registry)
    md_path = out / "system_capability_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    import json
    json_path = out / "system_capability_report.json"
    json_path.write_text(registry.model_dump_json(indent=2), encoding="utf-8")

    return md_path


def _build_report_lines(registry) -> list[str]:
    from collections import Counter

    lines = [
        "# System Capability Report",
        "",
        f"**Generated At:** {registry.generated_at}",
        f"**Registry Version:** {registry.registry_version}",
        "",
        "## 1. System Overview",
        "",
        "This report provides a unified view of all registered system capabilities",
        "including handlers, skills, design strategies, optical backends, and claim policies.",
        "",
        "## 2. Capability Registry Summary",
        "",
    ]

    by_type = Counter(e.capability_type for e in registry.entries)
    lines.append(f"- **Total Entries:** {len(registry.entries)}")
    for t, c in sorted(by_type.items()):
        lines.append(f"- **{t}:** {c}")

    lines.extend(["", "## 3. Handler Capability Table", ""])
    handlers = [e for e in registry.entries if e.capability_type == "handler"]
    lines.append("| handler_id | evidence_level | max_claim_ceiling | enabled | maturity | remote |")
    lines.append("|---|---|---|---|---|---|")
    for h in sorted(handlers, key=lambda x: x.capability_id):
        lines.append(f"| {h.capability_id} | {h.evidence_level} | {h.max_claim_ceiling} | {h.enabled} | {h.maturity_level} | {h.supports_remote} |")

    lines.extend(["", "## 4. Skill/Design Mapping Table", ""])
    skills = [e for e in registry.entries if e.capability_type == "skill"]
    designs = [e for e in registry.entries if e.capability_type == "design"]
    lines.append(f"- **Skills:** {len(skills)}")
    lines.append(f"- **Designs:** {len(designs)}")

    lines.extend(["", "## 5. Execution Contract Coverage", ""])
    lines.append("Execution contracts link handlers to skills, designs, and backends.")

    lines.extend(["", "## 6. Remote Execution Contract Coverage", ""])
    lines.append("Remote execution contracts govern WSL worker command execution.")

    lines.extend(["", "## 7. Artifact Contract Coverage", ""])
    lines.append("Artifact contracts specify required outputs for each handler type.")

    lines.extend(["", "## 8. Report Contract Coverage", ""])
    lines.append("Report contracts specify required sections for each report type.")

    lines.extend(["", "## 9. ClaimGate Policy Matrix Summary", ""])
    policies = [e for e in registry.entries if e.capability_type == "claim_policy"]
    lines.append(f"- **Evidence levels covered:** {len(policies)}")

    lines.extend(["", "## 10. Known Gaps", ""])
    vs = registry.validation_summary
    lines.append(f"- Missing evidence_level: {vs.get('missing_evidence_level', 'N/A')}")
    lines.append(f"- Missing claim_ceiling: {vs.get('missing_claim_ceiling', 'N/A')}")
    lines.append(f"- Inconsistent ceilings: {vs.get('inconsistent_ceilings', 'N/A')}")

    lines.extend(["", "## 11. Unsupported / Needs Followup Capabilities", ""])
    unsupported = [e for e in registry.entries if e.evidence_level in ("unsupported", "needs_followup", "structured_unsupported")]
    for u in unsupported:
        lines.append(f"- {u.capability_id} ({u.capability_type}): {u.evidence_level}")

    lines.extend(["", "## 12. Recommended Next Modules", ""])
    lines.append("1. Real HSI validation with physical camera data")
    lines.append("2. Wave-optics coherent propagation validation")
    lines.append("3. Cross-backend benchmark matrix")
    lines.append("4. Production deployment readiness assessment")

    lines.extend(["", "## 13. What Not to Claim", ""])
    lines.append("- Synthetic/mock results MUST NOT be presented as real HSI performance")
    lines.append("- Component-level results MUST NOT be presented as full lens validation")
    lines.append("- Geometric optics MUST NOT be presented as wave-optics coherent propagation")
    lines.append("- Remote execution availability MUST NOT imply production readiness")

    return lines
