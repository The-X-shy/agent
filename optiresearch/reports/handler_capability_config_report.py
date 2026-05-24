"""Handler Capability Config Report for Phase 42."""

from __future__ import annotations

import json
from pathlib import Path


def export_handler_capability_config_report(output_dir: str = "workspace/reports") -> tuple[Path, Path]:
    from optiresearch.skills.handler_capability_registry import (
        get_handler_capability_registry,
    )
    registry = get_handler_capability_registry()
    enabled = registry.list_enabled()
    disabled = registry.list_disabled()
    all_caps = registry.list_all()

    remote_count = sum(1 for c in all_caps if c.supports_remote)
    remote_required_count = sum(1 for c in all_caps if c.remote_required)
    remote_validation_count = sum(1 for c in all_caps if c.requires_remote_validation)

    # JSON report
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "handler_capability_config_report.json"
    json_data = {
        "schema_version": registry.schema_version,
        "loaded_from_config": registry.loaded_from_config,
        "enabled_count": len(enabled),
        "disabled_count": len(disabled),
        "remote_awareness": {
            "supports_remote": remote_count,
            "remote_required": remote_required_count,
            "requires_remote_validation": remote_validation_count,
        },
        "handlers": [
            {
                "handler_id": c.handler_id,
                "design_type": c.design_type,
                "enabled": c.enabled,
                "actual_evidence_level": c.actual_evidence_level,
                "max_claim_ceiling": c.max_claim_ceiling,
                "supports_remote": c.supports_remote,
                "supported_modes": c.supported_execution_modes,
            }
            for c in all_caps
        ],
    }
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown report
    md_path = out_dir / "handler_capability_config_report.md"
    lines = [
        "# Handler Capability Config Report",
        "",
        f"**Schema Version:** {registry.schema_version or 'N/A'}",
        f"**Loaded from Config:** {registry.loaded_from_config}",
        f"**Enabled Handlers:** {len(enabled)}",
        f"**Disabled Handlers:** {len(disabled)}",
        "",
        "## Remote Awareness",
        f"- **Supports Remote:** {remote_count}",
        f"- **Remote Required:** {remote_required_count}",
        f"- **Requires Remote Validation:** {remote_validation_count}",
        "",
        "## Enabled Handlers",
        "| Handler | Type | Evidence | Ceiling | Remote |",
        "|---|---|---|---|---|",
    ]
    for c in enabled:
        lines.append(
            f"| {c.handler_id} | {c.design_type} | {c.actual_evidence_level} | "
            f"{c.max_claim_ceiling} | {'yes' if c.supports_remote else 'no'} |"
        )
    lines.extend([
        "",
        "## Disabled Handlers",
        "| Handler | Type | Evidence | Ceiling | Remote |",
        "|---|---|---|---|---|",
    ])
    for c in disabled:
        lines.append(
            f"| {c.handler_id} | {c.design_type} | {c.actual_evidence_level} | "
            f"{c.max_claim_ceiling} | {'yes' if c.supports_remote else 'no'} |"
        )
    lines.extend([
        "",
        "## Backward Compatibility Check",
        "| Handler | Expected Ceiling | Actual Ceiling | Match |",
        "|---|---|---|---|",
    ])
    expected = {
        "objective_redesign_simpler_metric": "lightweight_scientific_execution",
        "param_reduction_sweep": "lightweight_scientific_execution",
        "backend_switch_waveoptics_coherent": "structured_unsupported",
        "report_negative_result_doc": "report_only",
        "real_data_request": "requires_user_data",
    }
    for hid, expected_ceiling in expected.items():
        cap = registry.get(hid)
        actual = cap.max_claim_ceiling if cap else "MISSING"
        match = "yes" if actual == expected_ceiling else "NO"
        lines.append(f"| {hid} | {expected_ceiling} | {actual} | {match} |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path
