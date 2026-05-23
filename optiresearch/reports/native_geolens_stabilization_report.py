"""Native GeoLens stabilization sweep report for Phase 35."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def export_native_geolens_stabilization_report(
    sweep_id: str,
    output_root: str | Path | None = None,
) -> Path:
    root = Path(output_root or os.getenv("OPTIRESEARCH_WORKSPACE", "workspace"))
    sweep_dir = root / "native_geolens_stabilization" / sweep_id
    results = _read_json(sweep_dir / "sweep_results.json", {})
    best_config = _read_json(sweep_dir / "best_config.json", {})
    path = sweep_dir / "stabilization_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_markdown(sweep_id, results, best_config), encoding="utf-8")
    return path


def _markdown(sweep_id: str, results: dict[str, Any], best_config: dict[str, Any]) -> str:
    best_result = best_config.get("result", {})
    lines = [
        "# Native GeoLens Stabilization Sweep Report",
        "",
        f"**Sweep ID:** `{sweep_id}`",
        f"**Configs Tested:** {results.get('configs_tested', 0)}",
        f"**Configs Succeeded:** {results.get('configs_succeeded', 0)}",
        f"**Configs with Accepted Updates:** {results.get('configs_with_accepted_updates', 0)}",
        "",
        "## Objective",
        "",
        "Stabilize native GeoLens optical updates on WSL so that at least one "
        "configuration achieves `accepted_update_count > 0` and ideally "
        "`stable_training_succeeded=true`.",
        "",
        "## Phase 34 Recap",
        "",
        "- `optical_gradient_norm=4098` (high) → updates overshoot even at `lr=1e-6`",
        "- `accepted_update_count=0`, `rejected_update_count=2` → all updates rejected by rollback",
        "- `reconstruction_loss 0.924978 → 0.924564` → reconstructor-only improvement",
        "",
        "## Sweep Configuration",
        "",
        "| Parameter | Values |",
        "|---|---|",
        "| optical_lr | 1e-6, 5e-7, 1e-7, 5e-8, 1e-8 |",
        "| optical_grad_clip | 1.0, 0.1, 0.01 |",
        "| trust_region_enabled | True |",
        "| max_optical_param_delta | 1e-3, 1e-4 |",
        "| rollback_on_psf_instability | True |",
        "| accept_tolerance | 1e-6 |",
        "",
        "## Best Configuration",
        "",
        f"**Name:** {best_config.get('name', '-')}",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    best_cfg = best_config.get("config", {})
    for k, v in sorted(best_cfg.items()):
        lines.append(f"| {k} | {v} |")

    lines.extend([
        "",
        "### Best Result",
        "",
        "| Metric | Value |",
        "|---|---|",
    ])
    for k, v in sorted(best_result.items()):
        if k != "overrides":
            lines.append(f"| {k} | {v} |")

    lines.extend([
        "",
        "## Accepted / Rejected Update Analysis",
        "",
        f"- accepted_update_count: {best_result.get('accepted_update_count', 0)}",
        f"- rejected_update_count: {best_result.get('rejected_update_count', 0)}",
        f"- rollback_count: {best_result.get('rollback_count', 0)}",
        f"- stable_training_succeeded: {best_result.get('stable_training_succeeded', False)}",
        "",
        "## PSF Stability Analysis",
        "",
        f"- psf_energy_delta: {best_result.get('psf_energy_delta', '-')}",
        f"- psf_width_delta: {best_result.get('psf_width_delta', '-')}",
        f"- trust_region_activated: {best_result.get('trust_region_activated', False)}",
        "",
        "## ClaimGate Decision",
        "",
    ])

    accepted = best_result.get("accepted_update_count", 0)
    stable = best_result.get("stable_training_succeeded", False)
    if accepted > 0 and stable:
        lines.append("- **stable native GeoLens optical update**: SUPPORTED")
    elif accepted > 0:
        lines.append("- **stable native GeoLens optical update**: PARTIAL (accepted updates but training not stable)")
    else:
        lines.append("- **stable native GeoLens optical update**: NOT SUPPORTED (no accepted updates)")
    lines.append("- **rollback-protected native GeoLens HSI**: SUPPORTED (Phase 34)")
    lines.append("- **full wave-optics HSI**: NOT SUPPORTED (geometric PSF only)")
    lines.append("- **real HSI**: NOT SUPPORTED (synthetic data only)")

    lines.extend([
        "",
        "## Remaining Limitations",
        "",
        "- Still limited to geometric PSF (not full wave-optics)",
        "- Synthetic dataset — no real camera validation",
        "- WSL-only execution — macOS still fails with GeoLens API IndexError",
    ])

    if accepted == 0:
        lines.append("- No configuration achieved accepted optical updates — native improvement claim NOT supported")

    return "\n".join(lines) + "\n"


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
