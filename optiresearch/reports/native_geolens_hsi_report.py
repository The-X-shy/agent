"""Native GeoLens HSI report export."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def export_native_geolens_hsi_report(
    run_id: str,
    output_root: str | Path | None = None,
) -> Path:
    root = Path(output_root or os.getenv("OPTIRESEARCH_WORKSPACE", "workspace"))
    run_dir = root / "native_geolens_hsi" / run_id
    spec = _read_json(run_dir / "spec.json", {})
    result = _read_json(run_dir / "result.json", {})
    path = run_dir / "native_geolens_hsi_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_markdown(run_id, spec, result), encoding="utf-8")
    return path


def _markdown(run_id: str, spec: dict[str, Any], result: dict[str, Any]) -> str:
    lines = [
        "# Native GeoLens HSI Report",
        "",
        f"**Run ID:** `{run_id}`",
        f"**Status:** {result.get('status', 'unknown')}",
        f"**Candidate:** {spec.get('candidate', result.get('candidate', '-'))}",
        f"**Reconstructor:** {spec.get('reconstructor', result.get('reconstructor', '-'))}",
        f"**Error code:** {result.get('error_code', '-')}",
        "",
        "## Metrics",
        "",
        "| Metric | Before | After |",
        "|---|---|---|",
    ]
    metric_pairs = [
        ("Reconstruction Loss", "reconstruction_loss_before", "reconstruction_loss_after"),
        ("MSE", "mse_before", "mse_after"),
        ("PSNR", "psnr_before", "psnr_after"),
        ("Spectral Angle", "spectral_angle_before", "spectral_angle_after"),
        ("SAM", "sam_before", "sam_after"),
        ("Measurement Consistency", "measurement_consistency_before", "measurement_consistency_after"),
    ]
    for label, before_key, after_key in metric_pairs:
        b = result.get(before_key)
        a = result.get(after_key)
        if b is not None or a is not None:
            lines.append(f"| {label} | {_fmt(b)} | {_fmt(a)} |")
    lines.append("")

    lines.extend([
        "## Execution Fidelity",
        "",
        "| Field | Value |",
        "|---|---|",
    ])
    fidelity_fields = [
        ("execution_fidelity", "deeplens_native_geometric"),
        ("actual_execution_fidelity", result.get("actual_execution_fidelity")),
        ("proxy_fallback_used", result.get("proxy_fallback_used")),
        ("deeplens_native_psf_path", result.get("deeplens_native_psf_path")),
        ("full_wave_optics", result.get("full_wave_optics")),
        ("phase_to_fft_proxy_used", result.get("phase_to_fft_proxy_used")),
        ("platform", result.get("platform")),
    ]
    for key, value in fidelity_fields:
        if value not in (None, "", "None"):
            lines.append(f"| {key} | {value} |")
    lines.append("")

    lines.extend([
        "## Optical Parameters",
        "",
        "| Field | Value |",
        "|---|---|",
    ])
    opt_fields = [
        ("optical_gradient_norm", result.get("optical_gradient_norm")),
        ("optical_parameters_changed", result.get("optical_parameters_changed")),
        ("rollback_count", result.get("rollback_count")),
        ("accepted_update_count", result.get("accepted_update_count")),
        ("rejected_update_count", result.get("rejected_update_count")),
        ("stable_training_succeeded", result.get("stable_training_succeeded")),
        ("evidence_level", result.get("evidence_level")),
    ]
    for key, value in opt_fields:
        if value not in (None, "", "None"):
            lines.append(f"| {key} | {value} |")
    lines.append("")

    lines.extend(["## Caveats", ""])
    caveats = result.get("caveats", [])
    if caveats:
        for c in caveats:
            lines.append(f"- {c}")
    else:
        lines.append("No caveats recorded.")

    return "\n".join(lines) + "\n"


def _fmt(val: Any) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.6f}"
    return str(val)


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
