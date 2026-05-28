"""Component probe report exporter for Phase 62.

Generates a markdown report from a remote component probe job result.
"""

from __future__ import annotations

import json
from pathlib import Path


def export_component_probe_report(
    remote_job_id: str,
    remote_jobs_root: str | Path | None = None,
) -> Path:
    """Export a markdown report for a remote component probe job.

    Parameters
    ----------
    remote_job_id:
        The remote job ID (e.g. ``remote_job_<16 hex chars>``).
    remote_jobs_root:
        Root directory for remote job outputs.  Defaults to
        ``workspace/remote_jobs``.

    Returns
    -------
    Path
        Path to the generated markdown report.
    """
    root = Path(remote_jobs_root or "workspace/remote_jobs")
    job_dir = root / remote_job_id

    result_json = job_dir / "result.json"
    metrics_json = job_dir / "component_probe_metrics.json"
    job_result_json = job_dir / "remote_job_result.json"

    result = _read_json(result_json) if result_json.exists() else {}
    metrics = _read_json(metrics_json) if metrics_json.exists() else {}
    job_result = _read_json(job_result_json) if job_result_json.exists() else {}

    lines: list[str] = []

    lines.append("# Component Probe Report")
    lines.append("")
    lines.append(f"- **Remote Job ID:** `{remote_job_id}`")

    # Job-level metadata.
    lines.append(f"- **Job Status:** {job_result.get('status', 'unknown')}")
    if job_result.get("remote_run_id"):
        lines.append(f"- **Remote Run ID:** `{job_result['remote_run_id']}`")

    lines.append("")
    lines.append("## Component Result")
    lines.append("")

    for field, label in [
        ("component", "Component"),
        ("surface_class", "Surface Class"),
        ("status", "Status"),
        ("differentiable", "Differentiable"),
        ("autograd_graph_exists", "Autograd Graph Exists"),
        ("parameters_changed", "Parameters Changed"),
        ("trainable_param_count", "Trainable Param Count"),
        ("params_with_grad", "Params With Grad"),
        ("gradient_norm", "Gradient Norm"),
        ("loss_before", "Loss Before"),
        ("loss_after", "Loss After"),
        ("evidence_level", "Evidence Level"),
        ("claim_ceiling", "Claim Ceiling"),
        ("error_code", "Error Code"),
        ("error_message", "Error Message"),
    ]:
        value = result.get(field) if result else metrics.get(field)
        if value is not None:
            lines.append(f"- **{label}:** {value}")

    # Trainable parameter names.
    param_names = result.get("trainable_param_names", [])
    if param_names:
        lines.append(f"- **Trainable Parameter Names:** {', '.join(param_names)}")

    # Zero gradient parameters.
    zero_grad = result.get("zero_gradient_parameters", [])
    if zero_grad:
        lines.append(f"- **Zero Gradient Parameters:** {', '.join(zero_grad)}")

    # Checked component candidates.
    checked = result.get("checked_component_candidates", [])
    if checked:
        lines.append(f"- **Checked Component Candidates:** {', '.join(checked)}")

    lines.append("")

    # Caveats section.
    caveats = result.get("caveats", [])
    if caveats:
        lines.append("## Caveats")
        lines.append("")
        for c in caveats:
            lines.append(f"- {c}")
        lines.append("")

    # Warnings section.
    warnings = result.get("warnings", [])
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    # Claim boundaries.
    lines.append("## Claim Boundaries")
    lines.append("")
    lines.append(f"- **Evidence Level:** {result.get('evidence_level', 'diagnostic_evidence')}")
    lines.append(f"- **Claim Ceiling:** {result.get('claim_ceiling', 'diagnostic_evidence')}")
    lines.append("- **Lens-level optimization:** NOT supported by component probe")
    lines.append("- **HSI performance claims:** NOT supported by component probe")
    lines.append("- **Real camera validation:** NOT supported by component probe")
    lines.append("")
    lines.append("Component-level evidence does not generalize to lens-level claims.")
    lines.append("")

    # Blocked overclaims.
    lines.append("## Blocked Overclaims")
    lines.append("")
    lines.append("- full_geolens_direct_update (blocked route)")
    lines.append("- native_lens_optimization")
    lines.append("- hsi_improvement")
    lines.append("- real_camera_validation")
    lines.append("")

    report_path = job_dir / "component_probe_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
