"""Remote diagnostic report exporter with lens resolution section."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def export_remote_diagnostic_report(
    remote_job_id: str,
    remote_jobs_root: str | Path | None = None,
) -> Path:
    root = Path(remote_jobs_root or os.getenv("OPTIRESEARCH_REMOTE_JOBS_ROOT", "workspace/remote_jobs"))
    job_dir = root / remote_job_id
    result = _read_json(job_dir / "remote_job_result.json", {})
    metrics = result.get("metrics_summary", {})
    path = job_dir / "remote_diagnostic_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_markdown(remote_job_id, result, metrics), encoding="utf-8")
    return path


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


def _markdown(job_id: str, result: dict[str, Any], metrics: dict[str, Any]) -> str:
    lines = [
        "# Remote Diagnostic Report",
        "",
        f"**Job ID:** `{job_id}`",
        f"**Status:** {result.get('status', 'unknown')}",
        f"**Error code:** {result.get('error_code', 'none')}",
        "",
        "## Lens Resolution",
        "",
    ]
    lines.extend(_lens_section(metrics))
    lines.extend(["", "## Diagnostic Results", ""])
    lines.extend(_diagnostic_section(metrics))
    lines.extend(["", "## Gradient Flow Interpretation", ""])
    lines.extend(_gradient_interpretation(metrics))
    lines.extend(["", "## Caveats", "",
                  "- Evidence level is `diagnostic_evidence` — no optical improvement is claimed.",
                  "- Remote execution may differ from local results due to platform differences.",
    ])
    return "\n".join(lines) + "\n"


def _lens_section(metrics: dict[str, Any]) -> list[str]:
    lines = [
        f"| Field | Value |",
        f"|---|---|",
        f"| requested_lens_file | {metrics.get('requested_lens_file', 'N/A')} |",
        f"| resolved_lens_file | `{metrics.get('resolved_lens_file')}` |",
        f"| lens_resolution_source | {metrics.get('lens_resolution_source', 'N/A')} |",
        f"| checked_paths_count | {len(metrics.get('checked_lens_paths', []))} |",
        f"| error_code | {metrics.get('error_code', 'none')} |",
    ]
    alternatives = metrics.get("alternatives", [])
    if alternatives:
        lines.append(f"| alternatives | {', '.join(str(a) for a in alternatives)} |")
    return lines


def _diagnostic_section(metrics: dict[str, Any]) -> list[str]:
    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| trainable_param_count | {metrics.get('trainable_param_count', 'N/A')} |",
        f"| parameter_count | {metrics.get('parameter_count', 'N/A')} |",
        f"| params_with_grad | {metrics.get('params_with_grad', 'N/A')} |",
        f"| zero_gradient_parameters | {metrics.get('zero_gradient_parameters', 'N/A')} |",
        f"| grad_norm_max | {metrics.get('grad_norm_max', 'N/A')} |",
        f"| grad_norm_mean | {metrics.get('grad_norm_mean', 'N/A')} |",
        f"| graph_connected | {metrics.get('graph_connected', 'N/A')} |",
        f"| psf_requires_grad | {metrics.get('psf_requires_grad', 'N/A')} |",
        f"| loss_requires_grad | {metrics.get('loss_requires_grad', 'N/A')} |",
        f"| candidate_update_changes_parameter | {metrics.get('candidate_update_changes_parameter', 'N/A')} |",
        f"| detach_suspected | {metrics.get('detach_suspected', 'N/A')} |",
        f"| recommended_next_strategy | {metrics.get('recommended_next_strategy', 'N/A')} |",
        f"| recommended_trainable_subset | {metrics.get('recommended_trainable_subset', 'N/A')} |",
    ]
    return lines


def _gradient_interpretation(metrics: dict[str, Any]) -> list[str]:
    tp = metrics.get("trainable_param_count", 0)
    pwg = metrics.get("params_with_grad", 0)
    gc = metrics.get("graph_connected", False)
    cucp = metrics.get("candidate_update_changes_parameter", False)

    lines = []
    if tp and tp > 0 and pwg == 0:
        lines.append("- **Likely cause:** gradient flow blocked — no parameters receive gradients.")
        lines.append("- **Recommendation:** verify autograd chain from PSF through loss.")
    elif pwg and pwg > 0 and not cucp:
        lines.append("- **Likely cause:** optimizer update blocked — gradients exist but parameters unchanged.")
        lines.append("- **Recommendation:** check optimizer step and parameter registration.")
    elif gc and not cucp:
        lines.append("- **Likely cause:** objective or update instability despite connected graph.")
        lines.append("- **Recommendation:** reduce learning rate, add gradient clipping.")
    elif gc:
        lines.append("- **Interpretation:** gradient flow is functional.")
        lines.append("- **Recommendation:** proceed to curriculum probe.")
    else:
        lines.append("- **Interpretation:** gradient flow status unclear — insufficient metrics.")
    return lines
