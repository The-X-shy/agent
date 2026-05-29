"""Component surrogate HSI co-design report exporter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_component_surrogate_hsi_report(run_id: str) -> Path:
    run_dir = Path("workspace/component_surrogate_hsi") / run_id
    result = _read_json(run_dir / "result.json", {})
    metrics = _read_json(run_dir / "metrics.json", {})
    path = run_dir / "component_surrogate_hsi_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_markdown(run_id, result, metrics), encoding="utf-8")
    return path


def _markdown(run_id: str, result: dict[str, Any], metrics: dict[str, Any]) -> str:
    component = result.get("component_type", metrics.get("component_type", "-"))
    lines = [
        "# Component Surrogate HSI Co-design Report",
        "",
        f"**Run ID:** `{run_id}`",
        f"**Component:** {component}",
        f"**Status:** {result.get('status', '-')}",
        "",
        "## 1. Component backend source",
        "- Fresnel and Binary2Phase parameter names follow the validated component probes.",
        "- This report covers component-level surrogate optimization only.",
        "",
        "## 2. Surrogate PSF construction",
        f"- **psf_requires_grad:** {result.get('psf_requires_grad', '-')}",
        f"- **PSF summary:** `{json.dumps(result.get('psf_summary', {}), ensure_ascii=False, default=str)}`",
        "",
        "## 3. HSI forward model",
        "- Synthetic HSI data only.",
        "- Per-band differentiable convolution and band integration are used.",
        "",
        "## 4. Reconstruction metrics",
        f"- **reconstruction_loss_before:** {metrics.get('reconstruction_loss_before', result.get('reconstruction_loss_before', '-'))}",
        f"- **reconstruction_loss_after:** {metrics.get('reconstruction_loss_after', result.get('reconstruction_loss_after', '-'))}",
        f"- **mse_before / mse_after:** {metrics.get('mse_before', '-')} / {metrics.get('mse_after', '-')}",
        f"- **psnr_before / psnr_after:** {metrics.get('psnr_before', '-')} / {metrics.get('psnr_after', '-')}",
        f"- **sam_before / sam_after:** {metrics.get('sam_before', '-')} / {metrics.get('sam_after', '-')}",
        "",
        "## 5. Gradient flow",
        f"- **loss_requires_grad:** {result.get('loss_requires_grad', '-')}",
        f"- **component_grad_norm_max:** {metrics.get('component_grad_norm_max', result.get('component_grad_norm_max', '-'))}",
        "",
        "## 6. Parameter update",
        f"- **component_parameter_changed:** {result.get('component_parameter_changed', '-')}",
        "",
        "## 7. Claim boundary",
        f"- **Evidence level:** {result.get('evidence_level', '-')}",
        f"- **Claim ceiling:** {result.get('claim_ceiling', '-')}",
        "",
        "## 8. What not to claim",
        "- Do not claim full GeoLens lens-level optimization.",
        "- Do not claim native physical lens optimization.",
        "- Do not claim real HSI performance or real camera validation.",
        "- Do not claim full wave-optics co-design.",
    ]
    return "\n".join(lines) + "\n"


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
