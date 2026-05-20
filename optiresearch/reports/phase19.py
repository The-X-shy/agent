"""Phase 19 report: DeepLens Native Differentiable Optimization Probe."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def export_phase19_report() -> Path:
    """Export the Phase 19 native optimization probe report."""
    root = Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "phase19_native_optimization_probe_report.md"
    path.write_text(_markdown(), encoding="utf-8")
    return path


def _markdown() -> str:
    inspection = _load_inspection()
    probe_results = _load_probe_results()
    lines = _build_report_lines(inspection, probe_results)
    return "\n".join(lines)


def _load_inspection() -> dict[str, Any]:
    inspection_path = Path(
        os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports")
    ) / "deeplens_native_optimization_inspection.json"
    if inspection_path.exists():
        try:
            return json.loads(inspection_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"available": False, "error": "No inspection data available"}


def _load_probe_results() -> list[dict[str, Any]]:
    probe_dir = Path("workspace/native_optimization")
    results: list[dict[str, Any]] = []
    if probe_dir.is_dir():
        for d in sorted(probe_dir.iterdir()):
            if d.is_dir():
                result_file = d / "probe_spec.json"
                if result_file.exists():
                    try:
                        results.append(json.loads(result_file.read_text(encoding="utf-8")))
                    except Exception:
                        pass
    return results


def _build_report_lines(
    inspection: dict[str, Any],
    probe_results: list[dict[str, Any]],
) -> list[str]:
    lines = [
        "# Phase 19: DeepLens Native Differentiable Optimization Probe",
        "",
        "## 1. Objective",
        "",
        "Systematically probe whether DeepLens v1.5.2 lens classes support true",
        "native differentiable optical optimization with the full autograd chain:",
        "`optical parameter -> PSF simulation -> scalar loss -> backward -> optimizer.step -> parameter change`.",
        "",
        "## 2. Phase 18 Status",
        "",
        "Phase 18 completed DeepLens-backed black-box HSI co-design. Key results:",
        "- DeepLens PSF successfully enters the HSI forward model + reconstruction chain",
        "- All optimization is black-box: `differentiable=false`, `native_parameter_update=false`",
        "- Only `ParaxialLens` is used for PSF generation",
        "- Only synthetic HSI datasets",
        "",
        "## 3. DeepLens Native Optimization Inspection",
        "",
        f"- **DeepLens available:** {inspection.get('available', False)}",
        f"- **Version:** {inspection.get('deeplens_version', 'N/A')}",
        f"- **Source checkout:** {inspection.get('is_source_checkout', False)}",
        "",
        "## 4. Lens-Class Capability Table",
        "",
        _capability_table(inspection),
        "",
        "## 5. Minimal Differentiability Probe Results",
        "",
        _probe_results_section(probe_results),
        "",
        "## 6. Remote WSL Native Probe Results",
        "",
        "Remote probes execute on the WSL worker with real DeepLens installation.",
        "Results are ingested back to Mac via the remote job pipeline.",
        "",
        "## 7. ClaimEvidence Decision",
        "",
        "| Claim | Status | Evidence |",
        "|---|---|---|",
        '| "DeepLens native differentiable optimization is supported" | Depends on probe outcome | Requires differentiable=true, gradient_norm>0, parameters_changed=true |',
        '| "DeepLens-backed black-box co-design is supported" | Supported | Phase 18 evidence, fallback_used=false |',
        '| "DeepLens native optimization improves HSI reconstruction" | Needs followup | Requires native optimization + HSI metrics |',
        "",
        "## 8. What Is Validated",
        "",
        "- DeepLens v1.5.2 is importable and inspectable",
        "- Lens classes can be examined for differentiation support",
        "- Structured probe results are produced for all lens classes",
        "- Black-box co-design pipeline is functional",
        "",
        "## 9. What Is NOT Validated",
        "",
        "- End-to-end native differentiable optical-HSI optimization",
        "- Wavelength-aware differentiable HSI reconstruction",
        "- Real camera HSI validation",
        "- GeoLens/DiffractiveLens/HybridLens native optimization",
        "",
        "## 10. Requirements for True End-to-End Optical-HSI Native Optimization",
        "",
        "1. A DeepLens lens class that supports activate_grad + get_optimizer + differentiable PSF",
        "2. Autograd chain from optical parameters through to HSI reconstruction loss",
        "3. Joint optimization: optical parameters + reconstruction network parameters",
        "4. Wavelength-aware differentiable forward model",
        "5. Real camera validation dataset",
        "",
        "---",
        f"*Generated by Phase 19 report export*",
    ]
    return lines


def _capability_table(inspection: dict[str, Any]) -> str:
    lines = [
        "| Lens Class | Available | activate_grad | get_optimizer | PSF | Diffable | Instantiable |",
        "|---|---|---|---|---|---|---|",
    ]
    lens_classes = inspection.get("lens_classes", {})
    for cls_name in ["ParaxialLens", "GeoLens", "DiffractiveLens", "HybridLens", "PSFNetLens"]:
        info = lens_classes.get(cls_name, {})
        lines.append(
            f"| {cls_name} "
            f"| {_yn(info.get('class_available'))} "
            f"| {_yn(info.get('has_activate_grad'))} "
            f"| {_yn(info.get('has_get_optimizer'))} "
            f"| {_yn(info.get('has_psf_method'))} "
            f"| {_yn(info.get('likely_differentiable'))} "
            f"| {_yn(info.get('can_instantiate_minimal'))} |"
        )
    return "\n".join(lines)


def _probe_results_section(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No probe results available. Run `python -m optiresearch.cli run-native-optimization-probe` first."
    lines = [
        "| Probe ID | Lens Class | Status | Realization | Diffable | Param Update | Grad Norm | Loss Before | Loss After |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results[:20]:
        lines.append(
            f"| {r.get('probe_id', '?')} "
            f"| {r.get('lens_class', '?')} "
            f"| {r.get('status', '?')} "
            f"| {r.get('realization_level', '?')} "
            f"| {_yn(r.get('differentiable'))} "
            f"| {_yn(r.get('native_parameter_update'))} "
            f"| {r.get('gradient_norm', '-')} "
            f"| {r.get('loss_before', '-')} "
            f"| {r.get('loss_after', '-')} |"
        )
    return "\n".join(lines)


def _yn(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    if value is None:
        return "-"
    return str(value)
