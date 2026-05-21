"""Phase 19B report: corrected DeepLens native optimization path discovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def export_phase19b_report() -> Path:
    root = Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
    root.mkdir(parents=True, exist_ok=True)
    summary = _summary()
    (root / "phase19b_deeplens_optimization_path_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    path = root / "phase19b_deeplens_optimization_path_report.md"
    path.write_text("\n".join(_markdown(summary)), encoding="utf-8")
    return path


def _summary() -> dict[str, Any]:
    scan = _load_json(Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports")) / "deeplens_optimization_path_scan.json")
    surface_results = _load_probe_results("surface_probe_*")
    lensfile_results = _load_probe_results("lensfile_probe_*")
    remote_results = _load_remote_results()
    return {
        "scan": scan,
        "surface_results": surface_results,
        "lensfile_results": lensfile_results,
        "remote_results": remote_results,
        "claim_decision": _claim_decision(surface_results, lensfile_results),
    }


def _load_probe_results(pattern: str) -> list[dict[str, Any]]:
    root = Path("workspace/native_optimization")
    results: list[dict[str, Any]] = []
    if not root.exists():
        return results
    for path in sorted(root.glob(f"{pattern}/probe_result.json")):
        payload = _load_json(path)
        if payload:
            results.append(payload)
    return results


def _load_remote_results() -> list[dict[str, Any]]:
    root = Path("workspace/remote_jobs")
    results: list[dict[str, Any]] = []
    if not root.exists():
        return results
    for path in sorted(root.glob("*/remote_job_result.json")):
        payload = _load_json(path)
        metrics = payload.get("metrics_summary", {})
        if metrics.get("job_type") in {"deeplens_surface_optimization_probe", "deeplens_lensfile_optimization_probe"}:
            results.append(payload)
    return results


def _claim_decision(surface_results: list[dict[str, Any]], lensfile_results: list[dict[str, Any]]) -> dict[str, Any]:
    component_supported = any(
        item.get("status") == "succeeded"
        and item.get("differentiable") is True
        and _positive(item.get("gradient_norm"))
        and item.get("parameters_changed") is True
        and item.get("metadata", {}).get("optimizer_step_executed") is True
        for item in surface_results
    )
    lens_supported = any(
        item.get("status") == "succeeded"
        and item.get("differentiable") is True
        and _positive(item.get("gradient_norm"))
        and item.get("parameters_changed") is True
        for item in lensfile_results
    )
    return {
        "component_level_native_differentiable_optimization": "supported" if component_supported else "not_supported_yet",
        "lens_level_native_differentiable_optimization": "supported" if lens_supported else "not_supported_yet",
        "optical_hsi_native_differentiable_codesign": "not_supported_yet",
        "supported_claims": [
            "DeepLens native differentiable component optimization is supported"
        ]
        if component_supported
        else [],
        "unsupported_claims": [
            "DeepLens native optical-HSI co-design is supported"
        ],
    }


def _markdown(summary: dict[str, Any]) -> list[str]:
    scan = summary.get("scan") or {}
    surface_results = summary.get("surface_results", [])
    lensfile_results = summary.get("lensfile_results", [])
    remote_results = summary.get("remote_results", [])
    decision = summary.get("claim_decision", {})
    lines = [
        "# Phase 19B: Correct DeepLens Native Optimization Path Discovery",
        "",
        "## Optimization Path Scan",
        "",
        f"- Available: {scan.get('available')}",
        f"- Entries: {scan.get('summary', {}).get('entry_count', 0)}",
        f"- Surface candidates: {scan.get('summary', {}).get('surface_candidates', 0)}",
        f"- Lens-file candidates: {scan.get('summary', {}).get('lens_file_candidates', 0)}",
        "",
        "## Surface Probe Results",
        "",
        _surface_table(surface_results),
        "",
        "## Lens-file Probe Results",
        "",
        _lensfile_table(lensfile_results),
        "",
        "## Remote Probe Results",
        "",
        _remote_table(remote_results),
        "",
        "## ClaimEvidence Boundary",
        "",
        "| Claim Level | Status | Requirement |",
        "|---|---|---|",
        f"| Component-level native differentiable optimization | {decision.get('component_level_native_differentiable_optimization')} | surface probe success + requires_grad + backward + optimizer.step + parameter change |",
        f"| Lens-level native differentiable optimization | {decision.get('lens_level_native_differentiable_optimization')} | lens file loaded + lens PSF/image loss backward + parameter change |",
        f"| Native optical-HSI co-design | {decision.get('optical_hsi_native_differentiable_codesign')} | lens optimization + DeepLens PSF/image feeds HSI loss + HSI loss reaches optical parameter |",
        "",
        "## Supported Claim",
        "",
    ]
    for claim in decision.get("supported_claims", []) or ["None yet"]:
        lines.append(f"- {claim}")
    lines.extend(["", "## Claims Still Unsupported", ""])
    for claim in decision.get("unsupported_claims", []) or ["None"]:
        lines.append(f"- {claim}")
    return lines


def _surface_table(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No surface probe results found."
    lines = [
        "| Surface | Status | Differentiable | Grad Norm | Param Changed | Loss Before | Loss After |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in results:
        lines.append(
            f"| {item.get('surface_class')} | {item.get('status')} | {_yn(item.get('differentiable'))} | {item.get('gradient_norm')} | {_yn(item.get('parameters_changed'))} | {item.get('loss_before')} | {item.get('loss_after')} |"
        )
    return "\n".join(lines)


def _lensfile_table(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No lens-file probe results found."
    lines = [
        "| Lens Class | Status | File | Differentiable | Grad Norm | Param Changed |",
        "|---|---|---|---|---|---|",
    ]
    for item in results:
        lines.append(
            f"| {item.get('lens_class')} | {item.get('status')} | {item.get('successful_file') or '-'} | {_yn(item.get('differentiable'))} | {item.get('gradient_norm')} | {_yn(item.get('parameters_changed'))} |"
        )
    return "\n".join(lines)


def _remote_table(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No remote Phase 19B probe results found."
    lines = [
        "| Job ID | Status | Job Type | Differentiable | Grad Norm |",
        "|---|---|---|---|---|",
    ]
    for item in results:
        metrics = item.get("metrics_summary", {})
        lines.append(
            f"| {item.get('job_id')} | {item.get('status')} | {metrics.get('job_type')} | {_yn(metrics.get('differentiable'))} | {metrics.get('gradient_norm')} |"
        )
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and float(value) > 0.0


def _yn(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "-"
