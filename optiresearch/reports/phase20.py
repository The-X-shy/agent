"""Phase 20 report: Native DeepLens Optical-HSI Loss Integration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def export_phase20_report() -> Path:
    root = Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "phase20_native_hsi_codesign_report.md"
    path.write_text(_markdown(), encoding="utf-8")
    return path


def _markdown() -> str:
    return "\n".join(_build_report_lines())


def _build_report_lines() -> list[str]:
    lines = [
        "# Phase 20: Native DeepLens Optical-HSI Loss Integration",
        "",
        "## 1. Objective",
        "",
        "Verify that HSI reconstruction loss can backpropagate through a differentiable PSF proxy",
        "into DeepLens native trainable optical parameters (Fresnel.f0, Binary2Phase orders).",
        "",
        "## 2. Phase 19B Findings",
        "",
        "- Fresnel DOE: native differentiable component optimization confirmed (f0, grad_norm > 0)",
        "- Binary2Phase: native differentiable component optimization confirmed (d, order2..order12)",
        "- GeoLens + cooke.json: native differentiable lens-file optimization confirmed",
        "- Remote WSL Fresnel probe: successful",
        "- Component-level native differentiable optimization = supported",
        "- Lens-level native differentiable optimization = supported for GeoLens with cooke.json",
        "- Optical-HSI native differentiable co-design = NOT supported (as of Phase 19B)",
        "",
        "## 3. Differentiable HSI Proxy Design",
        "",
        "Pure-torch pipeline (no numpy, no detach, no no_grad on the loss path):",
        "",
        "1. `generate_torch_synthetic_hsi()` — synthetic HSI cube [N, B, H, W]",
        "2. `make_measurement_from_psf_torch()` — PSF-weighted HSI measurement [N, 1, H, W]",
        "3. `reconstruct_proxy_torch()` — matched-filter reconstruction proxy [N, B, H, W]",
        "4. `hsi_proxy_loss()` — MSE between recon and target",
        "",
        "Phase-to-PSF differentiable bridge:",
        "`phase_func() -> phase map -> field = exp(1j*phase) -> FFT2 -> |.|^2 -> normalize -> PSF`",
        "",
        "Realization level: `native_component_proxy`",
        "- component_native_grad: true (DeepLens parameter is native)",
        "- full_lens_native_psf: false (using FFT proxy, not full wave propagation)",
        "",
    ]

    lines.extend(_fresnel_result_section())
    lines.extend(_binary2phase_result_section())
    lines.extend(_remote_result_section())
    lines.extend(_claim_decision_section())
    lines.extend(_what_is_validated())
    lines.extend(_what_is_not_validated())
    lines.extend(_requirements_section())

    return lines


def _fresnel_result_section() -> list[str]:
    result = _load_result("Fresnel")
    lines = [
        "## 4. Fresnel Native HSI Proxy Co-Design Result",
        "",
    ]
    if result:
        lines.extend([
            f"- status: {result.get('status')}",
            f"- differentiable: {result.get('differentiable')}",
            f"- hsi_loss_before: {result.get('hsi_loss_before')}",
            f"- hsi_loss_after: {result.get('hsi_loss_after')}",
            f"- gradient_norm: {result.get('gradient_norm')}",
            f"- parameters_changed: {result.get('parameters_changed')}",
            f"- optimizer_step_executed: {result.get('optimizer_step_executed')}",
            f"- evidence_level: {result.get('evidence_level')}",
            f"- caveats: {result.get('caveats')}",
        ])
    else:
        lines.append(
            "No Fresnel result found. Run "
            "`python -m optiresearch.cli run-native-hsi-codesign "
            "--optical-component Fresnel --objective minimize_hsi_proxy_loss` first."
        )
    lines.append("")
    return lines


def _binary2phase_result_section() -> list[str]:
    result = _load_result("Binary2Phase")
    lines = [
        "## 5. Binary2Phase Native HSI Proxy Co-Design Result",
        "",
    ]
    if result:
        lines.extend([
            f"- status: {result.get('status')}",
            f"- differentiable: {result.get('differentiable')}",
            f"- hsi_loss_before: {result.get('hsi_loss_before')}",
            f"- hsi_loss_after: {result.get('hsi_loss_after')}",
            f"- gradient_norm: {result.get('gradient_norm')}",
            f"- parameters_changed: {result.get('parameters_changed')}",
            f"- optimizer_step_executed: {result.get('optimizer_step_executed')}",
            f"- evidence_level: {result.get('evidence_level')}",
        ])
    else:
        lines.append("No Binary2Phase result found.")
    lines.append("")
    return lines


def _remote_result_section() -> list[str]:
    return [
        "## 6. Remote WSL Result",
        "",
        "Check `workspace/remote_jobs/` for the latest `native_hsi_codesign` job.",
        "Run: `python -m optiresearch.cli run-remote-native-hsi-codesign "
        "--worker-id windows_wsl --optical-component Fresnel "
        "--objective minimize_hsi_proxy_loss`",
        "",
    ]


def _claim_decision_section() -> list[str]:
    return [
        "## 7. ClaimEvidence Decision",
        "",
        "| Claim | Status | Evidence Level |",
        "|-------|--------|---------------|",
        "| DeepLens native differentiable component optimization | supported | deeplens_native_component_optimization |",
        "| DeepLens native differentiable optical-HSI proxy co-design | supported/needs_followup | native_hsi_proxy |",
        "| DeepLens full native optical-HSI reconstruction co-design | needs_followup | — |",
        "| DeepLens native optimization improves real HSI performance | unsupported | — |",
        "",
    ]


def _what_is_validated() -> list[str]:
    return [
        "## 8. What Is Validated",
        "",
        "- HSI proxy loss can backpropagate through a differentiable PSF into DeepLens parameters",
        "- Fresnel DOE f0 parameter responds to HSI-driven gradient signals",
        "- Binary2Phase order parameters respond to HSI-driven gradient signals",
        "- The end-to-end autograd chain (optical param -> phase -> PSF -> measurement -> recon -> loss) is intact",
        "",
    ]


def _what_is_not_validated() -> list[str]:
    return [
        "## 9. What Is NOT Validated",
        "",
        "- Full HSI reconstruction loss (linear reconstructor with closed-form solution) backprop",
        "- Real camera or real HSI dataset validation",
        "- Multi-step HSI co-design convergence (only 1-step test)",
        "- GeoLens + cooke.json HSI co-design (not yet implemented)",
        "- The PSF is a phase-to-FFT proxy, not a full wave-optics simulation",
        "",
    ]


def _requirements_section() -> list[str]:
    return [
        "## 10. Requirements for Full Native Optical-HSI Reconstruction Co-Design",
        "",
        "1. A fully differentiable HSI reconstructor (torch-based, not numpy closed-form)",
        "2. Full wave-optics PSF from DeepLens (not FFT proxy from phase map)",
        "3. Real or public HSI dataset for validation",
        "4. Multi-step co-design loop with convergence tracking",
        "5. GeoLens + lens file native PSF in the HSI loop",
        "",
    ]


def _load_result(component: str) -> dict[str, Any] | None:
    base = Path("workspace/native_hsi_codesign")
    if not base.exists():
        return None
    for d in sorted(base.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        spec_file = d / "spec.json"
        if not spec_file.exists():
            continue
        try:
            spec = json.loads(spec_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if spec.get("optical_component") == component:
            result_file = d / "result.json"
            if result_file.exists():
                return json.loads(result_file.read_text(encoding="utf-8"))
    return None
