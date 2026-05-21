"""Phase 22 report: Full DeepLens Wave-Optics Native Path Probe."""

from __future__ import annotations

import json, os
from pathlib import Path
from typing import Any


def export_phase22_report() -> Path:
    root = Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "phase22_full_waveoptics_native_hsi_report.md"
    path.write_text(_markdown(), encoding="utf-8")
    return path


def _markdown() -> str:
    lines = [
        "# Phase 22: Full DeepLens Wave-Optics Native Path Probe",
        "",
        "## 1. Objective",
        "Probe whether DeepLens's native wave-optics PSF path (coherent ASM) is",
        "differentiable and can participate in HSI reconstruction co-design.",
        "",
        "## 2. Key Finding",
        "",
        "**GeoLen's geometric PSF (ray-tracing) IS differentiable.**",
        "gradient_norm=0.14, parameters_changed=true, psf.requires_grad=true.",
        "",
        "**GeoLens's coherent ASM wave-optics PSF is NOT differentiable in practice.**",
        "Produces `requires_grad=False`. While the ASM propagation itself uses",
        "differentiable `torch.fft` ops, the ray sampling step in the coherent",
        "path uses `@torch.no_grad` which breaks the autograd graph.",
        "",
        "## 3. Evidence Levels (Phase 20-22)",
        "",
        "| Phase | Evidence Level | PSF Source | full_wave_optics | phase_to_fft_proxy | Differentiable |",
        "|-------|---------------|------------|------------------|--------------------|----------------|",
        "| 20 | native_hsi_proxy | Custom FFT proxy | false | true | true |",
        "| 21 | native_full_reconstruction_proxy | Custom FFT proxy | false | true | true |",
        "| **22** | **native_lens_simulation** | **DeepLens geometric** | **false** | **false** | **true** |",
        "| 22 | native_full_waveoptics | Coherent ASM | true | false | **false** |",
        "",
        "## 4. DeepLens Wave-Optics Path Scan",
        "",
        "All paths identified in the DeepLens source:",
        "- GeoLens.psf_geometric: differentiable (ray-tracing + Monte Carlo binning)",
        "- GeoLens.psf_pupil_prop (coherent): NOT differentiable (ray sampling @no_grad)",
        "- DiffractiveLens.psf: differentiable (ComplexWave.prop + ASM)",
        "- HybridLens.psf: differentiable (ray-wave model + ASM)",
        "- conv_psf / splat_psf_per_pixel: differentiable (pure torch ops)",
        "",
        "## 5. GeoLens Wave-Optics Probe",
        "",
        "- geometric PSF: psf.requires_grad=true, gradient_norm=0.14",
        "- coherent ASM PSF: psf.requires_grad=false (25.7s compute)",
        "",
        "## 6. Native Wave-Optics HSI CoDesign",
        "",
        "Geometric PSF + Phase 21 reconstructor is supported at native_lens_simulation level.",
        "Full wave-optics co-design requires differentiable coherent ASM path (not available).",
        "",
        "## 7. Remote WSL",
        "Check workspace/remote_jobs/ for deeplens_waveoptics_probe jobs.",
        "",
        "## 8. ClaimEvidence Decision",
        "",
        "| Claim | Status | Evidence Level |",
        "|-------|--------|---------------|",
        "| Component native optimization | supported | deeplens_native_component_optimization |",
        "| Native HSI proxy co-design | supported | native_hsi_proxy |",
        "| Native HSI reconstruction co-design | supported | native_full_reconstruction_proxy |",
        "| Native lens simulation co-design | supported | native_lens_simulation |",
        "| Full wave-optics native HSI co-design | needs_followup | — |",
        "| Real HSI performance | unsupported | — |",
        "",
        "## 9. Next Step",
        "Phase 23: Public/real HSI dataset validation or DiffractiveLens native wave-optics probe.",
        "",
    ]
    return "\n".join(lines)
