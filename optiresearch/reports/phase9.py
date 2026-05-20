"""Phase 9 HSI reconstruction report."""

from __future__ import annotations

import json
import os
from pathlib import Path

from optiresearch.schemas.hsi import (
    build_default_hsi_forward_model_spec,
    build_default_hsi_reconstruction_spec,
    build_default_synthetic_hsi_dataset_spec,
)


def export_phase9_report() -> Path:
    root = Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "phase9_hsi_reconstruction_report.md"
    path.write_text(_markdown(), encoding="utf-8")
    return path


def _load_baseline() -> dict:
    root = Path(os.getenv("OPTIRESEARCH_HSI_BASELINE_ROOT", "./workspace/hsi/baselines")) / "mock_deeplens" / "hsi_baseline_comparison.json"
    if root.exists():
        return json.loads(root.read_text(encoding="utf-8"))
    return {"runs": [], "warning": "No HSI baseline report found."}


def _markdown() -> str:
    dataset = build_default_synthetic_hsi_dataset_spec()
    forward = build_default_hsi_forward_model_spec()
    reconstruction = build_default_hsi_reconstruction_spec()
    baseline = _load_baseline()
    lines = [
        "# Phase 9 HSI Reconstruction Report",
        "",
        "## Objective",
        "",
        "Establish wavelength-aware synthetic HSI reconstruction evaluation for optical encoders.",
        "",
        "## HSI dataset spec",
        "",
        f"Synthetic dataset: `{dataset.spectral_bands}` bands, `{dataset.height}x{dataset.width}`, train/val/test `{dataset.train_size}/{dataset.val_size}/{dataset.test_size}`.",
        "",
        "## Forward model",
        "",
        f"Measurement type: `{forward.measurement_type}`. PSF cube is applied per wavelength band and integrated into a single-shot measurement.",
        "",
        "## Reconstruction baseline",
        "",
        f"Network: `{reconstruction.network_type}`. This is a linear baseline, not a final reconstruction network.",
        "",
        "## Optical encoder baseline",
        "",
        "Optical PSF artifacts are produced by mock or DeepLens backends and reused by the HSI forward model.",
        "",
        "## HSI reconstruction baseline",
        "",
        "| Encoder | PSNR | SAM | ERGAS | Evidence Level |",
        "|---|---:|---:|---:|---|",
    ]
    for item in baseline.get("runs", []):
        lines.append(f"| {item['encoder_type']} | {item['PSNR']} | {item['SAM']} | {item['ERGAS']} | {item['evidence_level']} |")
    if not baseline.get("runs"):
        lines.append("| missing |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Evidence level",
            "",
            "HSI reconstruction evidence is separate from optical-only evidence and requires reconstruction metrics.",
            "",
            "## What is validated",
            "",
            "- Synthetic HSI dataset generation.",
            "- PSF-based forward measurement rendering.",
            "- Linear reconstruction baseline execution.",
            "- Artifact-backed reconstruction metrics.",
            "",
            "## What is not validated",
            "",
            "- Real HSI dataset performance.",
            "- Native DeepLens HSI optical performance.",
            "- Final optimized reconstruction quality.",
            "",
            "## Next steps toward native optimized HSI",
            "",
            "1. Add real or public HSI datasets.",
            "2. Replace linear baseline with stronger reconstruction networks.",
            "3. Bind native DeepLens optimization and wavelength-aware PSF generation.",
            "",
        ]
    )
    return "\n".join(lines)
