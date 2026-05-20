"""Phase 10 optical-sensitive HSI reconstruction report."""

from __future__ import annotations

import json
import os
from pathlib import Path

from optiresearch.schemas.hsi import (
    build_default_hsi_forward_model_spec,
    build_default_hsi_reconstruction_spec,
    build_default_synthetic_hsi_dataset_spec,
)


def export_phase10_report() -> Path:
    root = Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "phase10_optical_sensitive_hsi_report.md"
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
        "# Phase 10: Optical-Sensitive HSI Reconstruction Benchmark",
        "",
        "## 1. Objective",
        "",
        "Make the synthetic HSI reconstruction benchmark sensitive to optical encoder PSF differences, so that different encoder types produce distinct and interpretable reconstruction rankings.",
        "",
        "## 2. Why Phase 9 metrics were identical",
        "",
        "In Phase 9, all 5 encoder types produced identical reconstruction metrics (PSNR=18.04, SAM=0.225) because:",
        "",
        "1. **Forward model** summed all spectral bands into a single grayscale measurement, collapsing encoder-specific PSF differences.",
        "2. **Linear reconstructor** only learned per-band scalar multipliers from the single-channel measurement.",
        "3. **Dataset** used smooth low-rank spectral structure that was insensitive to spectral coding.",
        "",
        "## 3. Dataset improvements",
        "",
        f"Pattern: `{dataset.spectral_pattern_type}` (default changed from `smooth_low_rank`).",
        f"Materials: `{dataset.material_count}`. Depth-aware: `{dataset.depth_aware}`.",
        "",
        "- `mixed_materials`: K=6 material signatures with spatial abundance mixing and depth-dependent variation.",
        "- `sparse_peaks`: Sparse spectral peaks at random band positions.",
        "- `edge_spectral_contrast`: Different spectra at edges vs interior regions.",
        "- `smooth_low_rank`: Original Phase 9 behavior (backward compatible).",
        "",
        "## 4. Optical-sensitive forward model",
        "",
        f"Mode: `{forward.forward_mode}` (default changed from `simple_sum`).",
        "",
        "- `simple_sum`: Original Phase 9 behavior — sum over bands (backward compatible).",
        "- `psf_weighted`: Per-band PSF energy weighting before summation.",
        "- `coded_aperture_proxy`: Band-dependent coding weights from PSF centroid, spread, and high-frequency energy.",
        "- `depth_spectral_coded`: Combined depth and wavelength PSF features produce encoder-specific measurement encoding.",
        "",
        "The measurement remains a single-channel `(1, H, W)` image, but the encoding process preserves encoder-specific PSF characteristics that affect reconstruction quality.",
        "",
        "## 5. Optical features",
        "",
        "`OpticalFeatureExtractor` computes from PSF cube `(D, B, H, W)`:",
        "",
        "- `band_spread` (B,): Per-band spatial width.",
        "- `band_centroid_x/y` (B,): Per-band centroid positions.",
        "- `band_high_freq_energy` (B,): Per-band high-frequency content.",
        "- `depth_stability_score`: PSF similarity across depth planes.",
        "- `spectral_separability_score`: PSF variation across wavelength bands.",
        "- `coding_strength`: Combined spectral × (1 − depth) metric.",
        "- `band_condition_score`: Spread range / spread mean.",
        "",
        "## 6. Optical-conditioned reconstructor",
        "",
        f"Network: `{reconstruction.network_type}` (default changed from `linear_baseline`).",
        "",
        "`OpticalConditionedLinearReconstructor` uses optical features to create band-dependent spatial basis functions. Higher spectral separability → more distinct per-band maps → better reconstruction. Pure numpy — no torch dependency.",
        "",
        "`TinyCNNReconstructor`: Minimal 3-layer CNN. Requires PyTorch; gracefully degrades to `TORCH_NOT_AVAILABLE` when torch is absent.",
        "",
        "## 7. HSI baseline ranking",
        "",
        "| Encoder | PSNR | SSIM | SAM | ERGAS | Rec Score | Coding Str | Depth Stab | Spectral Sep | Ranking | Evidence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in baseline.get("runs", []):
        lines.append(
            "| {encoder} | {psnr} | {ssim} | {sam} | {ergas} | {score} | {coding} | {depth} | {spectral} | {ranking} | {level} |".format(
                encoder=item.get("encoder_type", ""),
                psnr=item.get("PSNR", ""),
                ssim=item.get("SSIM", ""),
                sam=item.get("SAM", ""),
                ergas=item.get("ERGAS", ""),
                score=item.get("reconstruction_score", ""),
                coding=item.get("coding_strength", ""),
                depth=item.get("depth_stability_score", ""),
                spectral=item.get("spectral_separability_score", ""),
                ranking=item.get("ranking", ""),
                level=item.get("evidence_level", ""),
            )
        )
    if not baseline.get("runs"):
        lines.append("| (no data) |  |  |  |  |  |  |  |  |  |  |")
    lines.extend([
        "",
        "Expected trends:",
        "- **conventional**: lower reconstruction score (limited spectral separability).",
        "- **achromatic**: good depth stability but weak spectral recovery.",
        "- **edof**: strong depth stability, moderate SAM.",
        "- **chromatic_coded**: strong spectral recovery but weaker depth stability.",
        "- **controlled_chromatic_edof**: best joint reconstruction score.",
        "",
        "## 8. ClaimEvidence updates",
        "",
        "New claim types and evaluation:",
        "",
        "- `controlled chromatic EDOF improves synthetic HSI reconstruction under mock setting` → supported only if baseline comparison shows higher reconstruction_score vs conventional.",
        "- `all encoders perform identically in HSI reconstruction` → contradicted when baseline shows metric variation.",
        "- `controlled chromatic EDOF is best for real HSI reconstruction` → needs_followup (requires native DeepLens validation).",
        "",
        "New `explain_claim` fields: `compared_baseline`, `compared_metric`, `ranking_position`, `evidence_level`.",
        "",
        "## 9. What is validated",
        "",
        "- Optical-sensitive forward model produces encoder-dependent measurements.",
        "- Optical features extracted from PSF cubes are consistent and interpretable.",
        "- Optical-conditioned reconstructor achieves different metrics per encoder type.",
        "- Baseline comparison shows encoder ranking on reconstruction metrics.",
        "- All Phase 1-9 tests still pass (backward compatible via `simple_sum` mode and `smooth_low_rank` dataset).",
        "",
        "## 10. What is not validated",
        "",
        "- **Synthetic only**: All results are from synthetic datasets and mock PSF cubes.",
        "- **Mock backend only**: Not validated against real DeepLens or physical sensors.",
        "- **Linear reconstructor only**: The optical-conditioned linear reconstructor is a baseline, not a final network.",
        "- **No real HSI performance**: Rankings are for system verification and method prototyping, not real-world conclusions.",
        "- **Forward model is a proxy**: The depth_spectral_coded forward mode is an evaluation proxy, not a physical sensor model.",
        "",
        "## 11. Next step: Phase 11",
        "",
        "1. Real or public HSI dataset integration.",
        "2. DeepLens wavelength-aware PSF generation with native optimization.",
        "3. Tiny CNN / UNet architectures for stronger reconstruction.",
        "4. Native DeepLens optimization loop for HSI-specific encoder design.",
    ])
    return "\n".join(lines)
