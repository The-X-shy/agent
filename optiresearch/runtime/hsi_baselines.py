"""HSI reconstruction baseline batches with optical feature comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from optiresearch.runtime.baselines import ENCODER_TYPES
from optiresearch.runtime.hsi_pipeline import run_hsi_reconstruction_flow


def run_hsi_encoder_baselines(
    backend: str = "mock_deeplens",
    encoders: list[str] | None = None,
    objective: str = "Evaluate synthetic HSI reconstruction",
    workspace_id: str = "default",
    forward_mode: str = "depth_spectral_coded",
    reconstructor_type: str = "optical_conditioned_linear",
    dataset_pattern: str = "mixed_materials",
) -> dict[str, Any]:
    encoders = encoders or ENCODER_TYPES
    runs = []
    for encoder in encoders:
        result = run_hsi_reconstruction_flow(
            f"{objective} [{encoder}]",
            backend=backend,
            encoder_type=encoder,
            workspace_id=workspace_id,
            forward_mode=forward_mode,
            reconstructor_type=reconstructor_type,
            dataset_pattern=dataset_pattern,
        )
        metrics = result["metrics"]
        optical = result.get("optical_features", {})
        runs.append(
            {
                "encoder_type": encoder,
                "run_id": result["run_id"],
                "PSNR": metrics.get("PSNR"),
                "SSIM": metrics.get("SSIM"),
                "SAM": metrics.get("SAM"),
                "ERGAS": metrics.get("ERGAS"),
                "worst_depth_SAM": metrics.get("worst_depth_SAM"),
                "joint_optical_score": result["run_memory"]["best_metrics"].get("joint_score"),
                "reconstruction_score": _score(metrics),
                "coding_strength": optical.get("coding_strength", metrics.get("optical_coding_strength")),
                "depth_stability_score": optical.get("depth_stability_score", metrics.get("optical_depth_stability_score")),
                "spectral_separability_score": optical.get("spectral_separability_score", metrics.get("optical_spectral_separability_score")),
                "evidence_level": result["evidence_level"],
            }
        )
    for idx, item in enumerate(sorted(runs, key=lambda r: -(r["reconstruction_score"]))):
        item["ranking"] = idx + 1
    best = max(runs, key=lambda item: item["reconstruction_score"])
    report = {"backend": backend, "objective": objective, "forward_mode": forward_mode, "reconstructor_type": reconstructor_type, "dataset_pattern": dataset_pattern, "runs": runs, "best_reconstruction": best}
    _write(report)
    return report


def _score(metrics: dict[str, Any]) -> float:
    psnr = float(metrics.get("PSNR", 0.0))
    sam = float(metrics.get("SAM", 1.0))
    ergas = float(metrics.get("ERGAS", 100.0))
    return round(psnr - 5.0 * sam - 0.02 * ergas, 6)


def _write(report: dict[str, Any]) -> None:
    root = Path(os.getenv("OPTIRESEARCH_HSI_BASELINE_ROOT", "./workspace/hsi/baselines")) / report["backend"]
    root.mkdir(parents=True, exist_ok=True)
    (root / "hsi_baseline_comparison.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    lines = [
        "# HSI Baseline Comparison",
        "",
        f"Backend: {report['backend']}",
        f"Forward mode: {report.get('forward_mode', 'N/A')}",
        f"Reconstructor: {report.get('reconstructor_type', 'N/A')}",
        f"Dataset pattern: {report.get('dataset_pattern', 'N/A')}",
        "",
        "| Encoder | PSNR | SSIM | SAM | ERGAS | Worst-depth SAM | Rec Score | Coding Str | Depth Stab | Spectral Sep | Ranking | Evidence Level |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["runs"]:
        lines.append(
            "| {encoder} | {psnr} | {ssim} | {sam} | {ergas} | {worst} | {score} | {coding} | {depth} | {spectral} | {ranking} | {level} |".format(
                encoder=item["encoder_type"],
                psnr=item["PSNR"],
                ssim=item["SSIM"],
                sam=item["SAM"],
                ergas=item["ERGAS"],
                worst=item["worst_depth_SAM"],
                score=item["reconstruction_score"],
                coding=item.get("coding_strength", "N/A"),
                depth=item.get("depth_stability_score", "N/A"),
                spectral=item.get("spectral_separability_score", "N/A"),
                ranking=item.get("ranking", "N/A"),
                level=item["evidence_level"],
            )
        )
    lines.append("")
    lines.append(f"Best reconstruction: `{report['best_reconstruction']['encoder_type']}` (score={report['best_reconstruction']['reconstruction_score']}).")
    (root / "hsi_baseline_comparison.md").write_text("\n".join(lines), encoding="utf-8")
