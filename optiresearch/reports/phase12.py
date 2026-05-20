"""Phase 12 public HSI, DeepLens PSF contract, and protocol freeze report."""

from __future__ import annotations

import json
import os
from pathlib import Path

from optiresearch.adapters.deeplens import DeepLensAdapter
from optiresearch.hsi.public_datasets import list_hsi_dataset_adapters


def export_phase12_report() -> Path:
    root = Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "phase12_public_hsi_deeplens_protocol_report.md"
    path.write_text(_markdown(), encoding="utf-8")
    return path


def _latest_public_matrix_summary() -> dict:
    matrix_root = Path(os.getenv("OPTIRESEARCH_HSI_ROOT", "./workspace/hsi")) / "public_matrix"
    summaries = sorted(matrix_root.glob("*/public_hsi_matrix_summary.json"), key=lambda item: item.stat().st_mtime, reverse=True) if matrix_root.exists() else []
    if not summaries:
        return {"status": "not_run", "row_count": 0}
    try:
        return json.loads(summaries[0].read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "read_failed", "error": str(exc)}


def _markdown() -> str:
    adapters = list_hsi_dataset_adapters()
    deeplens = DeepLensAdapter().validate_environment()
    public_summary = _latest_public_matrix_summary()
    protocol_path = Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports")) / "paper_experiment_protocol_v0.1_freeze.md"
    lines = [
        "# Phase 12: Public HSI, DeepLens PSF Contract, and Protocol Freeze",
        "",
        "## 1. Objective",
        "",
        "Connect local/public HSI dataset paths, make DeepLens PSF wavelength scope explicit, and freeze the paper experiment protocol.",
        "",
        "## 2. Current system maturity",
        "",
        "The system supports synthetic HSI, local-path HSI preparation, mock DeepLens, DeepLens smoke/proxy paths, ClaimEvidence, matrix evaluation, and DesignRule memory.",
        "",
        "## 3. Dataset adapter status",
        "",
        "| Dataset | Available | Download policy |",
        "|---|---:|---|",
    ]
    for dataset_id, item in adapters.items():
        lines.append(f"| {dataset_id} | {item['available']} | {item['download_policy']} |")
    lines.extend(
        [
            "",
            "## 4. Local/public dataset preparation",
            "",
            "Local NPZ supports split files, dataset.npz with split labels, and single cube files patched into train/val/test.",
            "CAVE and ICVL are local-path adapters only. No automatic download is performed.",
            "",
            "## 5. DeepLens wavelength-aware PSF contract",
            "",
            f"DeepLens available: `{deeplens.get('available')}`.",
            "The contract records wavelength_aware_psf, wavelengths_nm, wavelength_count, psf_band_axis, depth_count, psf_cube_shape, wavelength_sampling_method, hsi_forward_compatible, and native_wavelength_physics.",
            "Adapter-proxy wavelength-aware PSF means the cube has a wavelength axis; it does not prove native wavelength physics.",
            "",
            "## 6. Public HSI matrix status",
            "",
            f"Latest summary: `{json.dumps(public_summary, ensure_ascii=False, sort_keys=True)}`",
            "",
            "## 7. Evidence levels and claim boundaries",
            "",
            "Evidence levels include mock, deeplens_smoke, deeplens_adapter_proxy, deeplens_semi_native, synthetic_hsi, public_hsi_mock, public_hsi_deeplens_proxy, public_hsi_deeplens_semi_native, native_optimized, and real_lab.",
            "",
            "## 8. Frozen paper experiment protocol",
            "",
            f"Frozen path: `{protocol_path}`",
            "",
            "## 9. What is validated",
            "",
            "- Local/public dataset ingestion contracts.",
            "- Structured skips when data or DeepLens is unavailable.",
            "- Wavelength-aware PSF metadata contract.",
            "- Public matrix evidence boundaries.",
            "",
            "## 10. What is not validated",
            "",
            "- Real HSI performance without local public data and full matrix evidence.",
            "- Real camera HSI behavior without lab data.",
            "- Native DeepLens optical validation when realization is adapter_proxy or semi_native.",
            "",
            "## 11. Remaining path to native optimization / real lab validation",
            "",
            "Phase 13 should run native optimization, real dataset experiments, and final paper-ready benchmarks.",
        ]
    )
    return "\n".join(lines)
