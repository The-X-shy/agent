"""Phase 11 HSI dataset/reconstructor matrix report."""

from __future__ import annotations

import json
import os
from pathlib import Path

from optiresearch.hsi.public_datasets import list_hsi_dataset_adapters


def export_phase11_report() -> Path:
    root = Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "phase11_hsi_network_dataset_report.md"
    path.write_text(_markdown(), encoding="utf-8")
    return path


def _latest_matrix_summary() -> dict:
    matrix_root = Path(os.getenv("OPTIRESEARCH_HSI_ROOT", "./workspace/hsi")) / "matrix"
    summaries = sorted(matrix_root.glob("*/hsi_matrix_summary.json"), key=lambda item: item.stat().st_mtime, reverse=True) if matrix_root.exists() else []
    if not summaries:
        return {"warning": "No HSI matrix summary found.", "best_by_reconstructor": {}}
    try:
        return json.loads(summaries[0].read_text(encoding="utf-8"))
    except Exception as exc:
        return {"warning": f"Failed to read matrix summary: {exc}", "best_by_reconstructor": {}}


def _markdown() -> str:
    adapters = list_hsi_dataset_adapters()
    summary = _latest_matrix_summary()
    lines = [
        "# Phase 11: HSI Dataset and Reconstructor Matrix",
        "",
        "## 1. Objective",
        "",
        "Validate whether optical-sensitive HSI encoder ranking holds when the dataset source and reconstruction network are made explicit.",
        "",
        "## 2. Phase 10 limitation",
        "",
        "Phase 10 used synthetic HSI data and a simple optical-conditioned linear reconstructor. Achromatic ranking first may reflect reconstructor preference for uniform PSF rather than a general optical conclusion.",
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
            "Public datasets are local-path only. The project does not download CAVE or ICVL automatically.",
            "",
            "## 4. Reconstructor matrix",
            "",
            f"Matrix ID: `{summary.get('matrix_id', 'N/A')}`",
            f"Rows: `{summary.get('row_count', 0)}`. Succeeded: `{summary.get('succeeded', 0)}`. Skipped: `{summary.get('skipped', 0)}`.",
            "",
            "## 5. Encoder ranking by reconstructor",
            "",
            "| Reconstructor | Best encoder | Score | Status |",
            "|---|---|---:|---|",
        ]
    )
    best = summary.get("best_by_reconstructor", {})
    if best:
        for reconstructor, item in best.items():
            lines.append(f"| {reconstructor} | {item.get('encoder')} | {item.get('score')} | {item.get('status', 'available')} |")
    else:
        lines.append("| (no matrix yet) |  |  |  |")
    lines.extend(
        [
            "",
            "## 6. Does chromatic coding benefit from stronger networks?",
            "",
            "The answer is only supported when a matrix contains an available stronger reconstructor row. If TinyCNN is skipped because Torch is unavailable, this remains follow-up work.",
            "",
            "## 7. Matrix-level ClaimEvidence",
            "",
            "Claims must state dataset, backend, reconstructor, and realization level. ClaimEvidence remains the final gate; LLM output cannot decide final claim status.",
            "",
            "## 8. DesignRule updates",
            "",
            "Design rules compiled from the matrix keep their scope and caveats, including synthetic/public dataset status and mock/proxy/native backend status.",
            "",
            "## 9. What is validated",
            "",
            "- Dataset adapters can prepare synthetic and local-path HSI splits.",
            "- Matrix rows distinguish dataset, backend, encoder, reconstructor, and forward mode.",
            "- Optional Torch reconstructors are skipped cleanly when unavailable.",
            "",
            "## 10. What is not validated",
            "",
            "- Synthetic results are not real HSI performance.",
            "- Public/local dataset with mock optical encoder is not real camera validation.",
            "- DeepLens adapter_proxy is not native physical validation.",
            "- TinyCNN/UNet results, when available, are small optional baselines, not final paper-scale networks.",
            "",
            "## 11. Next phase: real dataset + native DeepLens optimization",
            "",
            "Phase 12 should freeze a real HSI dataset split, add wavelength-aware DeepLens PSF generation, run native optimization, and lock paper experiments.",
        ]
    )
    return "\n".join(lines)
