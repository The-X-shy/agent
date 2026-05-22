"""Diagnose Phase 22 native lens HSI co-design instability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def diagnose_native_lens_hsi_codesign(run_dir: str) -> dict[str, Any]:
    root = Path(run_dir)
    result_file = root / "result.json"
    if not result_file.exists():
        for alt in root.rglob("result.json"):
            result_file = alt
            break

    result = _load_json(result_file)
    if not result:
        return {"error": f"No result.json found in {run_dir}"}

    diagnosis: dict[str, Any] = {
        "run_id": result.get("run_id"),
        "status": result.get("status"),
    }

    opt_gn = result.get("optical_gradient_norm")
    diagnosis["optical_gradient_norm"] = opt_gn
    diagnosis["optical_gradient_risk"] = (
        "high" if opt_gn and opt_gn > 100 else
        "medium" if opt_gn and opt_gn > 10 else
        "low"
    )

    recon_gn = result.get("recon_gradient_norm")
    diagnosis["recon_gradient_norm"] = recon_gn
    diagnosis["gradient_ratio"] = (opt_gn / recon_gn) if opt_gn and recon_gn else None
    diagnosis["optical_dominates_recon"] = (
        diagnosis.get("gradient_ratio", 0) or 0 > 10
    )

    loss_b = result.get("reconstruction_loss_before")
    loss_a = result.get("reconstruction_loss_after")
    diagnosis["loss_before"] = loss_b
    diagnosis["loss_after"] = loss_a
    diagnosis["loss_increased"] = loss_a is not None and loss_b is not None and loss_a > loss_b
    diagnosis["loss_delta_pct"] = (
        100 * (loss_a - loss_b) / (loss_b + 1e-8)
        if loss_b and loss_a else None
    )

    mse_b = result.get("mse_before")
    mse_a = result.get("mse_after")
    diagnosis["mse_delta"] = (mse_a - mse_b) if mse_b and mse_a else None
    diagnosis["psnr_degraded"] = (
        (result.get("psnr_after") or 0) < (result.get("psnr_before") or 0)
    )

    diagnosis["full_wave_optics"] = result.get("full_wave_optics", False)
    diagnosis["phase_to_fft_proxy_used"] = result.get("phase_to_fft_proxy_used", True)
    diagnosis["evidence_level"] = result.get("evidence_level")

    # PSF analysis
    psf_before_path = root / "psf_before.npz"
    psf_after_path = root / "psf_after.npz"
    for alt in root.rglob("psf_before.npz"):
        psf_before_path = alt
        break
    for alt in root.rglob("psf_after.npz"):
        psf_after_path = alt
        break

    if psf_before_path.exists() and psf_after_path.exists():
        import numpy as np
        try:
            psf_b = np.load(psf_before_path)
            psf_a = np.load(psf_after_path)
            psf_b_arr = psf_b[list(psf_b.keys())[0]]
            psf_a_arr = psf_a[list(psf_a.keys())[0]]
            diagnosis["psf_energy_before"] = float(psf_b_arr.sum())
            diagnosis["psf_energy_after"] = float(psf_a_arr.sum())
            diagnosis["psf_energy_delta"] = diagnosis["psf_energy_after"] - diagnosis["psf_energy_before"]
            diagnosis["psf_energy_changed_pct"] = (
                100 * diagnosis["psf_energy_delta"] / (diagnosis["psf_energy_before"] + 1e-8)
            )
        except Exception:
            pass

    diagnosis["main_causes"] = []
    if diagnosis["optical_gradient_risk"] == "high":
        diagnosis["main_causes"].append("optical_gradient_too_large → needs gradient clipping or smaller LR")
    if diagnosis.get("optical_dominates_recon"):
        diagnosis["main_causes"].append("optical_gradient_dominates_recon → reduce optical_lr vs recon_lr ratio")
    if diagnosis.get("loss_increased"):
        diagnosis["main_causes"].append("loss_increased → needs rollback or staged training")
    if diagnosis.get("psnr_degraded"):
        diagnosis["main_causes"].append("psnr_degraded → optical update harms reconstruction quality")
    diagnosis["recommendations"] = [
        "Use optical_lr=1e-6 (1000x smaller than Phase 22 default)",
        "Clip optical gradients to 1.0",
        "Warm up reconstructor before joint training",
        "Enable rollback on loss increase",
        "Add PSF energy conservation regularization",
    ]

    # Save report
    report_dir = Path("workspace/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phase23_native_lens_hsi_diagnostics.json").write_text(
        json.dumps(diagnosis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (report_dir / "phase23_native_lens_hsi_diagnostics.md").write_text(
        _diagnosis_md(diagnosis), encoding="utf-8"
    )

    return diagnosis


def _diagnosis_md(d: dict[str, Any]) -> str:
    lines = [
        "# Phase 23: Native Lens HSI Co-Design Diagnosis",
        "",
        f"**Run:** {d.get('run_id')} — Status: {d.get('status')}",
        "",
        "## Gradient Analysis",
        f"- optical_gradient_norm: {d.get('optical_gradient_norm')}",
        f"- optical_gradient_risk: {d.get('optical_gradient_risk')}",
        f"- recon_gradient_norm: {d.get('recon_gradient_norm')}",
        f"- gradient_ratio (opt/recon): {d.get('gradient_ratio')}",
        f"- optical_dominates_recon: {d.get('optical_dominates_recon')}",
        "",
        "## Loss Analysis",
        f"- loss_before: {d.get('loss_before')}",
        f"- loss_after: {d.get('loss_after')}",
        f"- loss_increased: {d.get('loss_increased')}",
        f"- loss_delta_pct: {d.get('loss_delta_pct')}",
        f"- mse_delta: {d.get('mse_delta')}",
        f"- psnr_degraded: {d.get('psnr_degraded')}",
        "",
        "## Main Causes",
    ]
    for c in d.get("main_causes", []):
        lines.append(f"- {c}")
    lines.extend(["", "## Recommendations"])
    for r in d.get("recommendations", []):
        lines.append(f"- {r}")
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
