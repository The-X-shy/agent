"""Ablation matrix for Phase 23: compare 5 stabilization strategies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from optiresearch.schemas.stable_native_lens_hsi import (
    StableNativeLensHSISpec,
    StableNativeLensHSIResult,
    make_stable_lens_id,
)
from optiresearch.runtime.stable_native_lens_hsi_loop import run_stable_native_lens_hsi_codesign


def run_stable_native_lens_hsi_ablation(
    candidate: str = "GeoLensCooke",
    reconstructor: str = "differentiable_linear",
    device: str = "cpu",
    save_artifacts: bool = True,
) -> dict[str, Any]:
    base = make_stable_lens_id(candidate, reconstructor)
    strategies = {
        "baseline_phase22": dict(
            optical_lr=1e-3, optical_grad_clip=100.0,
            optical_warmup_steps=0, rollback_on_loss_increase=False,
            psf_energy_reg_weight=0, psf_width_reg_weight=0,
            recon_grad_clip=100.0, max_steps=5,
        ),
        "small_lr": dict(
            optical_lr=1e-6, optical_grad_clip=100.0,
            optical_warmup_steps=0, rollback_on_loss_increase=False,
            psf_energy_reg_weight=0, psf_width_reg_weight=0,
            recon_grad_clip=100.0, max_steps=5,
        ),
        "grad_clip": dict(
            optical_lr=1e-3, optical_grad_clip=1.0,
            optical_warmup_steps=0, rollback_on_loss_increase=False,
            psf_energy_reg_weight=0, psf_width_reg_weight=0,
            recon_grad_clip=100.0, max_steps=5,
        ),
        "staged": dict(
            optical_lr=1e-6, optical_grad_clip=100.0,
            optical_warmup_steps=3, rollback_on_loss_increase=False,
            psf_energy_reg_weight=0, psf_width_reg_weight=0,
            recon_grad_clip=5.0, max_steps=8,
        ),
        "full_stable": dict(
            optical_lr=1e-6, optical_grad_clip=1.0,
            optical_warmup_steps=3, rollback_on_loss_increase=True,
            psf_energy_reg_weight=0.1, psf_width_reg_weight=0.05,
            recon_grad_clip=5.0, max_steps=10,
        ),
    }

    results: dict[str, dict[str, Any]] = {}
    for name, overrides in strategies.items():
        spec = StableNativeLensHSISpec(
            run_id=f"{base}_{name}",
            candidate=candidate, reconstructor=reconstructor,
            device=device, save_artifacts=False,
            **overrides,
        )
        r = run_stable_native_lens_hsi_codesign(spec)
        results[name] = {
            "loss_before": r.reconstruction_loss_before,
            "loss_after": r.reconstruction_loss_after,
            "best_loss": r.best_reconstruction_loss,
            "accepted": r.accepted_update_count,
            "rejected": r.rejected_update_count,
            "rollbacks": r.rollback_count,
            "opt_grad_max": r.optical_gradient_norm_max,
            "opt_grad_mean": r.optical_gradient_norm_mean,
            "recon_grad_max": r.recon_gradient_norm_max,
            "recon_grad_mean": r.recon_gradient_norm_mean,
            "optical_changed": r.optical_parameters_changed,
            "mse_before": r.mse_before,
            "mse_after": r.mse_after,
            "psnr_before": r.psnr_before,
            "psnr_after": r.psnr_after,
            "stable": r.stable_training_succeeded,
            "evidence": r.evidence_level,
        }

    best_name = _find_best(results)
    summary = {
        "run_id_base": base,
        "candidate": candidate,
        "reconstructor": reconstructor,
        "strategies": results,
        "best_config": best_name,
    }

    if save_artifacts:
        out_dir = Path("workspace/stable_native_lens_hsi_ablation") / base
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ablation_results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "ablation_table.md").write_text(_ablation_md(summary), encoding="utf-8")
        (out_dir / "best_config.json").write_text(json.dumps({"best": best_name, "overrides": strategies.get(best_name, {})}, indent=2), encoding="utf-8")
        summary["artifact_dir"] = str(out_dir)

    return summary


def _find_best(results: dict[str, dict[str, Any]]) -> str | None:
    best_name = None
    best_loss = float("inf")
    for name, r in results.items():
        after = r.get("loss_after")
        if after is not None and after < best_loss and r.get("stable") is True:
            best_loss = after
            best_name = name
    if best_name is None:
        for name, r in results.items():
            after = r.get("loss_after")
            if after is not None and r.get("loss_before") is not None and after <= r["loss_before"]:
                best_name = name
                break
    return best_name


def _ablation_md(summary: dict[str, Any]) -> str:
    lines = ["# Ablation: Stable Native Lens HSI Co-Design", "",
             "| Strategy | Loss Before | Loss After | Best Loss | Accept/Reject/Rollback | Stable | Evidence |",
             "|----------|------------|------------|-----------|------------------------|--------|----------|"]
    for name, r in summary["strategies"].items():
        lines.append(
            f"| {name} | {r['loss_before']:.4f} | {r['loss_after']:.4f} | "
            f"{r['best_loss']:.4f} | {r['accepted']}/{r['rejected']}/{r['rollbacks']} | "
            f"{r['stable']} | {r['evidence']} |"
        )
    lines.extend(["", f"**Best config:** {summary['best_config']}"])
    return "\n".join(lines)
