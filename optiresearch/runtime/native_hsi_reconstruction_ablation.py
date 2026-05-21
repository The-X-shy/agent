"""Ablation study for Phase 21: compare optimization modes.

Four modes:
- reconstructor_only: optics frozen, only reconstructor trained
- optics_only: reconstructor frozen, only optics trained
- joint_optics_reconstructor: both trained (full co-design)
- no_native_optics: fixed initial PSF, only reconstructor trained (baseline)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from optiresearch.adapters.deeplens_differentiable_bridge import (
    FresnelHSIBridge,
    Binary2PhaseHSIBridge,
)
from optiresearch.hsi.differentiable_proxy import (
    generate_torch_synthetic_hsi,
    make_measurement_from_psf_torch,
)
from optiresearch.hsi.differentiable_reconstructor import (
    DifferentiableLinearHSIReconstructor,
    TinyDifferentiableHSIReconstructor,
    hsi_reconstruction_losses,
)
from optiresearch.schemas.native_hsi_reconstruction_codesign import (
    make_recon_codesign_id,
)

BRIDGE_CLASSES = {"Fresnel": FresnelHSIBridge, "Binary2Phase": Binary2PhaseHSIBridge}
RECON_CLASSES = {"differentiable_linear": DifferentiableLinearHSIReconstructor, "tiny_cnn": TinyDifferentiableHSIReconstructor}


def run_native_hsi_reconstruction_ablation(
    optical_component: str = "Fresnel",
    reconstructor_name: str = "differentiable_linear",
    bands: int = 4,
    image_size: int = 16,
    psf_size: int = 8,
    max_steps: int = 5,
    optical_lr: float = 1e-3,
    recon_lr: float = 1e-3,
    device: str = "cpu",
    save_artifacts: bool = True,
) -> dict[str, Any]:
    bridge_cls = BRIDGE_CLASSES[optical_component]
    recon_cls = RECON_CLASSES[reconstructor_name]

    run_id_base = make_recon_codesign_id(optical_component, reconstructor_name)
    results: dict[str, Any] = {}

    # Common setup
    bridge = bridge_cls(device=device)
    bridge.build_component()
    hsi_target = generate_torch_synthetic_hsi(batch=1, bands=bands, height=image_size, width=image_size, device=device)

    # Mode 1: reconstructor_only
    results["reconstructor_only"] = _run_mode(
        "reconstructor_only", bridge, recon_cls, hsi_target,
        train_optics=False, train_recon=True, bands=bands, psf_size=psf_size,
        max_steps=max_steps, optical_lr=optical_lr, recon_lr=recon_lr, device=device,
    )

    # Reset bridge
    bridge = bridge_cls(device=device)
    bridge.build_component()

    # Mode 2: optics_only (freeze reconstructor)
    results["optics_only"] = _run_mode(
        "optics_only", bridge, recon_cls, hsi_target,
        train_optics=True, train_recon=False, bands=bands, psf_size=psf_size,
        max_steps=max_steps, optical_lr=optical_lr, recon_lr=recon_lr, device=device,
    )

    # Reset bridge
    bridge = bridge_cls(device=device)
    bridge.build_component()

    # Mode 3: joint
    results["joint_optics_reconstructor"] = _run_mode(
        "joint_optics_reconstructor", bridge, recon_cls, hsi_target,
        train_optics=True, train_recon=True, bands=bands, psf_size=psf_size,
        max_steps=max_steps, optical_lr=optical_lr, recon_lr=recon_lr, device=device,
    )

    # Mode 4: no_native_optics (fixed initial PSF)
    bridge = bridge_cls(device=device)
    bridge.build_component()
    fixed_psf = bridge.psf_from_component_torch(num_bands=bands, psf_size=psf_size).detach()
    results["no_native_optics"] = _run_mode_no_optics(
        recon_cls, hsi_target, fixed_psf, bands=bands,
        max_steps=max_steps, recon_lr=recon_lr, device=device,
    )

    summary = {"run_id_base": run_id_base, "optical_component": optical_component,
               "reconstructor": reconstructor_name, "modes": results}

    if save_artifacts:
        out_dir = Path("workspace/native_hsi_reconstruction_ablation") / run_id_base
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ablation_results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "report.md").write_text("\n".join(_ablation_table(summary)), encoding="utf-8")
        summary["artifact_dir"] = str(out_dir)

    return summary


def _run_mode(
    mode: str, bridge, recon_cls, hsi_target,
    train_optics: bool, train_recon: bool,
    bands: int, psf_size: int, max_steps: int,
    optical_lr: float, recon_lr: float, device: str,
) -> dict[str, Any]:
    reconstructor = recon_cls(bands=bands).to(device)
    opt_opt = bridge.get_optimizer(learning_rate=optical_lr) if train_optics else None
    recon_opt = torch.optim.Adam(reconstructor.parameters(), lr=recon_lr) if train_recon else None

    psf = bridge.psf_from_component_torch(num_bands=bands, psf_size=psf_size)
    measurement = make_measurement_from_psf_torch(hsi_target, psf)
    recon = reconstructor(measurement, psf)
    losses = hsi_reconstruction_losses(recon, hsi_target, measurement, psf)
    loss_before = float(losses["total_loss"].detach().cpu().item())
    mse_before = float(losses["mse_loss"].detach().cpu().item())

    for _ in range(max_steps):
        if opt_opt:
            opt_opt.zero_grad()
        if recon_opt:
            recon_opt.zero_grad()

        psf = bridge.psf_from_component_torch(num_bands=bands, psf_size=psf_size)
        measurement = make_measurement_from_psf_torch(hsi_target, psf)
        recon = reconstructor(measurement, psf)
        losses = hsi_reconstruction_losses(recon, hsi_target, measurement, psf)
        losses["total_loss"].backward()

        if opt_opt:
            opt_opt.step()
        if recon_opt:
            recon_opt.step()

    psf = bridge.psf_from_component_torch(num_bands=bands, psf_size=psf_size)
    measurement = make_measurement_from_psf_torch(hsi_target, psf)
    recon = reconstructor(measurement, psf)
    losses = hsi_reconstruction_losses(recon, hsi_target, measurement, psf)
    loss_after = float(losses["total_loss"].detach().cpu().item())
    mse_after = float(losses["mse_loss"].detach().cpu().item())

    return {"mode": mode, "train_optics": train_optics, "train_recon": train_recon,
            "loss_before": loss_before, "loss_after": loss_after,
            "mse_before": mse_before, "mse_after": mse_after}


def _run_mode_no_optics(
    recon_cls, hsi_target, fixed_psf, bands: int, max_steps: int, recon_lr: float, device: str,
) -> dict[str, Any]:
    reconstructor = recon_cls(bands=bands).to(device)
    recon_opt = torch.optim.Adam(reconstructor.parameters(), lr=recon_lr)

    measurement = make_measurement_from_psf_torch(hsi_target, fixed_psf)
    recon = reconstructor(measurement, fixed_psf)
    losses = hsi_reconstruction_losses(recon, hsi_target, measurement, fixed_psf)
    loss_before = float(losses["total_loss"].detach().cpu().item())

    for _ in range(max_steps):
        recon_opt.zero_grad()
        recon = reconstructor(measurement, fixed_psf)
        losses = hsi_reconstruction_losses(recon, hsi_target, measurement, fixed_psf)
        losses["total_loss"].backward()
        recon_opt.step()

    recon = reconstructor(measurement, fixed_psf)
    losses = hsi_reconstruction_losses(recon, hsi_target, measurement, fixed_psf)
    loss_after = float(losses["total_loss"].detach().cpu().item())

    return {"mode": "no_native_optics", "train_optics": False, "train_recon": True,
            "loss_before": loss_before, "loss_after": loss_after}


def _ablation_table(summary: dict[str, Any]) -> list[str]:
    lines = ["# Ablation: Native HSI Reconstruction CoDesign", "",
             "| Mode | Train Optics | Train Recon | Loss Before | Loss After |",
             "|------|-------------|-------------|-------------|------------|"]
    for key, r in summary["modes"].items():
        lines.append(f"| {key} | {r['train_optics']} | {r['train_recon']} | {r['loss_before']:.6f} | {r['loss_after']:.6f} |")
    return lines
