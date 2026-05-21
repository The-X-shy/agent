"""Native HSI Reconstruction CoDesign optimization loop for Phase 21.

Jointly optimizes DeepLens optical parameters and a trainable HSI reconstructor
using full differentiable reconstruction loss.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from optiresearch.adapters.deeplens_differentiable_bridge import (
    Binary2PhaseHSIBridge,
    FresnelHSIBridge,
    validate_autograd_chain,
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
from optiresearch.hsi.torch_metrics import torch_psnr, torch_sam
from optiresearch.schemas.native_hsi_reconstruction_codesign import (
    NativeHSIReconstructionCoDesignResult,
    NativeHSIReconstructionCoDesignSpec,
)

BRIDGE_CLASSES = {"Fresnel": FresnelHSIBridge, "Binary2Phase": Binary2PhaseHSIBridge}
RECONSTRUCTOR_CLASSES = {
    "differentiable_linear": DifferentiableLinearHSIReconstructor,
    "tiny_cnn": TinyDifferentiableHSIReconstructor,
}


def run_native_hsi_reconstruction_codesign(
    spec: NativeHSIReconstructionCoDesignSpec,
) -> NativeHSIReconstructionCoDesignResult:
    metadata: dict[str, Any] = {"optimizer_step_executed": False}
    caveats: list[str] = []

    bridge_cls = BRIDGE_CLASSES.get(spec.optical_component)
    if bridge_cls is None:
        return _unsupported(spec, "UNSUPPORTED_COMPONENT",
                            f"No bridge for {spec.optical_component}", metadata, caveats)

    recon_cls = RECONSTRUCTOR_CLASSES.get(spec.reconstructor)
    if recon_cls is None:
        return _unsupported(spec, "UNSUPPORTED_RECONSTRUCTOR",
                            f"No reconstructor for {spec.reconstructor}", metadata, caveats)

    try:
        bridge = bridge_cls(device=spec.device)
        bridge.build_component()
    except Exception as exc:
        return _unsupported(spec, "BUILD_FAILED", str(exc), metadata, caveats)

    try:
        opt_optimizer = bridge.get_optimizer(learning_rate=spec.optical_lr)
    except Exception as exc:
        return _unsupported(spec, "OPTIMIZER_UNAVAILABLE", str(exc), metadata, caveats)

    try:
        reconstructor = recon_cls(bands=spec.bands).to(spec.device)
        recon_optimizer = torch.optim.Adam(
            reconstructor.parameters(), lr=spec.recon_lr
        ) if spec.optimize_reconstructor else None
    except Exception as exc:
        return _unsupported(spec, "RECONSTRUCTOR_BUILD_FAILED", str(exc), metadata, caveats)

    optical_before = bridge.parameter_snapshot()
    metadata["optical_parameter_before"] = optical_before

    optical_grad_norm: float | None = None
    recon_grad_norm: float | None = None
    loss_before: float | None = None
    loss_after: float | None = None
    mse_before: float | None = None
    mse_after: float | None = None
    psnr_before: float | None = None
    psnr_after: float | None = None
    sam_before: float | None = None
    sam_after: float | None = None
    loss_trace: list[dict[str, Any]] = []

    try:
        hsi_target = generate_torch_synthetic_hsi(
            batch=spec.batch_size, bands=spec.bands,
            height=spec.image_size, width=spec.image_size, device=spec.device,
        )

        psf = bridge.psf_from_component_torch(num_bands=spec.bands, psf_size=spec.psf_size)
        measurement = make_measurement_from_psf_torch(hsi_target, psf)
        recon = reconstructor(measurement, psf)
        losses = hsi_reconstruction_losses(recon, hsi_target, measurement, psf, spec.loss_weights)
        loss_before = float(losses["total_loss"].detach().cpu().item())
        mse_before = float(losses["mse_loss"].detach().cpu().item())
        psnr_before = float(torch_psnr(recon, hsi_target).detach().cpu().item())
        sam_before = float(torch_sam(recon, hsi_target).detach().cpu().item())

        if not losses["total_loss"].requires_grad:
            return _autograd_break(spec, "LOSS_NOT_DIFFERENTIABLE",
                                   "Reconstruction loss does not require grad",
                                   metadata, caveats, loss_before=loss_before)

        for step in range(spec.max_steps):
            if spec.optimize_optics:
                opt_optimizer.zero_grad()
            if spec.optimize_reconstructor and recon_optimizer is not None:
                recon_optimizer.zero_grad()

            psf = bridge.psf_from_component_torch(num_bands=spec.bands, psf_size=spec.psf_size)
            measurement = make_measurement_from_psf_torch(hsi_target, psf)
            recon = reconstructor(measurement, psf)
            losses = hsi_reconstruction_losses(recon, hsi_target, measurement, psf, spec.loss_weights)
            step_loss = losses["total_loss"]

            step_loss.backward()

            chain = validate_autograd_chain(
                step_loss,
                bridge.get_trainable_parameters(),
                list(reconstructor.parameters()) if spec.optimize_reconstructor else None,
            )
            optical_grad_norm = chain["optical_gradient_norm"]
            recon_grad_norm = chain["recon_gradient_norm"]

            if spec.optimize_optics:
                opt_optimizer.step()
            if spec.optimize_reconstructor and recon_optimizer is not None:
                recon_optimizer.step()
            metadata["optimizer_step_executed"] = True

            loss_trace.append({
                "step": step,
                "total_loss": float(step_loss.detach().cpu().item()),
                "mse": float(losses["mse_loss"].detach().cpu().item()),
                "optical_gradient_norm": optical_grad_norm,
                "recon_gradient_norm": recon_grad_norm,
            })

        psf_after = bridge.psf_from_component_torch(num_bands=spec.bands, psf_size=spec.psf_size)
        measurement_after = make_measurement_from_psf_torch(hsi_target, psf_after)
        recon_after = reconstructor(measurement_after, psf_after)
        losses_after = hsi_reconstruction_losses(
            recon_after, hsi_target, measurement_after, psf_after, spec.loss_weights
        )
        loss_after = float(losses_after["total_loss"].detach().cpu().item())
        mse_after = float(losses_after["mse_loss"].detach().cpu().item())
        psnr_after = float(torch_psnr(recon_after, hsi_target).detach().cpu().item())
        sam_after = float(torch_sam(recon_after, hsi_target).detach().cpu().item())

    except Exception as exc:
        return _failed(spec, "LOOP_FAILED", str(exc), metadata, caveats,
                       loss_before=loss_before, optical_grad_norm=optical_grad_norm)

    optical_after = bridge.parameter_snapshot()
    metadata["optical_parameter_after"] = optical_after

    optical_changed = _parameters_changed(optical_before, optical_after)
    differentiable = bool(
        optical_grad_norm is not None and optical_grad_norm > 0
        and optical_changed and metadata["optimizer_step_executed"]
    )
    evidence_level = "native_full_reconstruction_proxy" if differentiable else None

    result = NativeHSIReconstructionCoDesignResult(
        run_id=spec.run_id,
        status="succeeded" if differentiable else "unsupported",
        optical_component=spec.optical_component,
        reconstructor=spec.reconstructor,
        differentiable=differentiable,
        full_reconstruction_loss_used=True,
        native_parameter_update=differentiable,
        full_wave_optics=False,
        phase_to_fft_proxy_used=True,
        reconstruction_loss_before=loss_before,
        reconstruction_loss_after=loss_after,
        mse_before=mse_before,
        mse_after=mse_after,
        psnr_before=psnr_before,
        psnr_after=psnr_after,
        sam_before=sam_before,
        sam_after=sam_after,
        optical_gradient_norm=optical_grad_norm,
        recon_gradient_norm=recon_grad_norm,
        optical_parameters_changed=optical_changed,
        optimizer_step_executed=metadata["optimizer_step_executed"],
        autograd_graph_exists=differentiable,
        evidence_level=evidence_level,
        caveats=caveats if differentiable else [*caveats, "Reconstruction loss did not backprop to optical params"],
        metadata=metadata,
    )

    if spec.save_artifacts:
        result.artifact_paths = _save_artifacts(spec, result, loss_trace, optical_before, optical_after)
    return result


def _parameters_changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    for key, before_value in before.items():
        after_value = after.get(key)
        if isinstance(before_value, dict) or isinstance(after_value, dict):
            before_norm = float((before_value or {}).get("norm", 0.0))
            after_norm = float((after_value or {}).get("norm", 0.0))
            if abs(after_norm - before_norm) > 1e-12:
                return True
        elif after_value is not None and abs(float(after_value) - float(before_value)) > 1e-12:
            return True
    return False


def _save_artifacts(
    spec: NativeHSIReconstructionCoDesignSpec,
    result: NativeHSIReconstructionCoDesignResult,
    loss_trace: list[dict[str, Any]],
    optical_before: dict[str, Any],
    optical_after: dict[str, Any],
) -> list[str]:
    out_dir = Path("workspace/native_hsi_reconstruction_codesign") / spec.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "spec.json").write_text(
        json.dumps(spec.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "result.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (out_dir / "loss_trace.json").write_text(
        json.dumps(loss_trace, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (out_dir / "optical_parameter_trace.json").write_text(
        json.dumps({"before": optical_before, "after": optical_after}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metrics = {
        "reconstruction_loss_before": result.reconstruction_loss_before,
        "reconstruction_loss_after": result.reconstruction_loss_after,
        "mse_before": result.mse_before,
        "mse_after": result.mse_after,
        "psnr_before": result.psnr_before,
        "psnr_after": result.psnr_after,
        "sam_before": result.sam_before,
        "sam_after": result.sam_after,
        "optical_gradient_norm": result.optical_gradient_norm,
        "recon_gradient_norm": result.recon_gradient_norm,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "report.md").write_text("\n".join(_report_lines(result)), encoding="utf-8")

    paths = []
    for name in ["spec.json", "result.json", "loss_trace.json", "optical_parameter_trace.json",
                 "metrics.json", "report.md"]:
        if (out_dir / name).exists():
            paths.append(str((out_dir / name).relative_to(Path("workspace"))))
    return paths


def _report_lines(result: NativeHSIReconstructionCoDesignResult) -> list[str]:
    return [
        f"# Native HSI Reconstruction CoDesign: {result.optical_component} / {result.reconstructor}",
        "",
        f"- status: {result.status}",
        f"- differentiable: {result.differentiable}",
        f"- full_reconstruction_loss_used: {result.full_reconstruction_loss_used}",
        f"- full_wave_optics: {result.full_wave_optics}",
        f"- phase_to_fft_proxy_used: {result.phase_to_fft_proxy_used}",
        f"- evidence_level: {result.evidence_level}",
        f"- reconstruction_loss_before: {result.reconstruction_loss_before}",
        f"- reconstruction_loss_after: {result.reconstruction_loss_after}",
        f"- mse_before: {result.mse_before} / mse_after: {result.mse_after}",
        f"- psnr_before: {result.psnr_before} / psnr_after: {result.psnr_after}",
        f"- sam_before: {result.sam_before} / sam_after: {result.sam_after}",
        f"- optical_gradient_norm: {result.optical_gradient_norm}",
        f"- recon_gradient_norm: {result.recon_gradient_norm}",
        f"- optical_parameters_changed: {result.optical_parameters_changed}",
        f"- optimizer_step_executed: {result.optimizer_step_executed}",
        f"- caveats: {result.caveats}",
    ]


def _unsupported(
    spec: NativeHSIReconstructionCoDesignSpec,
    error_code: str,
    error_message: str,
    metadata: dict[str, Any],
    caveats: list[str],
) -> NativeHSIReconstructionCoDesignResult:
    return NativeHSIReconstructionCoDesignResult(
        run_id=spec.run_id, status="unsupported",
        optical_component=spec.optical_component, reconstructor=spec.reconstructor,
        error_code=error_code, error_message=error_message,
        caveats=[*caveats, error_message], metadata=metadata,
    )


def _autograd_break(
    spec: NativeHSIReconstructionCoDesignSpec,
    error_code: str, error_message: str,
    metadata: dict[str, Any], caveats: list[str],
    loss_before: float | None = None,
) -> NativeHSIReconstructionCoDesignResult:
    return NativeHSIReconstructionCoDesignResult(
        run_id=spec.run_id, status="unsupported",
        optical_component=spec.optical_component, reconstructor=spec.reconstructor,
        differentiable=False, autograd_graph_exists=False,
        reconstruction_loss_before=loss_before,
        error_code=error_code, error_message=error_message,
        caveats=[*caveats, error_message], metadata=metadata,
    )


def _failed(
    spec: NativeHSIReconstructionCoDesignSpec,
    error_code: str, error_message: str,
    metadata: dict[str, Any], caveats: list[str],
    loss_before: float | None = None,
    optical_grad_norm: float | None = None,
) -> NativeHSIReconstructionCoDesignResult:
    return NativeHSIReconstructionCoDesignResult(
        run_id=spec.run_id, status="failed",
        optical_component=spec.optical_component, reconstructor=spec.reconstructor,
        differentiable=False, reconstruction_loss_before=loss_before,
        optical_gradient_norm=optical_grad_norm,
        error_code=error_code, error_message=error_message,
        caveats=[*caveats, error_message], metadata=metadata,
    )
