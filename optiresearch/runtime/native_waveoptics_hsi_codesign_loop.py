"""Full wave-optics HSI reconstruction co-design loop for Phase 22.

Integrates DeepLens's native differentiable wave-optics PSF path
(GeoLens.psf(model="coherent") → ASM) with Phase 21's trainable
HSI reconstructor for joint optics+reconstructor optimization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from optiresearch.adapters.geolens_waveoptics_bridge import GeoLensWaveOpticsBridge
from optiresearch.adapters.deeplens_differentiable_bridge import validate_autograd_chain
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
)

RECON_CLASSES = {"differentiable_linear": DifferentiableLinearHSIReconstructor,
                  "tiny_cnn": TinyDifferentiableHSIReconstructor}


def run_native_waveoptics_hsi_codesign(
    spec: Any,
) -> NativeHSIReconstructionCoDesignResult:
    metadata: dict[str, Any] = {"optimizer_step_executed": False}
    caveats: list[str] = []

    recon_cls = RECON_CLASSES.get(spec.reconstructor)
    if recon_cls is None:
        return _unsupported(spec, "UNSUPPORTED_RECONSTRUCTOR", f"No reconstructor: {spec.reconstructor}", metadata, caveats)

    try:
        bridge = GeoLensWaveOpticsBridge(device=spec.device)
        bridge.build_component(lens_file=getattr(spec, "lens_file", None))
    except Exception as exc:
        return _unsupported(spec, "BUILD_FAILED", str(exc), metadata, caveats)

    try:
        opt_optimizer = bridge.get_optimizer(learning_rate=spec.optical_lr)
    except Exception as exc:
        return _unsupported(spec, "OPTIMIZER_UNAVAILABLE", str(exc), metadata, caveats)

    try:
        reconstructor = recon_cls(bands=spec.bands).to(spec.device)
        recon_optimizer = torch.optim.Adam(reconstructor.parameters(), lr=spec.recon_lr)
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

    try:
        hsi_target = generate_torch_synthetic_hsi(
            batch=1, bands=spec.bands, height=spec.image_size, width=spec.image_size, device=spec.device,
        )

        psf = _normalize_psf_cube_for_hsi(bridge.psf_cube_torch(num_bands=spec.bands, ks=spec.psf_size))
        measurement = make_measurement_from_psf_torch(hsi_target, psf)
        recon = reconstructor(measurement, psf)
        losses = hsi_reconstruction_losses(recon, hsi_target, measurement, psf, spec.loss_weights)
        loss_before = float(losses["total_loss"].detach().cpu().item())
        mse_before = float(losses["mse_loss"].detach().cpu().item())
        psnr_before = float(torch_psnr(recon, hsi_target).detach().cpu().item())
        sam_before = float(torch_sam(recon, hsi_target).detach().cpu().item())

        if not losses["total_loss"].requires_grad:
            return _unsupported(spec, "LOSS_NOT_DIFFERENTIABLE",
                                "Reconstruction loss does not require grad", metadata, caveats)

        for _ in range(spec.max_steps):
            opt_optimizer.zero_grad()
            recon_optimizer.zero_grad()

            psf = _normalize_psf_cube_for_hsi(bridge.psf_cube_torch(num_bands=spec.bands, ks=spec.psf_size))
            measurement = make_measurement_from_psf_torch(hsi_target, psf)
            recon = reconstructor(measurement, psf)
            losses = hsi_reconstruction_losses(recon, hsi_target, measurement, psf, spec.loss_weights)
            losses["total_loss"].backward()

            chain = validate_autograd_chain(losses["total_loss"], bridge.get_trainable_parameters(),
                                            list(reconstructor.parameters()))
            optical_grad_norm = chain["optical_gradient_norm"]
            recon_grad_norm = chain["recon_gradient_norm"]

            opt_optimizer.step()
            recon_optimizer.step()
            metadata["optimizer_step_executed"] = True

        psf_after = _normalize_psf_cube_for_hsi(bridge.psf_cube_torch(num_bands=spec.bands, ks=spec.psf_size))
        measurement_after = make_measurement_from_psf_torch(hsi_target, psf_after)
        recon_after = reconstructor(measurement_after, psf_after)
        losses_after = hsi_reconstruction_losses(recon_after, hsi_target, measurement_after, psf_after, spec.loss_weights)
        loss_after = float(losses_after["total_loss"].detach().cpu().item())
        mse_after = float(losses_after["mse_loss"].detach().cpu().item())
        psnr_after = float(torch_psnr(recon_after, hsi_target).detach().cpu().item())
        sam_after = float(torch_sam(recon_after, hsi_target).detach().cpu().item())

    except Exception as exc:
        return _failed(spec, "LOOP_FAILED", str(exc), metadata, caveats)

    optical_after = bridge.parameter_snapshot()
    metadata["optical_parameter_after"] = optical_after
    optical_changed = _params_changed(optical_before, optical_after)
    differentiable = bool(optical_grad_norm and optical_grad_norm > 0 and optical_changed and metadata["optimizer_step_executed"])
    evidence_level = "native_lens_hsi_codesign" if differentiable else None
    metadata["deeplens_native_psf_path"] = getattr(bridge, "deeplens_native_psf_path", "geolens.psf_geometric")

    result = NativeHSIReconstructionCoDesignResult(
        run_id=spec.run_id, status="succeeded" if differentiable else "unsupported",
        optical_component="GeoLensCooke", reconstructor=spec.reconstructor,
        differentiable=differentiable, full_reconstruction_loss_used=True,
        native_parameter_update=differentiable,
        full_wave_optics=False, phase_to_fft_proxy_used=False,
        reconstruction_loss_before=loss_before, reconstruction_loss_after=loss_after,
        mse_before=mse_before, mse_after=mse_after,
        psnr_before=psnr_before, psnr_after=psnr_after,
        sam_before=sam_before, sam_after=sam_after,
        optical_gradient_norm=optical_grad_norm, recon_gradient_norm=recon_grad_norm,
        optical_parameters_changed=optical_changed,
        optimizer_step_executed=metadata["optimizer_step_executed"],
        autograd_graph_exists=differentiable, evidence_level=evidence_level,
        caveats=caveats if differentiable else [*caveats, "Wave-optics HSI co-design did not update optical params"],
        metadata=metadata,
    )

    if spec.save_artifacts:
        out_dir = Path("workspace/waveoptics_hsi_codesign") / spec.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (out_dir / "report.md").write_text(f"# Wave-Optics HSI CoDesign\n- status: {result.status}\n- full_wave_optics: {result.full_wave_optics}\n- phase_to_fft_proxy_used: {result.phase_to_fft_proxy_used}\n- evidence_level: {result.evidence_level}\n- loss: {loss_before} -> {loss_after}\n- mse: {mse_before} -> {mse_after}\n- optical_grad: {optical_grad_norm}\n- recon_grad: {recon_grad_norm}\n", encoding="utf-8")
        result.artifact_paths = [str((out_dir / n).relative_to(Path("workspace"))) for n in ["result.json", "report.md"]]
    return result


def _params_changed(before, after):
    for key, bv in before.items():
        av = after.get(key)
        if isinstance(bv, dict):
            if abs(float((av or {}).get("norm", 0)) - float((bv or {}).get("norm", 0))) > 1e-12:
                return True
        elif av is not None and abs(float(av) - float(bv)) > 1e-12:
            return True
    return False


def _normalize_psf_cube_for_hsi(psf: torch.Tensor) -> torch.Tensor:
    if psf.ndim == 4 and psf.shape[1] == 1:
        return psf[:, 0]
    if psf.ndim != 3:
        raise ValueError(f"Expected PSF cube [bands, k, k] or [bands, 1, k, k], got shape {tuple(psf.shape)}")
    return psf


def _unsupported(spec, code, msg, meta, cav):
    return NativeHSIReconstructionCoDesignResult(run_id=spec.run_id, status="unsupported", optical_component="GeoLensCooke", reconstructor=spec.reconstructor, error_code=code, error_message=msg, caveats=[*cav, msg], metadata=meta)


def _failed(spec, code, msg, meta, cav):
    return NativeHSIReconstructionCoDesignResult(run_id=spec.run_id, status="failed", optical_component="GeoLensCooke", reconstructor=spec.reconstructor, error_code=code, error_message=msg, caveats=[*cav, msg], metadata=meta)
