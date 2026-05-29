"""Synthetic HSI forward loop for component surrogate PSF co-design."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from optiresearch.hsi.differentiable_proxy import (
    generate_torch_synthetic_hsi,
    make_measurement_from_psf_torch,
)
from optiresearch.hsi.differentiable_reconstructor import (
    DifferentiableLinearHSIReconstructor,
    hsi_reconstruction_losses,
)
from optiresearch.hsi.torch_metrics import torch_psnr, torch_sam
from optiresearch.optics.component_surrogate_psf import build_component_surrogate_psf
from optiresearch.schemas.component_surrogate_psf import (
    ComponentSurrogateHSICoDesignResult,
    ComponentSurrogateHSICoDesignSpec,
)

CLAIM = "component_surrogate_hsi_codesign"


def run_component_surrogate_hsi_forward(
    spec: ComponentSurrogateHSICoDesignSpec,
) -> ComponentSurrogateHSICoDesignResult:
    if spec.dataset != "synthetic":
        return ComponentSurrogateHSICoDesignResult(
            run_id=spec.run_id or "component_surrogate_hsi_unset",
            component_type=spec.component_type,
            status="needs_followup",
            error_code="UNSUPPORTED_DATASET",
            error_message="Component surrogate HSI co-design currently supports synthetic data only",
            errors=["Component surrogate HSI co-design currently supports synthetic data only"],
            evidence_level="diagnostic_evidence",
            claim_ceiling="diagnostic_evidence",
        )

    torch.manual_seed(spec.seed)
    device = torch.device(spec.device)
    psf_spec = spec.psf_spec
    if psf_spec is None:
        raise ValueError("ComponentSurrogateHSICoDesignSpec.psf_spec was not populated")

    psf_result = build_component_surrogate_psf(psf_spec)
    if psf_result.status != "succeeded":
        return ComponentSurrogateHSICoDesignResult(
            run_id=spec.run_id or "component_surrogate_hsi_unset",
            component_type=spec.component_type,
            status="needs_followup",
            psf_requires_grad=psf_result.psf_requires_grad,
            evidence_level="diagnostic_evidence",
            claim_ceiling="diagnostic_evidence",
            psf_summary=psf_result.model_dump(mode="json"),
            error_code=psf_result.error_code,
            error_message=psf_result.error_message,
            errors=psf_result.errors,
        )

    params: list[torch.nn.Parameter] = list(psf_result.component_parameters)
    before_params = [p.detach().clone() for p in params]
    reconstructor = DifferentiableLinearHSIReconstructor(bands=spec.band_count).to(device)
    optim_params: list[torch.nn.Parameter] = [*params]
    if spec.optimize_reconstructor:
        optim_params.extend(list(reconstructor.parameters()))
    optimizer = torch.optim.Adam(
        [
            {"params": params, "lr": spec.component_lr},
            {"params": list(reconstructor.parameters()), "lr": spec.recon_lr},
        ]
        if spec.optimize_reconstructor
        else [{"params": params, "lr": spec.component_lr}]
    )

    hsi_target = generate_torch_synthetic_hsi(
        batch=spec.batch_size,
        bands=spec.band_count,
        height=spec.image_size,
        width=spec.image_size,
        device=spec.device,
    )

    before = _evaluate(spec, reconstructor, hsi_target, params)
    loss_requires_grad = bool(before["loss"].requires_grad)
    if not loss_requires_grad:
        return ComponentSurrogateHSICoDesignResult(
            run_id=spec.run_id or "component_surrogate_hsi_unset",
            component_type=spec.component_type,
            status="failed",
            reconstruction_loss_before=_float(before["loss"]),
            psf_requires_grad=bool(before["psf"].requires_grad),
            loss_requires_grad=False,
            error_code="LOSS_GRAPH_DISCONNECTED",
            error_message="Reconstruction loss does not require gradients",
            errors=["Reconstruction loss does not require gradients"],
        )

    loss_trace: list[dict[str, Any]] = []
    component_grad_norm_max = 0.0
    best_loss = _float(before["loss"])
    best_metrics = before
    best_params = [p.detach().clone() for p in params]
    best_recon_state = {k: v.detach().clone() for k, v in reconstructor.state_dict().items()}

    for step in range(spec.steps):
        optimizer.zero_grad()
        current = _evaluate(spec, reconstructor, hsi_target, params)
        loss = current["loss"]
        loss.backward()
        grad_norm = _component_grad_norm(params)
        component_grad_norm_max = max(component_grad_norm_max, grad_norm)
        optimizer.step()
        step_loss = _float(loss)
        loss_trace.append({
            "step": step,
            "reconstruction_loss": step_loss,
            "mse": _float(current["mse"]),
            "component_grad_norm": grad_norm,
        })
        after_step = _evaluate(spec, reconstructor, hsi_target, params)
        after_step_loss = _float(after_step["loss"])
        if after_step_loss <= best_loss:
            best_loss = after_step_loss
            best_metrics = after_step
            best_params = [p.detach().clone() for p in params]
            best_recon_state = {k: v.detach().clone() for k, v in reconstructor.state_dict().items()}

    for param, best in zip(params, best_params):
        param.data.copy_(best)
    reconstructor.load_state_dict(best_recon_state)
    after = best_metrics

    parameter_changed = any(
        not torch.allclose(param.detach(), before_value, atol=1e-12, rtol=1e-12)
        for param, before_value in zip(params, before_params)
    )

    result = ComponentSurrogateHSICoDesignResult(
        run_id=spec.run_id or "component_surrogate_hsi_unset",
        component_type=spec.component_type,
        status="succeeded" if parameter_changed and component_grad_norm_max > 0 else "needs_followup",
        reconstruction_loss_before=_float(before["loss"]),
        reconstruction_loss_after=_float(after["loss"]),
        mse_before=_float(before["mse"]),
        mse_after=_float(after["mse"]),
        psnr_before=_float(before["psnr"]),
        psnr_after=_float(after["psnr"]),
        sam_before=_float(before["sam"]),
        sam_after=_float(after["sam"]),
        component_grad_norm_max=component_grad_norm_max,
        component_parameter_changed=parameter_changed,
        psf_requires_grad=bool(before["psf"].requires_grad),
        loss_requires_grad=loss_requires_grad,
        evidence_level=CLAIM if parameter_changed and component_grad_norm_max > 0 else "diagnostic_evidence",
        claim_ceiling=CLAIM if parameter_changed and component_grad_norm_max > 0 else "diagnostic_evidence",
        psf_summary={
            **psf_result.model_dump(mode="json"),
            "parameter_names": psf_result.parameter_names,
        },
        warnings=[
            "surrogate_psf_not_full_geolens",
            "synthetic_hsi_only",
            "no_real_camera_validation",
        ],
        metadata={
            "dataset": spec.dataset,
            "component_type": spec.component_type,
            "full_geolens_psf_used": False,
            "full_wave_optics": False,
            "phase_to_fft_proxy_used": True,
            "synthetic_data": True,
            "physical_backend": False,
            "native_backend": False,
        },
    )
    if spec.save_artifacts:
        result.artifacts = _save_artifacts(spec, result, loss_trace, before, after, params)
    return result


def _evaluate(
    spec: ComponentSurrogateHSICoDesignSpec,
    reconstructor: DifferentiableLinearHSIReconstructor,
    hsi_target: torch.Tensor,
    params: list[torch.nn.Parameter],
) -> dict[str, torch.Tensor]:
    psf_spec = spec.psf_spec
    if psf_spec is None:
        raise ValueError("Missing PSF spec")
    psf_result = build_component_surrogate_psf(psf_spec, initial_parameters=params)
    psf = psf_result.psf
    measurement = make_measurement_from_psf_torch(hsi_target, psf)
    recon = reconstructor(measurement, psf)
    losses = hsi_reconstruction_losses(recon, hsi_target, measurement, psf, spec.loss_weights)
    return {
        "psf": psf,
        "measurement": measurement,
        "recon": recon,
        "loss": losses["total_loss"],
        "mse": losses["mse_loss"],
        "psnr": torch_psnr(recon, hsi_target),
        "sam": torch_sam(recon, hsi_target),
    }


def _component_grad_norm(params: list[torch.nn.Parameter]) -> float:
    norms = [
        float(param.grad.detach().norm().cpu())
        for param in params
        if param.grad is not None and bool((param.grad.detach().abs().sum() > 0).cpu())
    ]
    return max(norms) if norms else 0.0


def _save_artifacts(
    spec: ComponentSurrogateHSICoDesignSpec,
    result: ComponentSurrogateHSICoDesignResult,
    loss_trace: list[dict[str, Any]],
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    params: list[torch.nn.Parameter],
) -> list[str]:
    out_dir = Path("workspace/component_surrogate_hsi") / (spec.run_id or result.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "spec.json").write_text(
        json.dumps(spec.model_dump(mode="json"), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out_dir / "result.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    metrics = {
        "component_type": result.component_type,
        "status": result.status,
        "reconstruction_loss_before": result.reconstruction_loss_before,
        "reconstruction_loss_after": result.reconstruction_loss_after,
        "mse_before": result.mse_before,
        "mse_after": result.mse_after,
        "psnr_before": result.psnr_before,
        "psnr_after": result.psnr_after,
        "sam_before": result.sam_before,
        "sam_after": result.sam_after,
        "component_grad_norm_max": result.component_grad_norm_max,
        "component_parameter_changed": result.component_parameter_changed,
        "psf_requires_grad": result.psf_requires_grad,
        "loss_requires_grad": result.loss_requires_grad,
        "evidence_level": result.evidence_level,
        "claim_ceiling": result.claim_ceiling,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "loss_trace.json").write_text(
        json.dumps(loss_trace, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    parameter_trace = {
        "after": [float(p.detach().cpu().reshape(())) for p in params],
        "parameter_names": result.psf_summary.get("parameter_names", []),
    }
    (out_dir / "component_parameter_trace.json").write_text(
        json.dumps(parameter_trace, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez(
        out_dir / "psf_artifact.npz",
        psf_before=before["psf"].detach().cpu().numpy(),
        psf_after=after["psf"].detach().cpu().numpy(),
    )
    manifest = {
        "run_id": result.run_id,
        "component_type": result.component_type,
        "artifacts": [
            "spec.json",
            "result.json",
            "metrics.json",
            "loss_trace.json",
            "component_parameter_trace.json",
            "psf_artifact.npz",
            "report.md",
        ],
        "claim_ceiling": result.claim_ceiling,
    }
    (out_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(_report_text(result), encoding="utf-8")
    return [str(path.relative_to(Path("workspace"))) for path in sorted(out_dir.iterdir()) if path.is_file()]


def _report_text(result: ComponentSurrogateHSICoDesignResult) -> str:
    return "\n".join([
        f"# Component Surrogate HSI Co-design: {result.component_type}",
        "",
        f"- status: {result.status}",
        f"- evidence_level: {result.evidence_level}",
        f"- claim_ceiling: {result.claim_ceiling}",
        f"- reconstruction_loss_before: {result.reconstruction_loss_before}",
        f"- reconstruction_loss_after: {result.reconstruction_loss_after}",
        f"- mse_before: {result.mse_before}",
        f"- mse_after: {result.mse_after}",
        f"- psnr_before: {result.psnr_before}",
        f"- psnr_after: {result.psnr_after}",
        f"- sam_before: {result.sam_before}",
        f"- sam_after: {result.sam_after}",
        f"- component_grad_norm_max: {result.component_grad_norm_max}",
        f"- component_parameter_changed: {result.component_parameter_changed}",
        f"- psf_requires_grad: {result.psf_requires_grad}",
        f"- loss_requires_grad: {result.loss_requires_grad}",
        "",
        "Claim boundary: component surrogate HSI co-design only; not full GeoLens lens-level optimization.",
    ]) + "\n"


def _float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(()))
    return float(value)
