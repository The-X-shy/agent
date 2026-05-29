"""Differentiable surrogate PSFs from validated component parameter sets."""

from __future__ import annotations

from typing import Any

import torch

from optiresearch.schemas.component_surrogate_psf import (
    ComponentSurrogatePSFResult,
    ComponentSurrogatePSFSpec,
)

COMPONENT_SURROGATE_CLAIM = "component_surrogate_hsi_codesign"


def build_component_surrogate_psf(
    spec: ComponentSurrogatePSFSpec,
    initial_parameters: list[torch.nn.Parameter] | None = None,
) -> ComponentSurrogatePSFResult:
    """Build a differentiable [B, K, K] surrogate PSF from component parameters."""

    try:
        if spec.component_type == "fresnel":
            psf, params, names = _build_fresnel_psf(spec, initial_parameters)
        elif spec.component_type == "binary2phase":
            psf, params, names = _build_binary2phase_psf(spec, initial_parameters)
        elif spec.component_type == "diffractive_candidate":
            psf, params, names = _build_diffractive_candidate_psf(spec, initial_parameters)
        else:
            return _failed(spec, "UNKNOWN_COMPONENT", f"Unknown component: {spec.component_type}")
    except Exception as exc:
        return _failed(spec, "SURROGATE_PSF_BUILD_FAILED", str(exc))

    if spec.normalize_psf:
        psf = _normalize_psf(psf)

    psf_requires_grad = bool(getattr(psf, "requires_grad", False))
    if not psf_requires_grad:
        return ComponentSurrogatePSFResult(
            component_type=spec.component_type,
            status="failed",
            evidence_level="diagnostic_evidence",
            claim_ceiling="diagnostic_evidence",
            psf_shape=list(psf.shape),
            psf_requires_grad=False,
            parameter_count=len(params),
            trainable_param_count=sum(1 for p in params if p.requires_grad),
            component_parameter_changed=False,
            error_code="PSF_GRAPH_DISCONNECTED",
            error_message="Surrogate PSF tensor does not require gradients",
            errors=["Surrogate PSF tensor does not require gradients"],
            parameter_names=names,
            psf=psf,
            component_parameters=params,
        )

    grad_info = _probe_gradient_flow(psf, params)
    if grad_info["params_with_grad"] == 0:
        return ComponentSurrogatePSFResult(
            component_type=spec.component_type,
            status="failed",
            evidence_level="diagnostic_evidence",
            claim_ceiling="diagnostic_evidence",
            psf_shape=list(psf.shape),
            psf_requires_grad=True,
            parameter_count=len(params),
            trainable_param_count=sum(1 for p in params if p.requires_grad),
            params_with_grad=0,
            grad_norm_max=0.0,
            component_parameter_changed=False,
            psf_energy=_psf_energy(psf),
            psf_centroid=_psf_centroid(psf),
            psf_width=_psf_width(psf),
            error_code="PSF_GRAPH_DISCONNECTED",
            error_message="No component parameter received gradient from surrogate PSF",
            errors=["No component parameter received gradient from surrogate PSF"],
            parameter_names=names,
            psf=psf,
            component_parameters=params,
        )

    return ComponentSurrogatePSFResult(
        component_type=spec.component_type,
        status="succeeded",
        evidence_level=COMPONENT_SURROGATE_CLAIM,
        claim_ceiling=COMPONENT_SURROGATE_CLAIM,
        psf_shape=list(psf.shape),
        psf_requires_grad=True,
        parameter_count=len(params),
        trainable_param_count=sum(1 for p in params if p.requires_grad),
        params_with_grad=grad_info["params_with_grad"],
        grad_norm_max=grad_info["grad_norm_max"],
        component_parameter_changed=False,
        psf_energy=_psf_energy(psf),
        psf_centroid=_psf_centroid(psf),
        psf_width=_psf_width(psf),
        warnings=_warnings_for_spec(spec),
        parameter_names=names,
        metadata={
            "surrogate_model": spec.surrogate_model,
            "strict_native_component": spec.strict_native_component,
            "allow_proxy_psf": spec.allow_proxy_psf,
            "full_geolens_psf_used": False,
        },
        psf=psf,
        component_parameters=params,
    )


def _build_fresnel_psf(
    spec: ComponentSurrogatePSFSpec,
    initial_parameters: list[torch.nn.Parameter] | None,
) -> tuple[torch.Tensor, list[torch.nn.Parameter], list[str]]:
    device = torch.device(spec.device)
    if initial_parameters:
        params = initial_parameters
    else:
        params = [torch.nn.Parameter(torch.tensor(0.3, dtype=torch.float32, device=device))]
    f0 = params[0]
    yy, xx = _grid(spec.psf_size, device)
    r2 = xx * xx + yy * yy
    psfs: list[torch.Tensor] = []
    for scale in _band_scales(spec):
        width = 0.55 + torch.sigmoid(f0) * (1.2 + 0.15 * scale)
        gaussian = torch.exp(-r2 / (2.0 * width * width + 1e-8))
        phase = -scale * r2 / (torch.abs(f0) + 1.0)
        phase_modulation = 1.0 + 0.08 * torch.cos(phase)
        psfs.append(gaussian * phase_modulation)
    return torch.stack(psfs, dim=0), params, ["f0"]


def _build_binary2phase_psf(
    spec: ComponentSurrogatePSFSpec,
    initial_parameters: list[torch.nn.Parameter] | None,
) -> tuple[torch.Tensor, list[torch.nn.Parameter], list[str]]:
    device = torch.device(spec.device)
    names = ["d", "order2", "order4", "order6", "order8", "order10", "order12"]
    if initial_parameters:
        params = initial_parameters
    else:
        values = [0.10, 1.00, 0.35, 0.12, 0.05, 0.02, 0.01]
        params = [
            torch.nn.Parameter(torch.tensor(v, dtype=torch.float32, device=device))
            for v in values
        ]
    yy, xx = _grid(spec.psf_size, device)
    r2 = xx * xx + yy * yy
    powers = [1, 1, 2, 3, 4, 5, 6]
    base_phase = torch.zeros_like(r2)
    for param, power in zip(params, powers):
        base_phase = base_phase + param * torch.pow(r2 + 1e-6, power) / float(power)
    aperture = torch.exp(-1.4 * r2)
    psfs: list[torch.Tensor] = []
    for scale in _band_scales(spec):
        phase = base_phase * scale
        field = aperture * torch.exp(1j * phase)
        intensity = torch.abs(torch.fft.fftshift(torch.fft.fft2(field))) ** 2
        psfs.append(intensity)
    return torch.stack(psfs, dim=0), params, names


def _build_diffractive_candidate_psf(
    spec: ComponentSurrogatePSFSpec,
    initial_parameters: list[torch.nn.Parameter] | None,
) -> tuple[torch.Tensor, list[torch.nn.Parameter], list[str]]:
    device = torch.device(spec.device)
    if not spec.allow_proxy_psf:
        raise RuntimeError("Diffractive candidate needs follow-up without proxy PSF")
    if initial_parameters:
        params = initial_parameters
    else:
        params = [torch.nn.Parameter(torch.tensor(0.4, dtype=torch.float32, device=device))]
    phase_scale = params[0]
    yy, xx = _grid(spec.psf_size, device)
    r2 = xx * xx + yy * yy
    aperture = torch.exp(-1.2 * r2)
    psfs = []
    for scale in _band_scales(spec):
        phase = phase_scale * scale * (r2 + 0.25 * torch.sin(3.14159265 * xx))
        field = aperture * torch.exp(1j * phase)
        psfs.append(torch.abs(torch.fft.fftshift(torch.fft.fft2(field))) ** 2)
    return torch.stack(psfs, dim=0), params, ["phase_scale"]


def _grid(size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.linspace(-1.0, 1.0, int(size), device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return yy, xx


def _band_scales(spec: ComponentSurrogatePSFSpec) -> torch.Tensor:
    device = torch.device(spec.device)
    if spec.wavelengths_nm:
        wavelengths = torch.tensor(spec.wavelengths_nm, dtype=torch.float32, device=device)
        return wavelengths / wavelengths.mean()
    if spec.band_count == 1:
        return torch.ones(1, dtype=torch.float32, device=device)
    return torch.linspace(0.92, 1.08, spec.band_count, dtype=torch.float32, device=device)


def _normalize_psf(psf: torch.Tensor) -> torch.Tensor:
    return psf / (psf.sum(dim=(-2, -1), keepdim=True) + 1e-8)


def _probe_gradient_flow(psf: torch.Tensor, params: list[torch.nn.Parameter]) -> dict[str, Any]:
    k = psf.shape[-1]
    yy, xx = torch.meshgrid(
        torch.arange(k, dtype=psf.dtype, device=psf.device),
        torch.arange(k, dtype=psf.dtype, device=psf.device),
        indexing="ij",
    )
    center = (k - 1) / 2.0
    weights = 1.0 + ((xx - center) ** 2 + (yy - center) ** 2) / max(float(k * k), 1.0)
    probe_loss = (psf * weights).sum()
    grads = torch.autograd.grad(probe_loss, params, retain_graph=True, allow_unused=True)
    norms = [
        _float(g.detach().norm())
        for g in grads
        if g is not None and bool((g.detach().abs().sum() > 0).cpu())
    ]
    return {
        "params_with_grad": len(norms),
        "grad_norm_max": max(norms) if norms else 0.0,
    }


def _psf_energy(psf: torch.Tensor) -> list[float]:
    energy = psf.detach().sum(dim=(-2, -1)).cpu()
    return [_float(v) for v in energy]


def _psf_centroid(psf: torch.Tensor) -> list[float]:
    with torch.no_grad():
        mean_psf = psf.detach().mean(dim=0)
        k = mean_psf.shape[-1]
        coords = torch.arange(k, dtype=mean_psf.dtype, device=mean_psf.device)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        energy = mean_psf.sum() + 1e-8
        cx = (mean_psf * xx).sum() / energy
        cy = (mean_psf * yy).sum() / energy
    return [_float(cx.cpu()), _float(cy.cpu())]


def _psf_width(psf: torch.Tensor) -> float:
    with torch.no_grad():
        mean_psf = psf.detach().mean(dim=0)
        k = mean_psf.shape[-1]
        coords = torch.arange(k, dtype=mean_psf.dtype, device=mean_psf.device)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        centroid = _psf_centroid(psf)
        cx = torch.tensor(centroid[0], dtype=mean_psf.dtype, device=mean_psf.device)
        cy = torch.tensor(centroid[1], dtype=mean_psf.dtype, device=mean_psf.device)
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        width = torch.sqrt((mean_psf * r2).sum() / (mean_psf.sum() + 1e-8))
    return _float(width.cpu())


def _warnings_for_spec(spec: ComponentSurrogatePSFSpec) -> list[str]:
    warnings = [
        "surrogate_psf_not_full_geolens",
        "synthetic_component_parameterization",
    ]
    if spec.strict_native_component and spec.allow_proxy_psf:
        warnings.append("native_component_semantics_with_proxy_psf")
    return warnings


def _failed(spec: ComponentSurrogatePSFSpec, code: str, message: str) -> ComponentSurrogatePSFResult:
    return ComponentSurrogatePSFResult(
        component_type=spec.component_type,
        status="needs_followup" if code != "PSF_GRAPH_DISCONNECTED" else "failed",
        evidence_level="diagnostic_evidence",
        claim_ceiling="diagnostic_evidence",
        error_code=code,
        error_message=message,
        errors=[message],
    )


def _float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(()))
    return float(value)
