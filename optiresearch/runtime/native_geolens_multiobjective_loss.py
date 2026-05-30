"""Multi-objective loss for native GeoLens HSI stabilization.

Unifies reconstruction MSE/MAE, real SAM (acos), measurement consistency,
PSF energy/centroid/width regularization, optical parameter delta, and
spatial smoothness into a single weighted loss.

All terms are pure torch — no .item(), no .detach() in the computation graph.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_WEIGHTS: dict[str, float] = {
    "reconstruction_mse": 1.0,
    "reconstruction_mae": 0.0,
    "spectral_angle": 0.2,
    "measurement_consistency": 0.1,
    "psf_energy": 0.1,
    "psf_centroid": 0.1,
    "psf_width": 0.05,
    "optical_param_delta": 1e-4,
    "smoothness": 0.0,
}


def compute_native_geolens_multiobjective_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    measurement: torch.Tensor | None,
    psf: torch.Tensor,
    psf_initial: torch.Tensor,
    optical_params: list[torch.Tensor],
    optical_params_initial: list[torch.Tensor],
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute weighted multi-objective loss with all terms.

    Args:
        recon: Reconstructed HSI [N, B, H, W]
        target: Ground truth HSI [N, B, H, W]
        measurement: Measured image [N, H, W] or None
        psf: Current PSF cube [B, K, K]
        psf_initial: Initial (pre-training) PSF cube [B, K, K]
        optical_params: Current optical parameter tensors
        optical_params_initial: Initial (pre-training) optical parameters
        weights: Optional weight overrides

    Returns:
        Dict with individual loss terms + "total_loss" key (scalar, grad attached)
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    terms: dict[str, torch.Tensor] = {}
    dev = recon.device

    # Reconstruction losses
    terms["reconstruction_mse"] = F.mse_loss(recon, target)
    terms["reconstruction_mae"] = F.l1_loss(recon, target)
    terms["spectral_angle"] = _spectral_angle_loss(recon, target)

    # Measurement consistency
    if measurement is not None:
        from optiresearch.hsi.differentiable_proxy import make_measurement_from_psf_torch

        meas_forward = make_measurement_from_psf_torch(recon, psf)
        if meas_forward.dim() == 4 and meas_forward.shape[1] == 1:
            meas_forward = meas_forward.squeeze(1)
        terms["measurement_consistency"] = F.mse_loss(meas_forward, measurement)
    else:
        terms["measurement_consistency"] = torch.tensor(0.0, device=dev)

    # PSF regularization (against initial PSF as reference)
    terms["psf_energy"] = _psf_energy_reg(psf, psf_initial)
    terms["psf_centroid"] = _psf_centroid_reg(psf, psf_initial)
    terms["psf_width"] = _psf_width_reg(psf, psf_initial)

    # Optical parameter delta regularization
    param_delta = torch.tensor(0.0, device=dev)
    for p, p0 in zip(optical_params, optical_params_initial):
        param_delta = param_delta + (p - p0.detach()).abs().mean()
    terms["optical_param_delta"] = param_delta

    # Spatial smoothness (Tikhonov on recon spatial gradients)
    dy = recon[:, :, 1:, :] - recon[:, :, :-1, :]
    dx = recon[:, :, :, 1:] - recon[:, :, :, :-1]
    terms["smoothness"] = (dy * dy).mean() + (dx * dx).mean()

    total = torch.tensor(0.0, device=dev)
    for k, v in terms.items():
        weight = w.get(k, 0.0)
        if weight > 0:
            total = total + weight * v

    terms["total_loss"] = total
    return terms


def _spectral_angle_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Real SAM via acos (not 1-cosine proxy)."""
    eps = 1e-8
    r_norm = recon / (recon.norm(dim=1, keepdim=True) + eps)
    t_norm = target / (target.norm(dim=1, keepdim=True) + eps)
    cos_sim = (r_norm * t_norm).sum(dim=1).clamp(-1 + eps, 1 - eps)
    return cos_sim.acos().mean()


def _psf_energy_reg(psf: torch.Tensor, psf_initial: torch.Tensor) -> torch.Tensor:
    """Penalize deviation in per-band PSF energy from initial."""
    per_band = psf.sum(dim=(-2, -1))
    per_band_init = psf_initial.detach().sum(dim=(-2, -1))
    return (per_band - per_band_init).abs().mean()


def _psf_centroid_reg(psf: torch.Tensor, psf_initial: torch.Tensor) -> torch.Tensor:
    """Penalize centroid shift from initial PSF."""
    B, H, W = psf.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=psf.dtype, device=psf.device),
        torch.arange(W, dtype=psf.dtype, device=psf.device),
        indexing="ij",
    )
    mass = psf.sum(dim=(-2, -1)) + 1e-8
    cy = (psf * yy).sum(dim=(-2, -1)) / mass
    cx = (psf * xx).sum(dim=(-2, -1)) / mass

    mass_init = psf_initial.detach().sum(dim=(-2, -1)) + 1e-8
    cy_init = (psf_initial.detach() * yy).sum(dim=(-2, -1)) / mass_init
    cx_init = (psf_initial.detach() * xx).sum(dim=(-2, -1)) / mass_init

    return ((cy - cy_init).abs() + (cx - cx_init).abs()).mean()


def _psf_width_reg(psf: torch.Tensor, psf_initial: torch.Tensor) -> torch.Tensor:
    """Penalize PSF width (second moment) deviation from initial."""
    B, H, W = psf.shape
    half_h, half_w = H // 2, W // 2
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=psf.dtype, device=psf.device) - half_h,
        torch.arange(W, dtype=psf.dtype, device=psf.device) - half_w,
        indexing="ij",
    )
    r2 = yy * yy + xx * xx
    mass = psf.sum(dim=(-2, -1)) + 1e-8
    moment2 = (psf * r2).sum(dim=(-2, -1)) / mass

    mass_init = psf_initial.detach().sum(dim=(-2, -1)) + 1e-8
    moment2_init = (psf_initial.detach() * r2).sum(dim=(-2, -1)) / mass_init

    return (moment2 - moment2_init).abs().mean()
