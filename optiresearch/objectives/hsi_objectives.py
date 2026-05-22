"""HSI reconstruction objective functions — pure torch, differentiable."""

from __future__ import annotations

import torch


def reconstruction_mse(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error between reconstructed and target HSI."""
    return ((recon - target) ** 2).mean()


def spectral_angle_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Spectral angle mapper loss — penalises spectral shape mismatch.

    Expects input shape (..., C, H, W) where C is the spectral dimension.
    """
    eps = 1e-8
    dot = (recon * target).sum(dim=-3, keepdim=True)
    norm_recon = recon.norm(dim=-3, keepdim=True).clamp_min(eps)
    norm_target = target.norm(dim=-3, keepdim=True).clamp_min(eps)
    cos_sim = dot / (norm_recon * norm_target)
    cos_sim = cos_sim.clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos_sim).mean()


def measurement_consistency_loss(
    recon: torch.Tensor,
    measurement: torch.Tensor,
    psf: torch.Tensor,
) -> torch.Tensor:
    """Penalise inconsistency between forward(rec) and actual measurement.

    Uses a simplified band-wise convolution via F.conv2d with groups.
    Inputs are assumed to be (..., C, H, W).
    """
    C = recon.shape[-3]
    # Flatten batch dims for conv2d: (N, C, H, W)
    recon_flat = recon.reshape(-1, C, *recon.shape[-2:])
    # PSF: expand to (C, 1, kH, kW) for grouped conv with groups=C
    if psf.ndim == 3:
        psf_kernel = psf.unsqueeze(1)  # (C, 1, kH, kW)
    elif psf.ndim == 4:
        psf_kernel = psf.unsqueeze(1)  # (N, 1, kH, kW) — take first
        psf_kernel = psf_kernel[0:1].expand(C, -1, -1, -1)
    else:
        psf_kernel = psf.reshape(C, 1, *psf.shape[-2:])

    sim = torch.nn.functional.conv2d(
        recon_flat,
        psf_kernel.to(device=recon_flat.device, dtype=recon_flat.dtype),
        padding="same",
        groups=C,
    )
    sim = sim.reshape(recon.shape)
    return ((sim - measurement) ** 2).mean()


def spectral_smoothness_loss(recon: torch.Tensor) -> torch.Tensor:
    """Penalise high-frequency variation along spectral dimension."""
    diff = torch.diff(recon, dim=-3)
    return (diff**2).mean()


def band_weighted_mse(
    recon: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Band-weighted MSE where weights tensor broadcasts over spatial dims."""
    while weights.ndim < recon.ndim:
        weights = weights.unsqueeze(-1)
    return (weights * (recon - target) ** 2).mean()


def task_aligned_hsi_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    task_weights: dict[str, float],
) -> torch.Tensor:
    """Combined task-aligned HSI loss with configurable component weights."""
    loss = torch.tensor(0.0, device=recon.device, dtype=recon.dtype)
    w_mse = task_weights.get("mse", 1.0)
    w_sam = task_weights.get("sam", 0.0)
    w_smooth = task_weights.get("spectral_smoothness", 0.0)
    if w_mse > 0:
        loss = loss + w_mse * reconstruction_mse(recon, target)
    if w_sam > 0:
        loss = loss + w_sam * spectral_angle_loss(recon, target)
    if w_smooth > 0:
        loss = loss + w_smooth * spectral_smoothness_loss(recon)
    return loss
