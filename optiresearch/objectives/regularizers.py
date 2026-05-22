"""Regularization functions for optical optimization — pure torch."""

from __future__ import annotations

import torch


def optical_param_l2(
    params: list[torch.Tensor],
    weight: float = 1e-4,
) -> torch.Tensor:
    """L2 penalty on optical parameters."""
    if not params:
        return torch.tensor(0.0)
    total = torch.tensor(0.0, device=params[0].device, dtype=params[0].dtype)
    for p in params:
        total = total + (p**2).sum()
    return weight * total


def optical_param_delta_limit(
    params_before: list[torch.Tensor],
    params_after: list[torch.Tensor],
    max_delta: float = 1e-3,
) -> torch.Tensor:
    """Penalise optical parameter changes exceeding max_delta."""
    if not params_before:
        return torch.tensor(0.0)
    total = torch.tensor(0.0, device=params_before[0].device, dtype=params_before[0].dtype)
    for pb, pa in zip(params_before, params_after):
        delta = (pa - pb).abs()
        excess = (delta - max_delta).clamp_min(0)
        total = total + (excess**2).sum()
    return total


def psf_energy_preservation(
    psf: torch.Tensor,
    target_energy: float = 1.0,
    weight: float = 0.1,
) -> torch.Tensor:
    """Regularizer that encourages PSF energy conservation."""
    energy = psf.sum(dim=(-2, -1))
    return weight * ((energy - target_energy) ** 2).mean()


def psf_centroid_preservation(
    psf: torch.Tensor,
    target_cy: float = 0.0,
    target_cx: float = 0.0,
    weight: float = 0.1,
) -> torch.Tensor:
    """Regularizer that penalises PSF centroid drift."""
    H, W = psf.shape[-2], psf.shape[-1]
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=psf.dtype, device=psf.device),
        torch.arange(W, dtype=psf.dtype, device=psf.device),
        indexing="ij",
    )
    total = psf.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    cy = (psf * yy).sum(dim=(-2, -1)) / total
    cx = (psf * xx).sum(dim=(-2, -1)) / total
    centre_y = H / 2.0 + target_cy
    centre_x = W / 2.0 + target_cx
    return weight * ((cy - centre_y) ** 2 + (cx - centre_x) ** 2).mean()


def psf_width_preservation(
    psf: torch.Tensor,
    initial_width: float,
    weight: float = 0.05,
) -> torch.Tensor:
    """Regularizer that discourages PSF width from deviating from initial value."""
    H, W = psf.shape[-2], psf.shape[-1]
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=psf.dtype, device=psf.device),
        torch.arange(W, dtype=psf.dtype, device=psf.device),
        indexing="ij",
    )
    cy = H / 2.0
    cx = W / 2.0
    total = psf.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    r_sq = (yy - cy) ** 2 + (xx - cx) ** 2
    rms = torch.sqrt((psf * r_sq).sum(dim=(-2, -1)) / total)
    return weight * ((rms - initial_width) ** 2).mean()


def rollback_penalty(
    rollback_count: int,
    max_rollbacks: int = 3,
    penalty_weight: float = 0.1,
) -> torch.Tensor:
    """Soft penalty that grows with rollback count (non-differentiable signal).

    Returns a scalar tensor — intended to be added to the main loss to signal
    instability to any downstream hyperparameter optimiser.
    """
    if rollback_count <= max_rollbacks:
        return torch.tensor(0.0)
    excess = rollback_count - max_rollbacks
    return torch.tensor(penalty_weight * excess)
