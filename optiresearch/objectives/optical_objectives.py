"""Optical-only objective functions — pure torch, differentiable."""

from __future__ import annotations

from typing import Any, Optional

import torch

from optiresearch.memory.schemas import StrictModel


def psf_width_loss(psf: torch.Tensor, target_width: float = 1.0) -> torch.Tensor:
    """Penalise deviation of PSF spatial width from a target value.

    Args:
        psf: PSF tensor of shape (..., H, W).
        target_width: desired RMS width in pixels.
    """
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
    return ((rms - target_width) ** 2).mean()


def psf_centroid_loss(
    psf: torch.Tensor,
    target_cy: float = 0.0,
    target_cx: float = 0.0,
) -> torch.Tensor:
    """Penalise PSF centroid drift from a target position (normalised coords)."""
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
    return ((cy - centre_y) ** 2 + (cx - centre_x) ** 2).mean()


def psf_energy_loss(psf: torch.Tensor, target_energy: float = 1.0) -> torch.Tensor:
    """Penalise deviation of total PSF energy from target."""
    energy = psf.sum(dim=(-2, -1))
    return ((energy - target_energy) ** 2).mean()


def psf_smoothness_loss(psf: torch.Tensor) -> torch.Tensor:
    """Total-variation smoothness regularizer on PSF."""
    dy = torch.diff(psf, dim=-2)
    dx = torch.diff(psf, dim=-1)
    return dy.abs().mean() + dx.abs().mean()


def spot_size_loss(psf: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Penalise the number of pixels above a fractional intensity threshold."""
    peak = psf.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
    peak = peak.clamp_min(1e-8)
    above = (psf / peak > threshold).float()
    return above.mean()


def field_consistency_loss(psf_cube: torch.Tensor) -> torch.Tensor:
    """Penalise variation of PSF shape across the field (wavelength/field dim)."""
    if psf_cube.ndim < 4:
        return torch.tensor(0.0, device=psf_cube.device, dtype=psf_cube.dtype)
    ref = psf_cube[0:1].detach()
    diff = psf_cube - ref
    return (diff**2).mean()


# ── Objective profiles ──────────────────────────────────────────────


class ObjectiveProfile(StrictModel):
    """Named collection of losses with weights and compatibility metadata."""

    profile_id: str
    losses: list[str]
    weights: dict[str, float]
    compatible_backends: list[str]
    claim_implications: Optional[str] = None
    description: str = ""
    metadata: dict[str, Any] = {}


_profile_registry: dict[str, ObjectiveProfile] = {}


PRESET_PROFILES: dict[str, ObjectiveProfile] = {
    "stable_lens_hsi_codesign": ObjectiveProfile(
        profile_id="stable_lens_hsi_codesign",
        losses=["reconstruction_mse", "spectral_angle_loss", "psf_energy_preservation"],
        weights={
            "reconstruction_mse": 1.0,
            "spectral_angle_loss": 0.05,
            "psf_energy_preservation": 0.1,
        },
        compatible_backends=["deeplens_geolens_geometric"],
        claim_implications="stable_native_lens_hsi_codesign",
        description="Phase 23 recommended stable training config: small LR + rollback",
    ),
    "psf_quality_probe": ObjectiveProfile(
        profile_id="psf_quality_probe",
        losses=["psf_width_loss", "psf_centroid_loss", "psf_energy_loss"],
        weights={"psf_width_loss": 1.0, "psf_centroid_loss": 0.5, "psf_energy_loss": 0.5},
        compatible_backends=[
            "mock_deeplens",
            "deeplens_blackbox_source_psf",
            "deeplens_geolens_geometric",
        ],
        claim_implications="deeplens_integration_smoke",
        description="Basic PSF quality evaluation",
    ),
    "component_optimization": ObjectiveProfile(
        profile_id="component_optimization",
        losses=["psf_width_loss", "psf_energy_loss", "optical_param_l2"],
        weights={"psf_width_loss": 1.0, "psf_energy_loss": 0.5, "optical_param_l2": 1e-3},
        compatible_backends=["deeplens_fresnel_component", "deeplens_binary2phase_component"],
        claim_implications="native_component_optimization",
        description="Component-level optical surface optimization",
    ),
}


def _seed_profiles() -> None:
    if not _profile_registry:
        for pid, p in PRESET_PROFILES.items():
            _profile_registry[pid] = p


def list_objective_profiles() -> list[ObjectiveProfile]:
    _seed_profiles()
    return list(_profile_registry.values())


def get_objective_profile(profile_id: str) -> Optional[ObjectiveProfile]:
    _seed_profiles()
    return _profile_registry.get(profile_id)


def register_objective_profile(profile: ObjectiveProfile) -> None:
    _seed_profiles()
    _profile_registry[profile.profile_id] = profile
