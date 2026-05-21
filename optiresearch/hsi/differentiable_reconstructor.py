"""Trainable differentiable HSI reconstructors for Phase 21.

Pure torch nn.Module implementations. No numpy, no detach, no no_grad
on the loss path. Replace Phase 20's fixed reconstruct_proxy_torch()
with learnable modules that can be jointly optimized with optics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_psf_condition_features(psf: torch.Tensor) -> torch.Tensor:
    """Extract differentiable PSF condition features without breaking autograd.

    psf: [B, K, K] -- per-band PSF stack
    Returns: [B, 4] -- per-band features: energy, center, moment2, spectral_centroid
    """
    b, k, _ = psf.shape
    half = k // 2
    center_region = psf[:, half - 1 : half + 2, half - 1 : half + 2]

    energy = psf.sum(dim=(1, 2))
    center = center_region.sum(dim=(1, 2))

    yy, xx = torch.meshgrid(
        torch.arange(k, dtype=psf.dtype, device=psf.device) - half,
        torch.arange(k, dtype=psf.dtype, device=psf.device) - half,
        indexing="ij",
    )
    r2 = yy * yy + xx * xx
    moment2 = (psf * r2).sum(dim=(1, 2)) / (energy + 1e-8)

    band_indices = torch.arange(b, dtype=psf.dtype, device=psf.device)
    spectral_centroid = (energy * band_indices).sum() / (energy.sum() + 1e-8)

    features = torch.stack([energy, center, moment2, spectral_centroid.expand(b)], dim=1)
    return features


def _expand_psf_features_to_spatial(
    psf_features: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Expand [B, C_psf] to [1, B*C_psf, H, W] for CNN input."""
    if psf_features.dim() == 2:
        b, c = psf_features.shape
        spatial = psf_features.view(1, b * c, 1, 1).expand(-1, -1, height, width)
        return spatial
    return psf_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)


class DifferentiableLinearHSIReconstructor(nn.Module):
    """Linear projection reconstructor: measurement -> 1x1 weight -> recon HSI.

    Input: measurement [N, 1, H, W] + psf [B, K, K]
    Output: recon_hsi [N, B, H, W]

    PSF condition features modulate per-band reconstruction weights via a
    learnable Linear layer, preserving the autograd chain from PSF -> features
    -> weights -> recon -> loss -> back to PSF parameters.
    """

    def __init__(self, bands: int = 31, psf_feature_dim: int = 4):
        super().__init__()
        self.bands = bands
        self.psf_feature_dim = psf_feature_dim
        self.psf_to_weight = nn.Linear(psf_feature_dim * bands, bands)
        self.fallback_weight = nn.Parameter(torch.ones(bands, 1, 1, 1) * 0.1)

    def forward(self, measurement: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
        n, _, h, w = measurement.shape
        psf_features = build_psf_condition_features(psf)
        flat_features = psf_features.reshape(1, -1)
        weights = self.psf_to_weight(flat_features)
        weights = weights.view(1, self.bands, 1, 1) + self.fallback_weight.view(
            1, self.bands, 1, 1
        )
        meas_expanded = measurement.expand(-1, self.bands, -1, -1)
        recon = meas_expanded * weights
        return recon


class TinyDifferentiableHSIReconstructor(nn.Module):
    """Lightweight CNN reconstructor: measurement + PSF features -> recon HSI.

    Architecture:
        Conv2d(1 + B*C_psf, 32, 3, padding=1) -> ReLU
        Conv2d(32, 64, 3, padding=1) -> ReLU
        Conv2d(64, bands, 3, padding=1)

    PSF features are broadcast spatially and concatenated with the measurement
    as additional input channels.
    """

    def __init__(self, bands: int = 31, psf_feature_dim: int = 4):
        super().__init__()
        self.bands = bands
        in_channels = 1 + bands * psf_feature_dim
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, bands, 3, padding=1)

    def forward(self, measurement: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
        n, _, h, w = measurement.shape
        psf_features = build_psf_condition_features(psf)
        psf_spatial = _expand_psf_features_to_spatial(psf_features, h, w)
        psf_spatial = psf_spatial.expand(n, -1, -1, -1)

        x = torch.cat([measurement, psf_spatial], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


def hsi_reconstruction_losses(
    recon: torch.Tensor,
    target: torch.Tensor,
    measurement: torch.Tensor | None = None,
    psf: torch.Tensor | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute full differentiable reconstruction losses.

    Returns dict with: mse_loss, spectral_angle_loss_proxy,
    measurement_consistency_loss, total_loss
    """
    w = weights or {"mse": 1.0, "spectral_angle": 0.05, "measurement_consistency": 0.1}

    mse = F.mse_loss(recon, target)

    recon_norm = F.normalize(recon, p=2, dim=1)
    target_norm = F.normalize(target, p=2, dim=1)
    spectral_angle = (1.0 - (recon_norm * target_norm).sum(dim=1).mean()).clamp(min=0)

    total = w["mse"] * mse + w["spectral_angle"] * spectral_angle

    if measurement is not None and psf is not None and w.get("measurement_consistency", 0) > 0:
        from optiresearch.hsi.differentiable_proxy import make_measurement_from_psf_torch

        meas_from_recon = make_measurement_from_psf_torch(recon, psf)
        meas_consistency = F.mse_loss(meas_from_recon, measurement)
        total = total + w["measurement_consistency"] * meas_consistency
    else:
        meas_consistency = torch.tensor(0.0, device=recon.device)

    return {
        "mse_loss": mse,
        "spectral_angle_loss_proxy": spectral_angle,
        "measurement_consistency_loss": meas_consistency,
        "total_loss": total,
    }
