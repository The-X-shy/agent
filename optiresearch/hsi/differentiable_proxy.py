"""Pure-torch differentiable HSI proxy loss for Phase 20.

No numpy, no detach, no no_grad on the loss path.
All operations must preserve the autograd chain from PSF -> measurement -> recon -> loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def generate_torch_synthetic_hsi(
    batch: int = 1,
    bands: int = 31,
    height: int = 32,
    width: int = 32,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate a synthetic HSI cube [N, B, H, W] with smooth spectral variation."""
    hsi = torch.zeros(batch, bands, height, width, device=device)
    for b in range(bands):
        wavelength_norm = b / max(bands - 1, 1)
        amp = 0.5 + 0.5 * torch.sin(torch.tensor(wavelength_norm * torch.pi, device=device))
        y = torch.linspace(-1.0, 1.0, height, device=device)
        x = torch.linspace(-1.0, 1.0, width, device=device)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        pattern = torch.sin((b + 1) * torch.pi * gy) * torch.cos((b + 1) * torch.pi * gx)
        hsi[0, b] = amp * (0.5 + 0.5 * pattern)
    return hsi


def make_measurement_from_psf_torch(
    hsi: torch.Tensor,
    psf: torch.Tensor,
) -> torch.Tensor:
    """Render a grayscale measurement from HSI and per-band PSFs.

    hsi: [N, B, H, W]
    psf: [B, K, K] where K is PSF size
    Returns: measurement [N, 1, H, W]

    For each band, convolve hsi with that band's PSF and sum across bands.
    """
    n, b, h, w = hsi.shape
    k = psf.shape[-1]
    pad = k // 2

    psf_kernels = psf.view(b, 1, k, k)
    # groups=b: each HSI band independently convolved with its PSF kernel
    convolved = F.conv2d(hsi, psf_kernels, padding=pad, groups=b)
    measurement = convolved.sum(dim=1, keepdim=True)
    return measurement


def reconstruct_proxy_torch(
    measurement: torch.Tensor,
    psf: torch.Tensor,
    bands: int,
) -> torch.Tensor:
    """Simple differentiable reconstruction proxy using PSF-matched filtering.

    measurement: [N, 1, H, W]
    psf: [B, K, K]
    Returns: recon [N, B, H, W]

    Uses transposed convolution as a matched filter for each band.
    """
    n, _, h, w = measurement.shape
    k = psf.shape[-1]
    pad = k // 2

    psf_kernels = psf.view(bands, 1, k, k)
    meas_expanded = measurement.expand(-1, bands, -1, -1)

    # groups=bands: each band independently transposed-convolved with its PSF
    recon = F.conv_transpose2d(meas_expanded, psf_kernels, padding=pad, groups=bands)
    return recon


def hsi_proxy_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mode: str = "mse",
) -> torch.Tensor:
    """Compute differentiable HSI proxy loss.

    recon, target: [N, B, H, W]
    mode: "mse" | "measurement_consistency"
    """
    if mode == "measurement_consistency":
        return recon.pow(2).mean()
    return F.mse_loss(recon, target)
