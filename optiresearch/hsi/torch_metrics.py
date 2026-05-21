"""Pure-torch HSI quality metrics. All functions accept [N, B, H, W] tensors.

No numpy, all operations preserve the computation graph for logging purposes.
For metric logging outside the training loop, call .detach() before .item().
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def torch_mse(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon, target)


def torch_psnr(recon: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(recon, target, reduction="none").mean(dim=(1, 2, 3))
    psnr = 20 * torch.log10(max_val / (mse.sqrt() + 1e-8))
    return psnr.mean()


def torch_sam(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Spectral Angle Mapper: mean angular difference in spectral dimension."""
    recon_n = F.normalize(recon, p=2, dim=1)
    target_n = F.normalize(target, p=2, dim=1)
    cos_sim = (recon_n * target_n).sum(dim=1).clamp(-1, 1)
    sam = torch.acos(cos_sim).mean()
    return sam


def torch_ergas_proxy(
    recon: torch.Tensor, target: torch.Tensor, ratio: float = 1.0
) -> torch.Tensor:
    """ERGAS proxy: relative dimensionless global error in synthesis."""
    n, b, h, w = recon.shape
    mse_per_band = F.mse_loss(recon, target, reduction="none").mean(dim=(0, 2, 3))
    mean_per_band = target.mean(dim=(0, 2, 3))
    normalized = (mse_per_band / (mean_per_band * mean_per_band + 1e-8)).sqrt()
    ergas = 100 * ratio * normalized.mean().sqrt()
    return ergas
