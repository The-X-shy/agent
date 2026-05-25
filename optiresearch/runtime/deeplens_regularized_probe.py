"""DeepLens Regularized Probe for Phase 54."""

from __future__ import annotations
from typing import Any


def run_deeplens_regularized_probe(max_steps: int = 3, device: str = "cpu") -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "succeeded", "evidence_level": "diagnostic_evidence",
        "base_loss": None, "regularized_loss": None, "reg_terms": {},
        "grad_norm_before": 0.0, "grad_norm_after": 0.0,
        "update_accepted": False, "claim_ceiling": "diagnostic_evidence",
    }
    try:
        import torch
        from optiresearch.runtime.lightweight_experiments import (
            _generate_synthetic_hsi_torch, _LinearReconstructor,
        )
        hsi = _generate_synthetic_hsi_torch(4, 16, device)
        phase_mask = torch.nn.Parameter(0.1 * torch.randn(1, 15, 15, device=device))
        recon = _LinearReconstructor(4, 16, device)
        opt = torch.optim.Adam([phase_mask, *recon.parameters()], lr=1e-6)
        for step in range(max_steps):
            bo = torch.linspace(0, 6.28, 4, device=device)
            psfs = [torch.abs(torch.fft.fft2(torch.exp(1j * (phase_mask + b)))) ** 2 for b in bo]
            psf_cube = torch.stack([p.squeeze() / (p.sum() + 1e-8) for p in psfs])
            measured = torch.stack([
                torch.fft.ifft2(torch.fft.fft2(hsi[b]) * torch.fft.fft2(psf_cube[b], s=hsi[b].shape[-2:])).real
                for b in range(4)
            ])
            mse_loss = torch.nn.MSELoss()(recon(measured), hsi)
            reg_energy = (psf_cube.sum(dim=(1, 2)) - 1.0).pow(2).mean()
            reg_centroid = (psf_cube.sum(dim=2).mean() - 7.5).pow(2).mean()
            loss = mse_loss + 0.01 * reg_energy + 0.01 * reg_centroid
            if step == 0:
                result["base_loss"] = float(mse_loss.item())
            opt.zero_grad()
            loss.backward()
            gn = float(phase_mask.grad.norm().item()) if phase_mask.grad is not None else 0.0
            if step == 0:
                result["grad_norm_before"] = gn
            opt.step()
            result["regularized_loss"] = float(loss.item())
            result["reg_terms"] = {"energy": float(reg_energy), "centroid": float(reg_centroid)}
        result["update_accepted"] = (result["regularized_loss"] or float("inf")) < (result["base_loss"] or float("inf"))
    except Exception:
        result["status"] = "unavailable"
    return result
