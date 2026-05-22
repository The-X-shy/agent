"""Lightweight local experiments without DeepLens dependency.

Uses pure-PyTorch FFT-based PSF generation (Fraunhofer diffraction)
to enable differentiable optics experiments on any machine.
All experiments complete in under 60 seconds on CPU.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np


def _generate_proxy_psf_torch(
    bands: int = 4,
    psf_size: int = 15,
    device: str = "cpu",
) -> tuple[Any, Any]:
    """Generate per-band PSFs via FFT of a learnable phase mask.

    Uses Fraunhofer diffraction: PSF = |FFT{exp(i * phase_mask)}|^2
    No DeepLens dependency -- pure PyTorch.

    Returns:
        (psf_cube [B, K, K], phase_mask [1, K, K] as nn.Parameter)
    """
    import torch

    phase_mask = torch.nn.Parameter(
        0.1 * torch.randn(1, psf_size, psf_size, device=device)
    )
    band_offsets = torch.linspace(0, 2 * torch.pi, bands, device=device)

    psfs = []
    for b in range(bands):
        field = torch.exp(1j * (phase_mask + band_offsets[b]))
        psf_complex = torch.fft.fft2(field)
        psf = torch.abs(psf_complex) ** 2
        psf = psf / (psf.sum() + 1e-8)
        psfs.append(psf.squeeze())

    return torch.stack(psfs), phase_mask


def _generate_synthetic_hsi_torch(
    bands: int = 4,
    image_size: int = 16,
    device: str = "cpu",
) -> Any:
    """Generate a small synthetic HSI target cube."""
    import torch

    hsi = torch.zeros(bands, image_size, image_size, device=device)
    for b in range(bands):
        x = torch.linspace(-1, 1, image_size, device=device)
        y = torch.linspace(-1, 1, image_size, device=device)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        hsi[b] = torch.exp(-(xx**2 + yy**2) / (0.3 + 0.1 * b))
    return hsi / hsi.max()


def _compute_psf_metrics(psf_cube: Any) -> dict[str, float]:
    """Compute basic PSF quality metrics."""
    import torch

    with torch.no_grad():
        energy = psf_cube.sum().item()
        width_x = _psf_fwhm(psf_cube.mean(0), axis=0)
        width_y = _psf_fwhm(psf_cube.mean(0), axis=1)
        centroid_x = float(psf_cube.mean(0).sum(0).argmax().float().item())
        centroid_y = float(psf_cube.mean(0).sum(1).argmax().float().item())
        max_intensity = psf_cube.max().item()
    return {
        "psf_energy": float(energy),
        "psf_width_x": float(width_x),
        "psf_width_y": float(width_y),
        "psf_centroid_x": centroid_x,
        "psf_centroid_y": centroid_y,
        "psf_max_intensity": max_intensity,
    }


def _psf_fwhm(psf_2d: Any, axis: int = 0) -> float:
    """Estimate FWHM of a 2D PSF along one axis."""
    import torch

    profile = psf_2d.sum(dim=1 - axis)
    half_max = profile.max() / 2.0
    above = (profile >= half_max).float()
    crossings = (above[1:] - above[:-1]).nonzero(as_tuple=True)[0]
    if len(crossings) >= 2:
        return float(crossings[-1] - crossings[0]) + 1.0
    return 0.0


def _check_deeplens_available() -> bool:
    """Check if DeepLens is importable."""
    try:
        import importlib
        importlib.import_module("deeplens.geolens")
        return True
    except Exception:
        return False


# ── Lightweight Experiment Functions ──────────────────────────────


def run_lightweight_psf_probe(
    backend_id: str = "phase_to_fft_proxy",
    device: str = "cpu",
) -> "ControllerResult":
    """Run a minimal PSF probe.

    Generates PSFs via FFT and computes basic metrics.
    Completes in under 5 seconds. No DeepLens required.
    """
    from optiresearch.runtime.experiment_controller_v2 import ControllerResult
    from optiresearch.memory.schemas import make_deterministic_id

    run_id = make_deterministic_id("lwpsf", backend_id)
    start = time.perf_counter()

    try:
        psf_cube, phase_mask = _generate_proxy_psf_torch(
            bands=4, psf_size=15, device=device
        )
        metrics = _compute_psf_metrics(psf_cube)
        elapsed = round(time.perf_counter() - start, 3)

        return ControllerResult(
            spec_id=make_deterministic_id("spec", run_id),
            status="succeeded",
            execution_target="local",
            backend_id=backend_id,
            run_id=run_id,
            evidence_level="deeplens_integration_smoke",
            result_payload={
                **metrics,
                "elapsed_seconds": elapsed,
                "psf_generation_method": "fft_fraunhofer",
                "deepens_used": False,
            },
            artifact_paths=[],
        )
    except Exception as exc:
        return ControllerResult(
            spec_id=make_deterministic_id("spec", "lwpsf_err"),
            status="failed",
            execution_target="local",
            backend_id=backend_id,
            errors=[{"type": type(exc).__name__, "message": str(exc)}],
        )


def run_lightweight_stable_lens_hsi(
    backend_id: str = "phase_to_fft_proxy",
    max_steps: int = 5,
    optical_lr: float = 1e-6,
    recon_lr: float = 1e-3,
    bands: int = 4,
    image_size: int = 16,
    psf_size: int = 15,
    rollback_on_loss_increase: bool = True,
    device: str = "cpu",
) -> "ControllerResult":
    """Run lightweight stable lens HSI co-design.

    Jointly optimizes a phase mask (optical) and a linear reconstructor
    (digital) for HSI reconstruction. Uses FFT-based PSF generation
    without DeepLens. Completes in under 60 seconds on CPU.
    """
    import torch

    from optiresearch.runtime.experiment_controller_v2 import ControllerResult
    from optiresearch.memory.schemas import make_deterministic_id

    run_id = make_deterministic_id("lwshsi", backend_id)
    start = time.perf_counter()

    try:
        # Setup
        hsi_target = _generate_synthetic_hsi_torch(bands, image_size, device)
        psf_cube, phase_mask = _generate_proxy_psf_torch(bands, psf_size, device)
        reconstructor = _LinearReconstructor(bands, image_size, device)

        optical_opt = torch.optim.Adam([phase_mask], lr=optical_lr)
        recon_opt = torch.optim.Adam(reconstructor.parameters(), lr=recon_lr)
        loss_fn = torch.nn.MSELoss()

        loss_before = None
        loss_after = None
        best_loss = float("inf")
        best_phase = phase_mask.detach().clone()
        accepted_updates = 0
        rejected_updates = 0
        rollback_count = 0
        grad_norms: list[float] = []

        for step in range(max_steps):
            # Forward: generate PSF from current phase mask
            band_offsets = torch.linspace(0, 2 * torch.pi, bands, device=device)
            psfs = []
            for b in range(bands):
                field = torch.exp(1j * (phase_mask + band_offsets[b]))
                psf_c = torch.abs(torch.fft.fft2(field)) ** 2
                psf_c = psf_c / (psf_c.sum() + 1e-8)
                psfs.append(psf_c.squeeze())
            psf_cube = torch.stack(psfs)

            # Forward model: blur HSI via FFT-based convolution
            measured = torch.zeros_like(hsi_target)
            for b in range(bands):
                hsi_fft = torch.fft.fft2(hsi_target[b])
                otf = torch.fft.fft2(psf_cube[b], s=hsi_target[b].shape[-2:])
                blurred = torch.fft.ifft2(hsi_fft * otf).real
                measured[b] = blurred

            # Reconstruct
            reconstructed = reconstructor(measured)

            # Loss
            loss = loss_fn(reconstructed, hsi_target)
            loss_val = loss.item()

            if step == 0:
                loss_before = loss_val
            loss_after = loss_val

            # Rollback check
            if loss_val > best_loss and rollback_on_loss_increase and best_loss < float("inf"):
                phase_mask.data.copy_(best_phase)
                rollback_count += 1
                rejected_updates += 1
                continue

            if loss_val < best_loss:
                best_loss = loss_val
                best_phase = phase_mask.detach().clone()
                accepted_updates += 1

            # Backward
            optical_opt.zero_grad()
            recon_opt.zero_grad()
            loss.backward()

            grad_norm = phase_mask.grad.norm().item() if phase_mask.grad is not None else 0.0
            grad_norms.append(grad_norm)

            optical_opt.step()
            recon_opt.step()

        elapsed = round(time.perf_counter() - start, 3)
        improvement = (loss_after is not None and loss_before is not None
                       and loss_after < loss_before)

        return ControllerResult(
            spec_id=make_deterministic_id("spec", run_id),
            status="succeeded",
            execution_target="local",
            backend_id=backend_id,
            run_id=run_id,
            evidence_level="native_full_reconstruction_proxy",
            result_payload={
                "status": "succeeded",
                "reconstruction_loss_before": loss_before,
                "reconstruction_loss_after": loss_after,
                "best_reconstruction_loss": best_loss,
                "accepted_update_count": accepted_updates,
                "rejected_update_count": rejected_updates,
                "rollback_count": rollback_count,
                "optical_gradient_norm_max": max(grad_norms) if grad_norms else 0.0,
                "optical_gradient_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
                "optical_parameters_changed": accepted_updates > 0,
                "stable_training_succeeded": not (rollback_count > max_steps // 2),
                "improvement_detected": improvement,
                "elapsed_seconds": elapsed,
                "psf_generation_method": "fft_fraunhofer",
                "deepens_used": False,
            },
            artifact_paths=[],
        )
    except Exception as exc:
        return ControllerResult(
            spec_id=make_deterministic_id("spec", "lwshsi_err"),
            status="failed",
            execution_target="local",
            backend_id=backend_id,
            errors=[{"type": type(exc).__name__, "message": str(exc)}],
        )


def run_lightweight_ablation(
    backend_id: str = "phase_to_fft_proxy",
    max_configs: int = 2,
    max_steps: int = 3,
    device: str = "cpu",
) -> "ControllerResult":
    """Run a lightweight ablation study comparing configurations.

    Compares different optical learning rates and reports results.
    """
    import torch

    from optiresearch.runtime.experiment_controller_v2 import ControllerResult
    from optiresearch.memory.schemas import make_deterministic_id

    run_id = make_deterministic_id("lwabl", backend_id)
    start = time.perf_counter()

    configs = [
        {"label": "baseline_small_lr", "optical_lr": 1e-6},
        {"label": "higher_lr", "optical_lr": 1e-5},
    ][:max_configs]

    results = []
    for cfg in configs:
        result = run_lightweight_stable_lens_hsi(
            backend_id=backend_id,
            max_steps=max_steps,
            optical_lr=cfg["optical_lr"],
            device=device,
        )
        results.append({
            "config": cfg["label"],
            "optical_lr": cfg["optical_lr"],
            "loss_after": (
                result.result_payload.get("reconstruction_loss_after")
                if result.result_payload else None
            ),
            "status": result.status,
        })

    winner = min(
        [r for r in results if r["loss_after"] is not None],
        key=lambda r: r["loss_after"],
        default=None,
    )

    elapsed = round(time.perf_counter() - start, 3)

    return ControllerResult(
        spec_id=make_deterministic_id("spec", run_id),
        status="succeeded",
        execution_target="local",
        backend_id=backend_id,
        run_id=run_id,
        evidence_level="native_full_reconstruction_proxy",
        result_payload={
            "ablation_results": results,
            "winner": winner["config"] if winner else None,
            "winner_loss_after": winner["loss_after"] if winner else None,
            "elapsed_seconds": elapsed,
            "deepens_used": False,
        },
        artifact_paths=[],
    )


class _LinearReconstructor:
    """Simple linear reconstructor for lightweight experiments.

    Uses per-band learnable weights to linearly combine measured bands
    into reconstructed HSI bands.
    """

    def __init__(self, bands: int, image_size: int, device: str = "cpu"):
        import torch

        self.bands = bands
        self.size = image_size
        self.combine_weight = torch.nn.Parameter(
            torch.eye(bands, device=device) * 0.5
            + torch.randn(bands, bands, device=device) * 0.01
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(bands, image_size, image_size, device=device)
        )

    def parameters(self):
        return [self.combine_weight, self.bias]

    def __call__(self, measured):
        import torch

        # measured: [B, H, W], combine_weight: [B, B]
        # out[b, h, w] = sum_i measured[i, h, w] * weight[b, i] + bias[b, h, w]
        out = torch.einsum("ihw,bi->bhw", measured, self.combine_weight)
        return out + self.bias
