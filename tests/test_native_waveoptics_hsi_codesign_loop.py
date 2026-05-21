"""Tests for native waveoptics HSI co-design loop shape handling."""

from __future__ import annotations

import torch

from optiresearch.runtime.native_waveoptics_hsi_codesign_loop import run_native_waveoptics_hsi_codesign
from optiresearch.schemas.native_hsi_reconstruction_codesign import (
    NativeHSIReconstructionCoDesignSpec,
)


class _BatchedPsfCubeBridge:
    deeplens_native_psf_path = "geolens.psf_geometric"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.param = torch.nn.Parameter(torch.tensor(0.25, device=device))

    def build_component(self, lens_file=None):
        return object()

    def get_optimizer(self, learning_rate: float = 1e-3):
        return torch.optim.SGD([self.param], lr=learning_rate)

    def parameter_snapshot(self):
        return {"param_0": float(self.param.detach().cpu().item())}

    def psf_cube_torch(self, num_bands: int = 4, ks: int = 7):
        return self.param.expand(num_bands, 1, ks, ks)

    def get_trainable_parameters(self):
        return [self.param]


def test_native_waveoptics_hsi_codesign_accepts_batched_psf_cube(monkeypatch):
    from optiresearch.runtime import native_waveoptics_hsi_codesign_loop as module

    monkeypatch.setattr(module, "GeoLensWaveOpticsBridge", _BatchedPsfCubeBridge)

    spec = NativeHSIReconstructionCoDesignSpec(
        run_id="native_wave_hsi_batched_psf",
        optical_component="GeoLensCooke",
        reconstructor="differentiable_linear",
        dataset="synthetic",
        bands=4,
        image_size=16,
        psf_size=7,
        max_steps=2,
        optical_lr=0.1,
        recon_lr=0.1,
        save_artifacts=False,
    )

    result = run_native_waveoptics_hsi_codesign(spec)

    assert result.error_message is None or "too many values to unpack" not in result.error_message
    assert result.status == "succeeded"
    assert result.full_wave_optics is False
    assert result.phase_to_fft_proxy_used is False
    assert result.evidence_level == "native_lens_hsi_codesign"
