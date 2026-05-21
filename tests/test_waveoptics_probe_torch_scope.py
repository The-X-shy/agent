"""Regression tests for Phase 22 waveoptics probe torch scoping."""

from __future__ import annotations

import torch

from optiresearch.schemas.deeplens_waveoptics_probe import DeepLensWaveOpticsProbeSpec


class _DifferentiableFakeBridge:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.param = torch.nn.Parameter(torch.tensor(0.25, device=device))

    def build_component(self, lens_file=None):
        return object()

    def get_optimizer(self, learning_rate: float = 1e-3):
        return torch.optim.SGD([self.param], lr=learning_rate)

    def parameter_snapshot(self):
        return {"param_0": float(self.param.detach().cpu().item())}

    def psf_from_component_torch(self, wvln: float = 0.55, ks: int = 8):
        return self.param.expand(ks, ks)

    def gradient_norm(self):
        if self.param.grad is None:
            return 0.0
        return float(self.param.grad.detach().norm().cpu().item())


class _BatchedPsfFakeBridge(_DifferentiableFakeBridge):
    def psf_from_component_torch(self, wvln: float = 0.55, ks: int = 8):
        return self.param.expand(1, ks, ks)


def test_waveoptics_probe_does_not_shadow_torch(monkeypatch):
    from optiresearch.runtime import deeplens_waveoptics_probe as module

    monkeypatch.setattr(module, "GeoLensWaveOpticsBridge", _DifferentiableFakeBridge)

    spec = DeepLensWaveOpticsProbeSpec(
        run_id="waveoptics_probe_torch_scope",
        candidate="GeoLensCooke",
        objective="minimize_psf_width",
        psf_size=8,
        max_steps=1,
        learning_rate=0.1,
        save_artifacts=False,
    )
    result = module.run_deeplens_waveoptics_probe(spec)

    assert result.error_message is None or "local variable 'torch'" not in result.error_message
    assert result.status == "succeeded"
    assert result.full_wave_optics is False
    assert result.phase_to_fft_proxy_used is False
    assert result.differentiable is True
    assert result.optical_gradient_norm and result.optical_gradient_norm > 0
    assert result.optical_parameters_changed is True
    assert result.deeplens_native_wave_path == "geolens.psf_geometric"
    assert result.evidence_level == "native_lens_simulation"


def test_waveoptics_probe_handles_batched_psf_shape(monkeypatch):
    from optiresearch.runtime import deeplens_waveoptics_probe as module

    monkeypatch.setattr(module, "GeoLensWaveOpticsBridge", _BatchedPsfFakeBridge)

    spec = DeepLensWaveOpticsProbeSpec(
        run_id="waveoptics_probe_batched_psf",
        candidate="GeoLensCooke",
        objective="minimize_psf_width",
        psf_size=8,
        max_steps=1,
        learning_rate=0.1,
        save_artifacts=False,
    )
    result = module.run_deeplens_waveoptics_probe(spec)

    assert result.error_message is None or "out of bounds" not in result.error_message
    assert result.status == "succeeded"
    assert result.differentiable is True
    assert result.evidence_level == "native_lens_simulation"


def test_waveoptics_probe_reports_structured_unavailable(monkeypatch):
    from optiresearch.runtime import deeplens_waveoptics_probe as module

    class MissingDeepLensBridge(_DifferentiableFakeBridge):
        def build_component(self, lens_file=None):
            raise ImportError("DeepLens unavailable")

    monkeypatch.setattr(module, "GeoLensWaveOpticsBridge", MissingDeepLensBridge)

    spec = DeepLensWaveOpticsProbeSpec(
        run_id="waveoptics_probe_missing_deeplens",
        candidate="GeoLensCooke",
        objective="minimize_psf_width",
        save_artifacts=False,
    )
    result = module.run_deeplens_waveoptics_probe(spec)

    assert result.status == "unsupported"
    assert result.error_code == "BUILD_FAILED"
    assert "DeepLens unavailable" in (result.error_message or "")
    assert "local variable 'torch'" not in (result.error_message or "")
