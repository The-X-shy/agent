"""Tests for full reconstruction HSI co-design loop."""

import torch
from optiresearch.runtime.native_hsi_reconstruction_codesign_loop import (
    run_native_hsi_reconstruction_codesign,
)
from optiresearch.schemas.native_hsi_reconstruction_codesign import (
    NativeHSIReconstructionCoDesignSpec,
    make_recon_codesign_id,
)


class FakeFresnelSurface:
    def __init__(self, d=0.0, f0=50.0, res=48, device="cpu", **_kwargs):
        self.f0 = torch.nn.Parameter(torch.tensor(float(f0), device=device))
        x = torch.linspace(-1.0, 1.0, int(res), device=device)
        self.grid = x[:, None] ** 2 + x[None, :] ** 2

    def phase_func(self):
        return -self.grid / self.f0

    def get_optimizer_params(self, lr=0.01):
        self.f0.requires_grad = True
        return [{"params": [self.f0], "lr": lr}]

    def get_optimizer(self, lr=0.01):
        return torch.optim.Adam(self.get_optimizer_params(lr=lr))


def _patch_bridge_import(monkeypatch, fake_class):
    import importlib as _il
    _orig = _il.import_module

    def _fake_import(path, package=None):
        if path in ("deeplens.diffractive_surface.fresnel", "deeplens.phase_surface.binary2"):
            name = path.split(".")[-1]
            cls_name = {"fresnel": "Fresnel", "binary2": "Binary2Phase"}.get(name, "Fresnel")
            return type("mod", (), {cls_name: fake_class})()
        return _orig(path, package=package)

    monkeypatch.setattr(
        "optiresearch.adapters.deeplens_differentiable_bridge.importlib.import_module",
        _fake_import,
    )


def test_fresnel_linear_recon_codesign_succeeds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _patch_bridge_import(monkeypatch, FakeFresnelSurface)

    spec = NativeHSIReconstructionCoDesignSpec(
        run_id=make_recon_codesign_id("Fresnel", "differentiable_linear"),
        optical_component="Fresnel",
        reconstructor="differentiable_linear",
        bands=4, image_size=16, psf_size=7, batch_size=1,
        max_steps=3, optical_lr=0.5, recon_lr=0.5,
        save_artifacts=True,
    )
    result = run_native_hsi_reconstruction_codesign(spec)

    assert result.status == "succeeded"
    assert result.differentiable is True
    assert result.full_reconstruction_loss_used is True
    assert result.optical_gradient_norm is not None and result.optical_gradient_norm > 0
    assert result.recon_gradient_norm is not None and result.recon_gradient_norm > 0
    assert result.optical_parameters_changed is True
    assert result.optimizer_step_executed is True
    assert result.evidence_level == "native_full_reconstruction_proxy"
    assert result.reconstruction_loss_before is not None
    assert result.reconstruction_loss_after is not None
    assert result.mse_before is not None and result.mse_after is not None
    assert result.psnr_before is not None and result.psnr_after is not None

    out_dir = tmp_path / "workspace" / "native_hsi_reconstruction_codesign" / spec.run_id
    assert (out_dir / "result.json").exists()
    assert (out_dir / "loss_trace.json").exists()
    assert (out_dir / "metrics.json").exists()


def test_unsupported_component_is_caught_before_reconstructor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = NativeHSIReconstructionCoDesignSpec(
        run_id="test-bad-comp",
        optical_component="GeoLensCooke",
        reconstructor="differentiable_linear",
    )
    result = run_native_hsi_reconstruction_codesign(spec)
    assert result.status == "unsupported"
    assert result.error_code == "UNSUPPORTED_COMPONENT"


def test_unsupported_component_returns_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = NativeHSIReconstructionCoDesignSpec(
        run_id="test-bad-comp",
        optical_component="GeoLensCooke",
        reconstructor="differentiable_linear",
    )
    result = run_native_hsi_reconstruction_codesign(spec)
    assert result.status == "unsupported"
    assert result.error_code == "UNSUPPORTED_COMPONENT"
