"""Integration tests for NativeOpticalHSICoDesignLoop."""

import json

import torch

from optiresearch.runtime.native_hsi_codesign_loop import run_native_optical_hsi_codesign
from optiresearch.schemas.native_hsi_codesign import (
    NativeOpticalHSICoDesignSpec,
    make_hsi_codesign_id,
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


class FakeBinary2PhaseSurface:
    def __init__(
        self,
        r=5.0, d=0.0, order2=1.0, order4=0.2, order6=0.05,
        order8=0.0, order10=0.0, order12=0.0,
        device="cpu", **_kwargs,
    ):
        self.d = torch.nn.Parameter(torch.tensor(float(d), device=device))
        self.order2 = torch.nn.Parameter(torch.tensor(float(order2), device=device))
        self.order4 = torch.nn.Parameter(torch.tensor(float(order4), device=device))
        self.order6 = torch.nn.Parameter(torch.tensor(float(order6), device=device))

    def phi(self, x, y):
        r2 = x * x + y * y
        return self.d * 0.01 + self.order2 * r2 + self.order4 * r2 * r2

    def get_optimizer_params(self, lr=0.01):
        params = []
        for name in ["d", "order2", "order4", "order6"]:
            val = getattr(self, name)
            val.requires_grad = True
            params.append({"params": [val], "lr": lr})
        return params

    def get_optimizer(self, lr=0.01):
        return torch.optim.Adam(self.get_optimizer_params(lr=lr))


def _patch_bridge_import(monkeypatch, fake_class):
    import importlib as _il
    _original_import = _il.import_module

    def _fake_import(module_path, package=None):
        if module_path in (
            "deeplens.diffractive_surface.fresnel",
            "deeplens.phase_surface.binary2",
        ):
            name = module_path.split(".")[-1]
            cls_name = {"fresnel": "Fresnel", "binary2": "Binary2Phase"}.get(name, "Fresnel")
            mod = type("mod", (), {cls_name: fake_class})()
            return mod
        return _original_import(module_path, package=package)

    monkeypatch.setattr(
        "optiresearch.adapters.deeplens_differentiable_bridge.importlib.import_module",
        _fake_import,
    )


def test_fresnel_hsi_codesign_produces_valid_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _patch_bridge_import(monkeypatch, FakeFresnelSurface)

    spec = NativeOpticalHSICoDesignSpec(
        run_id=make_hsi_codesign_id("Fresnel", "minimize_hsi_proxy_loss"),
        optical_component="Fresnel",
        objective="minimize_hsi_proxy_loss",
        bands=4,
        image_size=16,
        psf_size=8,
        max_steps=1,
        learning_rate=0.5,
        save_artifacts=True,
    )
    result = run_native_optical_hsi_codesign(spec)

    assert result.status == "succeeded"
    assert result.differentiable is True
    assert result.gradient_norm is not None and result.gradient_norm > 0
    assert result.parameters_changed is True
    assert result.optimizer_step_executed is True
    assert result.autograd_break_detected is False
    assert result.hsi_loss_before is not None
    assert result.hsi_loss_after is not None
    assert result.evidence_level == "native_hsi_proxy"

    out_dir = tmp_path / "workspace" / "native_hsi_codesign" / spec.run_id
    assert (out_dir / "result.json").exists()
    assert (out_dir / "psf_before.npz").exists()
    assert (out_dir / "psf_after.npz").exists()
    assert (out_dir / "hsi_proxy_metrics.json").exists()
    assert (out_dir / "report.md").exists()


def test_binary2phase_hsi_codesign_produces_valid_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _patch_bridge_import(monkeypatch, FakeBinary2PhaseSurface)

    spec = NativeOpticalHSICoDesignSpec(
        run_id=make_hsi_codesign_id("Binary2Phase", "minimize_hsi_proxy_loss"),
        optical_component="Binary2Phase",
        objective="minimize_hsi_proxy_loss",
        bands=4,
        image_size=16,
        psf_size=8,
        max_steps=1,
        learning_rate=0.5,
        save_artifacts=True,
    )
    result = run_native_optical_hsi_codesign(spec)

    assert result.status == "succeeded"
    assert result.differentiable is True
    assert result.gradient_norm is not None and result.gradient_norm > 0
    assert result.parameters_changed is True
    assert result.optimizer_step_executed is True


def test_unsupported_component_returns_structured_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = NativeOpticalHSICoDesignSpec(
        run_id="test-unsupported",
        optical_component="GeoLensCooke",
        objective="minimize_hsi_proxy_loss",
    )
    result = run_native_optical_hsi_codesign(spec)

    assert result.status == "unsupported"
    assert result.error_code == "UNSUPPORTED_COMPONENT"
    assert result.differentiable is False


def test_build_failure_returns_unsupported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def _failing_import(module_path):
        raise ImportError("No module named deeplens")

    monkeypatch.setattr(
        "optiresearch.adapters.deeplens_differentiable_bridge.importlib.import_module",
        _failing_import,
    )

    spec = NativeOpticalHSICoDesignSpec(
        run_id="test-build-fail",
        optical_component="Fresnel",
        objective="minimize_hsi_proxy_loss",
    )
    result = run_native_optical_hsi_codesign(spec)

    assert result.status == "unsupported"
    assert result.error_code == "BUILD_FAILED"
