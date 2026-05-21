"""Tests for Phase 19B surface-level DeepLens optimization probes."""

from __future__ import annotations

import json

import torch

from optiresearch.runtime.deeplens_surface_optimization_probe import (
    run_surface_optimization_probe,
)
from optiresearch.schemas.surface_optimization import (
    SurfaceOptimizationProbeSpec,
    make_surface_probe_id,
)


class FakeFresnel:
    def __init__(self, d=0.0, f0=50.0, res=32, device="cpu", **_kwargs):
        self.f0 = torch.tensor(float(f0), device=device)
        x = torch.linspace(-1.0, 1.0, int(res), device=device)
        self.grid = x[:, None] ** 2 + x[None, :] ** 2

    def phase_func(self):
        return -self.grid / self.f0

    def get_optimizer_params(self, lr=0.01):
        self.f0.requires_grad = True
        return [{"params": [self.f0], "lr": lr}]

    def get_optimizer(self, lr=0.01):
        return torch.optim.Adam(self.get_optimizer_params(lr=lr))


class FakeBinary2Phase:
    def __init__(
        self,
        r=5.0,
        d=0.0,
        order2=1.0,
        order4=0.2,
        order6=0.0,
        order8=0.0,
        order10=0.0,
        order12=0.0,
        device="cpu",
        **_kwargs,
    ):
        self.d = torch.tensor(float(d), device=device)
        self.order2 = torch.tensor(float(order2), device=device)
        self.order4 = torch.tensor(float(order4), device=device)
        self.order6 = torch.tensor(float(order6), device=device)
        self.order8 = torch.tensor(float(order8), device=device)
        self.order10 = torch.tensor(float(order10), device=device)
        self.order12 = torch.tensor(float(order12), device=device)

    def phi(self, x, y):
        r2 = x * x + y * y
        return self.d * 0.01 + self.order2 * r2 + self.order4 * r2 * r2

    def get_optimizer_params(self, lrs=[0.01, 0.01], optim_mat=False):
        del optim_mat
        params = []
        for name, lr in [
            ("d", lrs[0]),
            ("order2", lrs[1]),
            ("order4", lrs[1]),
            ("order6", lrs[1]),
            ("order8", lrs[1]),
            ("order10", lrs[1]),
            ("order12", lrs[1]),
        ]:
            value = getattr(self, name)
            value.requires_grad = True
            params.append({"params": [value], "lr": lr})
        return params

    def get_optimizer(self, lrs=[0.01, 0.01]):
        return torch.optim.Adam(self.get_optimizer_params(lrs=lrs))


def test_fresnel_surface_probe_executes_backward_step_and_writes_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_import(surface_class):
        assert surface_class == "Fresnel"
        return FakeFresnel, "fake.Fresnel", None

    monkeypatch.setattr(
        "optiresearch.runtime.deeplens_surface_optimization_probe._import_surface_class",
        fake_import,
    )
    spec = SurfaceOptimizationProbeSpec(
        probe_id=make_surface_probe_id("Fresnel", "minimize_phase_variance"),
        surface_class="Fresnel",
        objective="minimize_phase_variance",
        max_steps=3,
        learning_rate=0.05,
        save_artifacts=True,
    )

    result = run_surface_optimization_probe(spec)

    assert result.status == "succeeded"
    assert result.differentiable is True
    assert result.gradient_norm is not None and result.gradient_norm > 0
    assert result.parameters_changed is True
    out_dir = tmp_path / "workspace" / "native_optimization" / f"surface_probe_{spec.probe_id}"
    assert (out_dir / "probe_result.json").exists()
    payload = json.loads((out_dir / "probe_result.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["optimizer_step_executed"] is True
    assert (out_dir / "phase_before.npz").exists()
    assert (out_dir / "phase_after.npz").exists()


def test_binary2phase_surface_probe_tracks_named_parameter_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_import(surface_class):
        assert surface_class == "Binary2Phase"
        return FakeBinary2Phase, "fake.Binary2Phase", None

    monkeypatch.setattr(
        "optiresearch.runtime.deeplens_surface_optimization_probe._import_surface_class",
        fake_import,
    )
    spec = SurfaceOptimizationProbeSpec(
        probe_id=make_surface_probe_id("Binary2Phase", "match_target_phase"),
        surface_class="Binary2Phase",
        objective="match_target_phase",
        max_steps=3,
        learning_rate=0.05,
        save_artifacts=True,
    )

    result = run_surface_optimization_probe(spec)

    assert result.status == "succeeded"
    assert result.trainable_params[:2] == ["d", "order2"]
    assert result.metadata["requires_grad_true"] is True
    assert result.metadata["parameter_before"]["order2"] != result.metadata["parameter_after"]["order2"]
    assert result.loss_after is not None
