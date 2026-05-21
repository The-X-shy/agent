"""Tests for FresnelHSIBridge."""

import torch
from optiresearch.adapters.deeplens_differentiable_bridge import FresnelHSIBridge


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


def test_fresnel_bridge_psf_requires_grad():
    bridge = FresnelHSIBridge(device="cpu")
    bridge.surface = FakeFresnelSurface(f0=50.0, res=48)

    psf = bridge.psf_from_component_torch(num_bands=4, psf_size=16)
    assert psf.requires_grad is True
    assert psf.shape == (4, 16, 16)


def test_fresnel_bridge_loss_backward_reaches_f0():
    bridge = FresnelHSIBridge(device="cpu")
    bridge.surface = FakeFresnelSurface(f0=50.0, res=48)

    psf = bridge.psf_from_component_torch(num_bands=4, psf_size=16)
    loss = psf.pow(2).mean()
    loss.backward()

    f0_param = bridge.get_trainable_parameters()[0]
    assert f0_param.grad is not None
    assert f0_param.grad.abs().sum() > 0


def test_fresnel_bridge_optimizer_step_changes_f0():
    bridge = FresnelHSIBridge(device="cpu")
    bridge.surface = FakeFresnelSurface(f0=50.0, res=48)
    optimizer = bridge.get_optimizer(learning_rate=1.0)

    before = bridge.parameter_snapshot()["f0"]

    psf = bridge.psf_from_component_torch(num_bands=4, psf_size=16)
    loss = psf.pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    after = bridge.parameter_snapshot()["f0"]
    assert abs(after - before) > 1e-8


def test_fresnel_bridge_snapshot_keys():
    bridge = FresnelHSIBridge(device="cpu")
    bridge.surface = FakeFresnelSurface(f0=50.0, res=48)
    snap = bridge.parameter_snapshot()
    assert "f0" in snap
    assert isinstance(snap["f0"], float)


def test_fresnel_bridge_gradient_norm():
    bridge = FresnelHSIBridge(device="cpu")
    bridge.surface = FakeFresnelSurface(f0=50.0, res=48)

    psf = bridge.psf_from_component_torch(num_bands=4, psf_size=16)
    loss = psf.pow(2).mean()
    loss.backward()

    gn = bridge.gradient_norm()
    assert gn > 0


def test_fresnel_bridge_no_surface_raises():
    bridge = FresnelHSIBridge(device="cpu")
    try:
        bridge.psf_from_component_torch()
        assert False, "Should have raised"
    except RuntimeError as exc:
        assert "No surface built" in str(exc)
