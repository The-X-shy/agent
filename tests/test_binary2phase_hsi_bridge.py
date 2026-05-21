"""Tests for Binary2PhaseHSIBridge."""

import torch
from optiresearch.adapters.deeplens_differentiable_bridge import Binary2PhaseHSIBridge


class FakeBinary2PhaseSurface:
    def __init__(
        self,
        r=5.0,
        d=0.0,
        order2=1.0,
        order4=0.2,
        order6=0.05,
        order8=0.0,
        order10=0.0,
        order12=0.0,
        device="cpu",
        **_kwargs,
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


def test_binary2phase_bridge_psf_requires_grad():
    bridge = Binary2PhaseHSIBridge(device="cpu")
    bridge.surface = FakeBinary2PhaseSurface()

    psf = bridge.psf_from_component_torch(num_bands=4, psf_size=16)
    assert psf.requires_grad is True
    assert psf.shape == (4, 16, 16)


def test_binary2phase_bridge_loss_backward_has_grads():
    bridge = Binary2PhaseHSIBridge(device="cpu")
    bridge.surface = FakeBinary2PhaseSurface()

    psf = bridge.psf_from_component_torch(num_bands=4, psf_size=16)
    loss = psf.pow(2).mean()
    loss.backward()

    # d, order2, order4 participate in phi(); order6 (unused) may have no grad
    params_list = bridge.get_trainable_parameters()
    grads_found = sum(1 for p in params_list if p.grad is not None and p.grad.abs().sum() > 0)
    assert grads_found >= 3


def test_binary2phase_bridge_optimizer_step_changes_params():
    bridge = Binary2PhaseHSIBridge(device="cpu")
    bridge.surface = FakeBinary2PhaseSurface()
    optimizer = bridge.get_optimizer(learning_rate=1.0)

    before = bridge.parameter_snapshot()

    psf = bridge.psf_from_component_torch(num_bands=4, psf_size=16)
    loss = psf.pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    after = bridge.parameter_snapshot()
    assert abs(after["order2"] - before["order2"]) > 1e-8


def test_binary2phase_bridge_all_trainable_params_in_snapshot():
    bridge = Binary2PhaseHSIBridge(device="cpu")
    bridge.surface = FakeBinary2PhaseSurface()
    snap = bridge.parameter_snapshot()
    for name in ["d", "order2", "order4", "order6"]:
        assert name in snap


def test_binary2phase_bridge_no_surface_raises():
    bridge = Binary2PhaseHSIBridge(device="cpu")
    try:
        bridge.psf_from_component_torch()
        assert False, "Should have raised"
    except RuntimeError as exc:
        assert "No surface built" in str(exc)
