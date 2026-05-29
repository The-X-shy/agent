import torch

from optiresearch.adapters.geolens_waveoptics_bridge import GeoLensWaveOpticsBridge


class _ScalarParamGeoLens:
    def __init__(self):
        self.param = torch.tensor(0.25, requires_grad=False)
        self.default_dtype_seen = None
        self.points_dtype_seen = None

    def get_optimizer_params(self, lrs=None, optim_mat=False):
        self.param.requires_grad_(True)
        return [{"params": self.param, "lr": (lrs or [1e-6])[0]}]

    def get_optimizer(self, lrs=None, optim_mat=False):
        return torch.optim.SGD(self.get_optimizer_params(lrs=lrs, optim_mat=optim_mat), lr=1e-6)

    def psf(self, points, wvln=0.55, ks=9, model="geometric"):
        self.default_dtype_seen = torch.get_default_dtype()
        self.points_dtype_seen = points.dtype
        x = torch.linspace(-1.0, 1.0, ks, dtype=points.dtype, device=points.device)
        grid = x[:, None] ** 2 + x[None, :] ** 2
        return torch.exp(-grid * (1.0 + self.param))


def test_bridge_discovers_scalar_tensor_param_groups():
    bridge = GeoLensWaveOpticsBridge(device="cpu")
    fake = _ScalarParamGeoLens()
    bridge.geolens = fake

    params = bridge.get_trainable_parameters()

    assert params == [fake.param]
    assert fake.param.requires_grad is True


def test_bridge_geometric_psf_uses_float32_and_preserves_autograd():
    bridge = GeoLensWaveOpticsBridge(device="cpu")
    fake = _ScalarParamGeoLens()
    bridge.geolens = fake
    bridge.get_trainable_parameters()

    psf = bridge.psf_from_component_torch(ks=9, model="geometric")
    loss = (psf * psf).sum()
    loss.backward()

    assert fake.default_dtype_seen == torch.float32
    assert fake.points_dtype_seen == torch.float32
    assert psf.requires_grad is True
    assert fake.param.grad is not None
    assert float(fake.param.grad.abs().max().item()) > 0.0
