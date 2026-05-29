import torch

from optiresearch.adapters.deeplens_geolens_params import (
    activate_geolens_trainable_parameters,
    flatten_optimizer_param_groups,
)


def test_flatten_optimizer_param_groups_accepts_scalar_tensor_params():
    param = torch.tensor(1.0, requires_grad=True)

    params = flatten_optimizer_param_groups([{"params": param, "lr": 1e-6}])

    assert params == [param]


def test_flatten_optimizer_param_groups_accepts_sequences_and_deduplicates():
    p1 = torch.tensor(1.0, requires_grad=True)
    p2 = torch.tensor(2.0, requires_grad=True)
    frozen = torch.tensor(3.0, requires_grad=False)

    params = flatten_optimizer_param_groups([
        {"params": [p1, frozen]},
        {"params": (p2, p1)},
    ])

    assert params == [p1, p2]


class _NativeGeoLensWithoutModuleParameters:
    def __init__(self):
        self.param = torch.tensor(0.5, requires_grad=False)
        self.calls = []

    def get_optimizer_params(self, lrs=None, optim_mat=False):
        self.calls.append({"lrs": lrs, "optim_mat": optim_mat})
        self.param.requires_grad_(True)
        return [{"params": self.param, "lr": (lrs or [1e-6])[0]}]


def test_activate_geolens_trainable_parameters_uses_native_optimizer_api():
    geolens = _NativeGeoLensWithoutModuleParameters()

    groups, params = activate_geolens_trainable_parameters(geolens, lrs=[1e-7, 1e-7, 0.0, 0.0])

    assert len(groups) == 1
    assert params == [geolens.param]
    assert geolens.param.requires_grad is True
    assert geolens.calls == [{"lrs": [1e-7, 1e-7, 0.0, 0.0], "optim_mat": False}]
