import torch

from optiresearch.runtime.stable_native_lens_hsi_loop import run_stable_native_lens_hsi_codesign
from optiresearch.schemas.stable_native_lens_hsi import StableNativeLensHSISpec


class _DifferentiableGeoLensBridge:
    deeplens_native_psf_path = "geolens.psf_geometric"

    def __init__(self, device="cpu"):
        self.device = device
        self.param = torch.nn.Parameter(torch.tensor(0.15, device=device))

    def build_component(self, lens_file=None):
        return object()

    def get_trainable_parameters(self):
        return [self.param]

    def get_optimizer(self, learning_rate=1e-3):
        return torch.optim.SGD([self.param], lr=learning_rate)

    def parameter_snapshot(self):
        return {"param_0": float(self.param.detach().cpu().item())}

    def psf_cube_torch(self, num_bands=4, ks=9):
        x = torch.linspace(-1.0, 1.0, ks, device=self.device)
        grid = x[:, None] ** 2 + x[None, :] ** 2
        psf = torch.exp(-grid * (1.0 + self.param))
        psf = psf / psf.sum()
        return psf.expand(num_bands, ks, ks)


def test_stable_native_lens_result_records_geolens_graph_fields(monkeypatch):
    from optiresearch.runtime import stable_native_lens_hsi_loop as module

    monkeypatch.setattr(module, "GeoLensWaveOpticsBridge", _DifferentiableGeoLensBridge)

    spec = StableNativeLensHSISpec(
        run_id="stable-native-geolens-graph-fields",
        candidate="GeoLensCooke",
        reconstructor="differentiable_linear",
        max_steps=2,
        optical_warmup_steps=0,
        image_size=8,
        bands=3,
        psf_size=9,
        optical_lr=1e-3,
        recon_lr=1e-2,
        rollback_on_loss_increase=False,
        save_artifacts=False,
    )

    result = run_stable_native_lens_hsi_codesign(spec)

    assert result.trainable_param_count == 1
    assert result.params_with_grad == 1
    assert result.graph_connected is True
    assert result.psf_requires_grad is True
    assert result.loss_requires_grad is True
    assert result.optical_parameters_changed is True
