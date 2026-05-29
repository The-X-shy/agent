from optiresearch.runtime.deeplens_trainable_parameter_inspection import (
    inspect_deeplens_trainable_parameters,
)


def test_inspection_returns_structured_result():
    result = inspect_deeplens_trainable_parameters(device="cpu")
    assert "parameter_count" in result
    assert "evidence_level" in result


def test_inspection_uses_geolens_native_optimizer_params(monkeypatch, tmp_path):
    import importlib
    import torch

    lens_file = tmp_path / "cooke.json"
    lens_file.write_text("{}")
    param = torch.tensor(0.5, requires_grad=False)

    class _FakeGeoLens:
        def __init__(self, lens_path, device="cpu"):
            self.lens_path = lens_path
            self.device = device

        def get_optimizer_params(self, lrs=None, optim_mat=False):
            param.requires_grad_(True)
            return [{"params": param, "lr": (lrs or [1e-6])[0]}]

        def psf(self, points, wvln=0.55, ks=9, model="geometric"):
            x = torch.linspace(-1.0, 1.0, ks, device=points.device, dtype=points.dtype)
            grid = x[:, None] ** 2 + x[None, :] ** 2
            return torch.exp(-grid * (1.0 + param))

    class _FakeModule:
        GeoLens = _FakeGeoLens

    original_import = importlib.import_module

    def _fake_import(name, package=None):
        if name == "deeplens.geolens":
            return _FakeModule()
        return original_import(name, package=package)

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    monkeypatch.setattr(
        "optiresearch.optics.lens_file_resolver.resolve_lens_file",
        lambda lens_file, backend_id: type(
            "Resolution",
            (),
            {
                "exists": True,
                "resolved_path": str(tmp_path / "cooke.json"),
                "source": "test",
                "checked_paths": [str(tmp_path / "cooke.json")],
            },
        )(),
    )

    result = inspect_deeplens_trainable_parameters(device="cpu")

    assert result["status"] == "succeeded"
    assert result["parameter_count"] == 1
    assert result["trainable_count"] == 1
    assert result["psf_requires_grad"] is True
    assert result["params_with_grad"] == 1
    assert result["grad_norm_max"] > 0.0
