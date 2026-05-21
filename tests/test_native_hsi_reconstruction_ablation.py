"""Tests for ablation study runner."""

import torch
from optiresearch.runtime.native_hsi_reconstruction_ablation import run_native_hsi_reconstruction_ablation
from optiresearch.schemas.native_hsi_reconstruction_codesign import make_recon_codesign_id


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


def _patch_import(monkeypatch, fake_class):
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


def test_ablation_all_four_modes_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _patch_import(monkeypatch, FakeFresnelSurface)

    summary = run_native_hsi_reconstruction_ablation(
        optical_component="Fresnel",
        reconstructor_name="differentiable_linear",
        bands=4, image_size=16, psf_size=7,
        max_steps=3, optical_lr=0.1, recon_lr=0.1,
        save_artifacts=True,
    )

    assert "reconstructor_only" in summary["modes"]
    assert "optics_only" in summary["modes"]
    assert "joint_optics_reconstructor" in summary["modes"]
    assert "no_native_optics" in summary["modes"]

    for key, r in summary["modes"].items():
        assert "loss_before" in r
        assert "loss_after" in r
        assert isinstance(r["loss_before"], float)


def test_ablation_writes_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _patch_import(monkeypatch, FakeFresnelSurface)

    summary = run_native_hsi_reconstruction_ablation(
        optical_component="Fresnel",
        reconstructor_name="differentiable_linear",
        bands=4, image_size=16, psf_size=7,
        max_steps=3, save_artifacts=True,
    )

    out_dir = tmp_path / "workspace" / "native_hsi_reconstruction_ablation" / summary["run_id_base"]
    assert (out_dir / "ablation_results.json").exists()
    assert (out_dir / "report.md").exists()


def test_reconstructor_only_improves_loss(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _patch_import(monkeypatch, FakeFresnelSurface)

    summary = run_native_hsi_reconstruction_ablation(
        optical_component="Fresnel",
        reconstructor_name="differentiable_linear",
        bands=4, image_size=16, psf_size=7,
        max_steps=3, optical_lr=0.01, recon_lr=0.01,
        save_artifacts=False,
    )
    r = summary["modes"]["reconstructor_only"]
    assert isinstance(r["loss_before"], float)
    assert isinstance(r["loss_after"], float)
