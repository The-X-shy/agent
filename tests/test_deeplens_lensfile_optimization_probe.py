"""Tests for Phase 19B lens-file optimization probe."""

from __future__ import annotations

from pathlib import Path

import torch

from optiresearch.runtime.deeplens_lensfile_optimization_probe import (
    run_lensfile_optimization_probe,
)


class FakeSurface:
    def __init__(self):
        self.p = torch.tensor(2.0)

    def get_optimizer_params(self, lr=0.05):
        self.p.requires_grad = True
        return [{"params": [self.p], "lr": lr}]


class FakeGeoLens:
    def __init__(self, filename, device="cpu", **_kwargs):
        self.filename = filename
        self.device = device
        self.surfaces = [FakeSurface()]

    def get_optimizer(self, lr=0.05, **_kwargs):
        return torch.optim.Adam(self.surfaces[0].get_optimizer_params(lr=lr))

    def psf(self, *args, **kwargs):
        del args, kwargs
        return self.surfaces[0].p * torch.ones(1, 1, 8, 8)


def test_lensfile_probe_finds_file_and_executes_native_step(tmp_path, monkeypatch):
    repo = tmp_path / "DeepLens"
    lens_dir = repo / "datasets" / "lenses"
    lens_dir.mkdir(parents=True)
    lens_file = lens_dir / "fake_geolens.json"
    lens_file.write_text('{"surfaces": []}', encoding="utf-8")

    def fake_import(lens_class):
        assert lens_class == "GeoLens"
        return FakeGeoLens, "fake.GeoLens", None

    monkeypatch.setattr(
        "optiresearch.runtime.deeplens_lensfile_optimization_probe._import_lens_class",
        fake_import,
    )
    monkeypatch.chdir(tmp_path)

    result = run_lensfile_optimization_probe(
        lens_class="GeoLens",
        repo_path=repo,
        max_files=5,
        max_steps=2,
        learning_rate=0.05,
        save_artifacts=True,
    )

    assert result["status"] == "succeeded"
    assert result["successful_file"] == str(lens_file)
    assert result["differentiable"] is True
    assert result["gradient_norm"] > 0
    assert result["parameters_changed"] is True
    out_dir = Path(result["output_dir"])
    assert (out_dir / "probe_result.json").exists()
    assert (out_dir / "loss_trace.json").exists()


def test_lensfile_probe_returns_structured_unsupported_when_no_files(tmp_path, monkeypatch):
    repo = tmp_path / "DeepLens"
    repo.mkdir()

    def fake_import(lens_class):
        return FakeGeoLens, "fake.GeoLens", None

    monkeypatch.setattr(
        "optiresearch.runtime.deeplens_lensfile_optimization_probe._import_lens_class",
        fake_import,
    )

    result = run_lensfile_optimization_probe(
        lens_class="GeoLens",
        repo_path=repo,
        max_files=5,
        max_steps=2,
        save_artifacts=False,
    )

    assert result["status"] == "unsupported"
    assert result["error_code"] == "NO_LENS_FILES_FOUND"
    assert result["attempts"] == []
