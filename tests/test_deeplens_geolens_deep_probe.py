"""Phase 32: Deep GeoLens geometric deep probe tests."""

import time
import importlib

import torch

from optiresearch.runtime.lightweight_experiments import (
    run_deeplens_geolens_geometric_deep_probe,
)


def test_deep_probe_returns_structured_result():
    result = run_deeplens_geolens_geometric_deep_probe(
        backend_id="deeplens_geolens_geometric",
    )
    assert result.status in ("succeeded", "failed")
    assert result.result_payload is not None
    assert "probe_depth" in result.result_payload
    assert result.result_payload.get("probe_depth") == "deep"


def test_deep_probe_never_crashes():
    result = run_deeplens_geolens_geometric_deep_probe(
        backend_id="deeplens_geolens_geometric",
    )
    assert result.result_payload is not None
    assert "probe_time_seconds" in result.result_payload
    assert "deeplens_available" in result.result_payload


def test_deep_probe_completes_quickly():
    start = time.perf_counter()
    result = run_deeplens_geolens_geometric_deep_probe(
        backend_id="deeplens_geolens_geometric",
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 30.0, f"Deep probe took {elapsed:.1f}s, expected <30s"


def test_deep_probe_reports_correct_fields():
    result = run_deeplens_geolens_geometric_deep_probe(
        backend_id="deeplens_geolens_geometric",
    )
    payload = result.result_payload or {}
    assert "probe_depth" in payload
    assert "deeplens_available" in payload
    if result.status == "succeeded":
        assert payload.get("full_wave_optics") is False
        assert payload.get("phase_to_fft_proxy_used") is False
        assert payload.get("deeplens_native_psf_path") == "geolens.psf_geometric"
        assert "optical_gradient_norm" in payload
        assert "parameters_changed" in payload
    else:
        assert "error_code" in payload


def test_deep_probe_uses_geolens_native_optimizer_params(monkeypatch, tmp_path):
    param = torch.tensor(0.2, requires_grad=False)

    class _FakeGeoLens:
        def __init__(self, lens_path, device="cpu"):
            self.lens_path = lens_path
            self.device = device

        def get_optimizer_params(self, lrs=None, optim_mat=False):
            param.requires_grad_(True)
            return [{"params": param, "lr": (lrs or [1e-6])[0]}]

        def get_optimizer(self, lrs=None, optim_mat=False):
            return torch.optim.SGD(self.get_optimizer_params(lrs=lrs, optim_mat=optim_mat), lr=1e-3)

        def psf(self, points, wvln=0.55, ks=9, model="geometric"):
            size = ks if isinstance(ks, int) else 9
            x = torch.linspace(-1.0, 1.0, size, dtype=points.dtype, device=points.device)
            grid = x[:, None] ** 2 + x[None, :] ** 2
            return torch.exp(-grid * (1.0 + param))

    class _FakeModule:
        GeoLens = _FakeGeoLens

    original_import = importlib.import_module

    def _fake_import(name, package=None):
        if name == "deeplens.geolens":
            return _FakeModule()
        return original_import(name, package=package)

    from optiresearch.runtime import lightweight_experiments as module

    monkeypatch.setattr(module, "_check_deeplens_available", lambda: True)
    monkeypatch.setattr(module, "_find_lens_file", lambda lens_name: str(tmp_path / lens_name))
    monkeypatch.setattr(importlib, "import_module", _fake_import)

    result = run_deeplens_geolens_geometric_deep_probe(
        backend_id="deeplens_geolens_geometric",
    )

    payload = result.result_payload or {}
    assert result.status == "succeeded"
    assert payload["differentiable"] is True
    assert payload["parameters_changed"] is True
    assert payload["trainable_param_count"] == 1
    assert payload["params_with_grad"] == 1
    assert payload["optical_gradient_norm"] > 0.0
