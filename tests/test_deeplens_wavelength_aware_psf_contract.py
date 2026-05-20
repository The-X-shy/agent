from pathlib import Path

import numpy as np

from optiresearch.adapters.deeplens import DeepLensAdapter
from optiresearch.agents.method_builder import MethodBuilder


class _FakeParaxialLens:
    def __init__(self, *args, **kwargs):
        pass

    def refocus(self, depth):
        self.depth = depth

    def psf(self, points, ks):
        return np.ones((ks, ks), dtype=np.float32)


class _FakeDeepLens:
    __file__ = __file__
    __version__ = "fake"
    ParaxialLens = _FakeParaxialLens


def test_deeplens_capability_model_includes_wavelength_aware_psf():
    environment = DeepLensAdapter(_FakeDeepLens()).validate_environment()
    names = set(environment["capability_names"])

    assert "wavelength_aware_psf_export_available" in names
    assert "native_wavelength_physics_available" in names
    assert "hsi_forward_compatible_psf_available" in names


def test_deeplens_simulation_writes_wavelength_aware_metrics_and_manifest(tmp_path):
    spec = MethodBuilder().build_mock_optical_spec("Design controlled HSI encoder", encoder_type="controlled_chromatic_edof", backend="deeplens")

    result = DeepLensAdapter(_FakeDeepLens()).simulate_psf_cube(spec, None, tmp_path, realization="adapter_proxy")
    metrics = result.metric_bundle.metrics
    manifest = (tmp_path / "run_manifest.json").read_text(encoding="utf-8")

    assert result.status == "succeeded"
    assert metrics["wavelength_aware_psf"] is True
    assert metrics["wavelength_count"] == len(spec.sweep_spec.wavelengths_nm)
    assert metrics["psf_band_axis"] == 1
    assert metrics["hsi_forward_compatible"] is True
    assert metrics["native_wavelength_physics"] is False
    assert "wavelength_aware_psf" in manifest

