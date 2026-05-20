from optiresearch.adapters.deeplens import DeepLensAdapter
from optiresearch.adapters.base import AdapterRunResult
from optiresearch.schemas.experiment import build_default_mock_edof_hsi_experiment

import numpy as np


def test_translate_experiment_spec_produces_candidate_config():
    experiment = build_default_mock_edof_hsi_experiment("translate deeplens")
    config = DeepLensAdapter().translate_experiment_spec(experiment)

    assert config["config_type"] == "DeepLensCandidateConfig"
    assert config["backend"] == "deeplens"
    assert config["schema_version"] == "0.1"
    assert config["wavelengths_nm"] == experiment.sweep_spec.wavelengths_nm
    assert config["depths_mm"] == experiment.sweep_spec.depths_mm
    assert config["psf_size"] == experiment.optical_spec.psf_size
    assert config["encoder_type"] == experiment.optical_spec.encoder_type
    assert config["sensor_type"] == "hsi"
    assert "unsupported_fields" in config
    assert "notes" in config
    assert any(item["field"] == "optical_spec.constraints" for item in config["unsupported_fields"])


class _FakeDeepLensPsf:
    def __init__(self, array):
        self._array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _FakeParaxialLens:
    def __init__(self, foclen, fnum, sensor_size=None, sensor_res=None, device="cpu"):
        self.foclen = foclen
        self.fnum = fnum
        self.focus = -1000.0

    def refocus(self, foc_dist):
        self.focus = foc_dist

    def psf(self, points, ks):
        depth = float(np.asarray(points)[0, 2])
        axis = np.linspace(-1.0, 1.0, ks)
        xx, yy = np.meshgrid(axis, axis)
        sigma = 0.15 + min(abs(depth - self.focus) / 2000.0, 0.2)
        psf = np.exp(-((xx**2 + yy**2) / (2.0 * sigma**2))).astype(np.float32)
        psf /= psf.sum()
        return _FakeDeepLensPsf(psf[None, :, :])


class _FakeDeepLensModule:
    __version__ = "fake-vcc"
    __file__ = "/fake/deeplens/__init__.py"
    ParaxialLens = _FakeParaxialLens


def test_vcc_deeplens_paraxiallens_smoke_with_fake_module(tmp_path):
    experiment = build_default_mock_edof_hsi_experiment("fake vcc deeplens smoke")
    adapter = DeepLensAdapter(deeplens_module=_FakeDeepLensModule())

    environment = adapter.validate_environment()
    result = adapter.simulate_psf_cube(experiment, None, tmp_path)

    assert environment["available"] is True
    assert "psf_smoke_available" in environment["capability_names"]
    assert isinstance(result, AdapterRunResult)
    assert result.status == "succeeded"
    assert result.metrics["depth_planes"] == 9
    assert result.metrics["wavelength_bands"] == 31
    assert (tmp_path / "psf_cube.npz").exists()
    assert (tmp_path / "optical_metrics.json").exists()
    assert (tmp_path / "raw_base_psf_cube.npz").exists()
    assert (tmp_path / "proxy_transform_manifest.json").exists()
    assert (tmp_path / "realization_manifest.json").exists()
    assert len(result.artifact_refs) == 7
