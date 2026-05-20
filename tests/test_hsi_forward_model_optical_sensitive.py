"""Test optical-sensitive HSI forward model modes."""

import numpy as np

from optiresearch.hsi.forward_model import HSIForwardModel
from optiresearch.hsi.optical_features import OpticalFeatureExtractor
from optiresearch.schemas.hsi import build_default_hsi_forward_model_spec


def _make_psf_cube(depth_planes=3, wavelength_bands=4, psf_size=8):
    rng = np.random.default_rng(42)
    axis = np.linspace(-1.0, 1.0, psf_size)
    xx, yy = np.meshgrid(axis, axis)
    cube = np.zeros((depth_planes, wavelength_bands, psf_size, psf_size), dtype=np.float32)
    for d in range(depth_planes):
        for w in range(wavelength_bands):
            sigma = 0.15 + 0.03 * d + 0.05 * w
            psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            psf = psf / psf.sum()
            cube[d, w] = psf
    return cube


def _make_hsi(bands=4, h=8, w=8):
    rng = np.random.default_rng(7)
    return (rng.random((bands, h, w)) * 0.5).astype(np.float32)


def test_simple_sum_mode(tmp_path):
    psf = _make_psf_cube()
    hsi = _make_hsi()
    np.savez_compressed(tmp_path / "psf.npz", psf_cube=psf)
    spec = build_default_hsi_forward_model_spec(psf_cube_uri=str(tmp_path / "psf.npz"), depth_planes=3, wavelength_bands=4, forward_mode="simple_sum")
    model = HSIForwardModel(spec)
    loaded = model.load_psf_cube(str(tmp_path / "psf.npz"))
    meas = model.render_measurement(hsi, loaded, depth_index=1)
    assert meas.shape == (1, 8, 8)
    assert meas.min() >= 0.0


def test_psf_weighted_mode(tmp_path):
    psf = _make_psf_cube()
    hsi = _make_hsi()
    np.savez_compressed(tmp_path / "psf.npz", psf_cube=psf)
    spec = build_default_hsi_forward_model_spec(psf_cube_uri=str(tmp_path / "psf.npz"), depth_planes=3, wavelength_bands=4, forward_mode="psf_weighted")
    model = HSIForwardModel(spec)
    loaded = model.load_psf_cube(str(tmp_path / "psf.npz"))
    meas = model.render_measurement(hsi, loaded, depth_index=1)
    assert meas.shape == (1, 8, 8)


def test_coded_aperture_proxy_mode(tmp_path):
    psf = _make_psf_cube()
    hsi = _make_hsi()
    np.savez_compressed(tmp_path / "psf.npz", psf_cube=psf)
    feats = OpticalFeatureExtractor().extract(psf)
    spec = build_default_hsi_forward_model_spec(psf_cube_uri=str(tmp_path / "psf.npz"), depth_planes=3, wavelength_bands=4, forward_mode="coded_aperture_proxy")
    model = HSIForwardModel(spec)
    loaded = model.load_psf_cube(str(tmp_path / "psf.npz"))
    meas = model.render_measurement(hsi, loaded, depth_index=1, optical_features=feats)
    assert meas.shape == (1, 8, 8)


def test_depth_spectral_coded_mode(tmp_path):
    psf = _make_psf_cube()
    hsi = _make_hsi()
    np.savez_compressed(tmp_path / "psf.npz", psf_cube=psf)
    feats = OpticalFeatureExtractor().extract(psf)
    spec = build_default_hsi_forward_model_spec(psf_cube_uri=str(tmp_path / "psf.npz"), depth_planes=3, wavelength_bands=4, forward_mode="depth_spectral_coded")
    model = HSIForwardModel(spec)
    loaded = model.load_psf_cube(str(tmp_path / "psf.npz"))
    meas = model.render_measurement(hsi, loaded, depth_index=1, optical_features=feats)
    assert meas.shape == (1, 8, 8)


def test_coding_weights_saved(tmp_path):
    psf = _make_psf_cube()
    hsi = _make_hsi()
    np.savez_compressed(tmp_path / "psf.npz", psf_cube=psf)
    spec = build_default_hsi_forward_model_spec(psf_cube_uri=str(tmp_path / "psf.npz"), depth_planes=3, wavelength_bands=4, forward_mode="depth_spectral_coded")
    model = HSIForwardModel(spec)
    loaded = model.load_psf_cube(str(tmp_path / "psf.npz"))
    feats = OpticalFeatureExtractor().extract(loaded)
    meas, cw = model.render_measurement_with_coding_weights(hsi, loaded, 1, feats)
    assert meas.shape == (1, 8, 8)
    assert cw.shape == (4,)
    assert not np.allclose(cw[0], cw[-1])


def test_forward_model_artifacts(tmp_path):
    psf = _make_psf_cube()
    np.savez_compressed(tmp_path / "psf.npz", psf_cube=psf)
    spec = build_default_hsi_forward_model_spec(psf_cube_uri=str(tmp_path / "psf.npz"), depth_planes=3, wavelength_bands=4, forward_mode="depth_spectral_coded")
    model = HSIForwardModel(spec)
    feats = OpticalFeatureExtractor().extract(psf)
    paths = model.save_forward_artifacts(tmp_path, feats, np.ones(4, dtype=np.float32), {"mean": 0.5})
    assert any("optical_features.json" in str(p) for p in paths)
    assert any("forward_model_manifest.json" in str(p) for p in paths)
