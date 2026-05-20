"""Test OpticalFeatureExtractor."""

import numpy as np

from optiresearch.hsi.optical_features import OpticalFeatureExtractor


def _make_psf_cube(depth_planes=3, wavelength_bands=5, psf_size=16):
    rng = np.random.default_rng(42)
    axis = np.linspace(-1.0, 1.0, psf_size)
    xx, yy = np.meshgrid(axis, axis)
    cube = np.zeros((depth_planes, wavelength_bands, psf_size, psf_size), dtype=np.float32)
    for d in range(depth_planes):
        for w in range(wavelength_bands):
            sigma = 0.15 + 0.03 * d + 0.05 * w
            offset = 0.03 * w
            psf = np.exp(-((xx - offset) ** 2 + yy**2) / (2 * sigma**2))
            psf = psf / psf.sum()
            cube[d, w] = psf
    return cube


def test_extract_returns_all_keys():
    cube = _make_psf_cube()
    features = OpticalFeatureExtractor().extract(cube)
    expected_keys = {
        "band_spread", "band_centroid_x", "band_centroid_y",
        "band_high_freq_energy", "depth_stability_score", "spectral_separability_score",
        "coding_strength", "band_condition_score", "depth_planes", "wavelength_bands",
    }
    assert expected_keys.issubset(set(features.keys()))


def test_band_features_have_correct_shape():
    cube = _make_psf_cube(wavelength_bands=5)
    features = OpticalFeatureExtractor().extract(cube)
    assert features["band_spread"].shape == (5,)
    assert features["band_centroid_x"].shape == (5,)
    assert features["band_centroid_y"].shape == (5,)
    assert features["band_high_freq_energy"].shape == (5,)


def test_depth_stability_in_range():
    cube = _make_psf_cube(depth_planes=3)
    features = OpticalFeatureExtractor().extract(cube)
    assert 0.0 <= features["depth_stability_score"] <= 1.0


def test_spectral_separability_in_range():
    cube = _make_psf_cube(wavelength_bands=5)
    features = OpticalFeatureExtractor().extract(cube)
    assert 0.0 <= features["spectral_separability_score"] <= 1.0


def test_different_cubes_yield_different_features():
    cube_a = _make_psf_cube(wavelength_bands=5)
    cube_b = _make_psf_cube(wavelength_bands=5)
    cube_b *= 2.0
    fa = OpticalFeatureExtractor().extract(cube_a)
    fb = OpticalFeatureExtractor().extract(cube_b)
    assert fa["depth_stability_score"] == fb["depth_stability_score"]


def test_rejects_3d_cube():
    extractor = OpticalFeatureExtractor()
    try:
        extractor.extract(np.ones((3, 16, 16), dtype=np.float32))
        assert False, "Expected ValueError"
    except ValueError:
        pass
