"""Test OpticalConditionedLinearReconstructor."""

import json

import numpy as np

from optiresearch.hsi.optical_features import OpticalFeatureExtractor
from optiresearch.hsi.reconstruction import (
    OpticalConditionedLinearReconstructor,
    run_reconstruction,
)


def _make_psf_cube(depth_planes=3, wavelength_bands=4, psf_size=8):
    axis = np.linspace(-1.0, 1.0, psf_size)
    xx, yy = np.meshgrid(axis, axis)
    cube = np.zeros((depth_planes, wavelength_bands, psf_size, psf_size), dtype=np.float32)
    for d in range(depth_planes):
        for w in range(wavelength_bands):
            sigma = 0.15 + 0.04 * w
            psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            cube[d, w] = psf / psf.sum()
    return cube


def test_optical_conditioned_linear_fit_and_predict():
    measurements = np.ones((4, 1, 8, 8), dtype=np.float32) * 0.5
    targets = np.stack([
        measurements[:, 0] * 0.2,
        measurements[:, 0] * 0.5,
        measurements[:, 0] * 0.8,
    ], axis=1)
    feats = OpticalFeatureExtractor().extract(_make_psf_cube(wavelength_bands=3))
    rec = OpticalConditionedLinearReconstructor(output_bands=3)
    rec.fit(measurements, targets, feats)
    pred = rec.predict(measurements[:1], feats)
    assert pred.shape == (1, 3, 8, 8)
    assert pred.min() >= 0.0


def test_optical_conditioned_linear_band_ordering():
    measurements = np.ones((4, 1, 8, 8), dtype=np.float32) * 0.3
    targets = np.stack([
        measurements[:, 0] * 0.1,
        measurements[:, 0] * 0.5,
        measurements[:, 0] * 0.9,
    ], axis=1)
    feats = OpticalFeatureExtractor().extract(_make_psf_cube(wavelength_bands=3))
    rec = OpticalConditionedLinearReconstructor(output_bands=3)
    rec.fit(measurements, targets, feats)
    pred = rec.predict(measurements[:1], feats)
    assert pred[0, 2].mean() > pred[0, 0].mean()


def test_run_reconstruction_optical_conditioned(tmp_path):
    measurements = np.ones((4, 1, 8, 8), dtype=np.float32) * 0.4
    targets = np.stack([measurements[:, 0] * 0.2, measurements[:, 0] * 0.6, measurements[:, 0] * 0.9], axis=1)
    feats = OpticalFeatureExtractor().extract(_make_psf_cube(wavelength_bands=3))
    result = run_reconstruction(
        "optical_conditioned_linear",
        measurements, targets,
        measurements[:2], targets[:2],
        np.array([0, 1]),
        tmp_path,
        feats,
    )
    assert "PSNR" in result["metrics"]
    assert result["metrics"]["network_type"] == "optical_conditioned_linear"
    assert (tmp_path / "reconstruction_metrics.json").exists()


def test_run_reconstruction_linear_baseline_still_works(tmp_path):
    measurements = np.ones((4, 1, 8, 8), dtype=np.float32) * 0.4
    targets = np.stack([measurements[:, 0] * 0.2, measurements[:, 0] * 0.6], axis=1)
    result = run_reconstruction(
        "linear_baseline",
        measurements, targets,
        measurements[:2], targets[:2],
        np.array([0, 1]),
        tmp_path,
    )
    assert "PSNR" in result["metrics"]
    assert result["metrics"]["network_type"] == "linear_baseline"
