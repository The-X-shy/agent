import json

import numpy as np

from optiresearch.hsi.reconstruction import build_optical_feature_maps, run_reconstruction


def test_build_optical_feature_maps_expands_scalar_features():
    measurements = np.zeros((2, 1, 6, 5), dtype=np.float32)
    features = {
        "spectral_separability_score": 0.25,
        "depth_stability_score": 0.8,
        "coding_strength": 0.1,
        "band_condition_score": 0.4,
    }

    maps = build_optical_feature_maps(measurements, features)

    assert maps.shape == (2, 4, 6, 5)
    assert float(maps[0, 0, 0, 0]) == 0.25
    assert float(maps[1, 3, -1, -1]) == 0.4


def test_run_reconstruction_records_concat_scalar_maps_metadata(tmp_path):
    x = np.ones((2, 1, 8, 8), dtype=np.float32)
    y = np.ones((2, 3, 8, 8), dtype=np.float32) * 0.5
    features = {"spectral_separability_score": 0.25, "depth_stability_score": 0.8, "coding_strength": 0.1, "band_condition_score": 0.4}

    result = run_reconstruction(
        "optical_conditioned_linear",
        x,
        y,
        x[:1],
        y[:1],
        [0],
        tmp_path,
        optical_features=features,
        use_optical_feature_maps=True,
        optical_feature_injection="concat_scalar_maps",
    )
    manifest = json.loads((tmp_path / "reconstruction_manifest.json").read_text())

    assert result["status"] == "succeeded"
    assert manifest["optical_feature_injection"] == "concat_scalar_maps"
    assert manifest["input_channels"] == 5

