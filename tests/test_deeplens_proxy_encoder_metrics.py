from __future__ import annotations

import json

import numpy as np

from optiresearch.adapters.deeplens import DeepLensAdapter
from optiresearch.schemas.experiment import build_default_mock_edof_hsi_experiment


class FakeParaxialLens:
    def __init__(self, *args, **kwargs):
        pass

    def refocus(self, depth):
        self.depth = depth

    def psf(self, points, ks):
        depth = float(np.asarray(points)[0, 2])
        axis = np.linspace(-1.0, 1.0, ks)
        xx, yy = np.meshgrid(axis, axis)
        sigma = 0.2 + 0.00003 * abs(depth + 1000.0)
        psf = np.exp(-((xx**2 + yy**2) / (2 * sigma**2)))
        return psf.astype(np.float32)


class FakeDeepLens:
    __version__ = "fake"
    ParaxialLens = FakeParaxialLens


def test_deeplens_proxy_transform_produces_encoder_specific_metrics(tmp_path):
    adapter = DeepLensAdapter(deeplens_module=FakeDeepLens())
    edof = build_default_mock_edof_hsi_experiment("proxy metrics", encoder_type="edof").model_copy(
        update={"backend": "deeplens"},
        deep=True,
    )
    chromatic = build_default_mock_edof_hsi_experiment("proxy metrics", encoder_type="chromatic_coded").model_copy(
        update={"backend": "deeplens"},
        deep=True,
    )

    edof_result = adapter.simulate_psf_cube(edof, None, tmp_path / "edof")
    chromatic_result = adapter.simulate_psf_cube(chromatic, None, tmp_path / "chromatic")

    assert edof_result.status == "succeeded"
    assert chromatic_result.status == "succeeded"
    assert edof_result.metrics["encoder_behavior_realized"] is True
    assert edof_result.metrics["encoder_behavior_realization_level"] == "adapter_proxy"
    assert edof_result.metrics["physical_validation_level"] == "deeplens_base_psf_plus_adapter_proxy"
    assert edof_result.metrics["psf_depth_similarity"] > chromatic_result.metrics["psf_depth_similarity"]
    assert chromatic_result.metrics["spectral_separability"] > edof_result.metrics["spectral_separability"]
    assert (tmp_path / "edof" / "raw_base_psf_cube.npz").exists()
    assert (tmp_path / "edof" / "proxy_transform_manifest.json").exists()

    manifest = json.loads((tmp_path / "edof" / "proxy_transform_manifest.json").read_text(encoding="utf-8"))
    assert manifest["realization_level"] == "adapter_proxy"
    assert manifest["proxy_transform_applied"] is True
