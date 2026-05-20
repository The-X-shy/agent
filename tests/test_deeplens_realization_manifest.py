from tests.test_deeplens_proxy_encoder_metrics import FakeDeepLens

from optiresearch.adapters.deeplens import DeepLensAdapter
from optiresearch.schemas.experiment import build_default_mock_edof_hsi_experiment


def test_deeplens_realization_manifest_is_written(tmp_path):
    adapter = DeepLensAdapter(deeplens_module=FakeDeepLens())
    spec = build_default_mock_edof_hsi_experiment("realization", encoder_type="conventional").model_copy(
        update={"backend": "deeplens"},
        deep=True,
    )

    result = adapter.simulate_psf_cube(spec, None, tmp_path, realization="auto")

    assert result.status == "succeeded"
    assert (tmp_path / "realization_manifest.json").exists()
    assert result.metrics["selected_realization_level"] == "semi_native"
    assert result.metrics["semi_native_attempted"] is True
    assert result.metrics["semi_native_succeeded"] is True
    assert result.metrics["proxy_fallback_used"] is False
