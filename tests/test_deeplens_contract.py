from pathlib import Path

from optiresearch.adapters.base import AdapterRunResult
from optiresearch.adapters.deeplens import DeepLensAdapter
from optiresearch.adapters.mock_deeplens import MockDeepLensAdapter
from optiresearch.schemas.experiment import build_default_mock_edof_hsi_experiment


def test_mock_adapter_returns_shared_result_contract(tmp_path):
    experiment = build_default_mock_edof_hsi_experiment("contract mock")
    result = MockDeepLensAdapter(seed=42).simulate_psf_cube(experiment, None, tmp_path)

    assert isinstance(result, AdapterRunResult)
    assert result.status == "succeeded"
    assert result.metric_bundle.metrics["psf_depth_similarity"] >= 0.0
    assert result.metrics == result.metric_bundle.metrics
    assert result["metrics"] == result.metrics
    assert len(result.artifacts) >= 4
    assert all(artifact.path for artifact in result.artifact_refs)


def test_deeplens_missing_backend_returns_structured_error(tmp_path):
    experiment = build_default_mock_edof_hsi_experiment("real deeplens contract")
    adapter = DeepLensAdapter()

    environment = adapter.validate_environment()
    assert "ok" in environment
    if not environment["ok"]:
        assert environment["backend"] == "deeplens"
        assert environment["error"]["code"] == "DEEPLENS_NOT_INSTALLED"

    result = adapter.simulate_psf_cube(experiment, None, tmp_path)
    assert isinstance(result, AdapterRunResult)
    assert result.status in {"failed", "succeeded"}
    assert set(result.model_dump()).issuperset({"status", "artifacts", "metric_bundle", "logs", "errors"})
    if result.status == "failed":
        assert result.errors
        assert result.errors[0]["code"] == "DEEPLENS_NOT_INSTALLED"


def test_deeplens_and_mock_result_fields_are_compatible(tmp_path):
    experiment = build_default_mock_edof_hsi_experiment("adapter compatibility")
    mock_result = MockDeepLensAdapter(seed=42).simulate_psf_cube(experiment, None, tmp_path / "mock")
    deeplens_result = DeepLensAdapter().simulate_psf_cube(experiment, None, tmp_path / "real")

    assert set(mock_result.model_dump()) == set(deeplens_result.model_dump())
    assert isinstance(Path(mock_result.artifacts[0]), Path)
