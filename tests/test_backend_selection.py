from optiresearch.adapters.deeplens import DeepLensAdapter
from optiresearch.runtime.graph import run_mvp_flow


def test_run_mvp_backend_mock_deeplens_still_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "mock.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "mock_artifacts"))

    result = run_mvp_flow("Design a mock EDOF-HSI encoder", backend="mock_deeplens")

    assert result["experiment_spec"]["backend"] == "mock_deeplens"
    assert result["run_memory"]["current_status"] == "succeeded"
    assert result["artifact_ids"]
    assert any(claim["status"] in {"supported", "partially_supported"} for claim in result["claims"])


def test_run_mvp_backend_deeplens_unavailable_fails_safely(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "deeplens.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "deeplens_artifacts"))

    if DeepLensAdapter().validate_environment()["available"]:
        return

    result = run_mvp_flow("Design a minimal DeepLens PSF smoke run", backend="deeplens")

    assert result["experiment_spec"]["backend"] == "deeplens"
    assert result["run_memory"]["current_status"] == "failed"
    assert result["errors"]
    assert result["errors"][0]["code"] == "DEEPLENS_NOT_INSTALLED"
    assert not result["artifact_ids"]
    assert result["claims"]
    assert all(claim["status"] != "supported" for claim in result["claims"])
