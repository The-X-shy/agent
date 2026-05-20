from optiresearch.runtime.hsi_pipeline import run_hsi_reconstruction_flow


def test_hsi_reconstruction_flow_mock_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    result = run_hsi_reconstruction_flow(
        "Evaluate synthetic HSI reconstruction",
        backend="mock_deeplens",
        encoder_type="controlled_chromatic_edof",
        workspace_id="hsi_test",
    )

    assert result["run_id"]
    assert result["metrics"]["PSNR"] > 0.0
    assert "SAM" in result["metrics"]
    assert "reconstruction_metrics.json" in result["artifact_names"]
    assert result["claims"]
