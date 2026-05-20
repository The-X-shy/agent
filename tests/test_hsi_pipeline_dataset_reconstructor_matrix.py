from optiresearch.runtime.hsi_pipeline import run_hsi_reconstruction_flow


def test_hsi_pipeline_accepts_dataset_and_reconstructor_options(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))

    result = run_hsi_reconstruction_flow(
        "Evaluate Phase 11 matrix-ready synthetic HSI",
        backend="mock_deeplens",
        encoder_type="controlled_chromatic_edof",
        workspace_id="phase11_pipeline",
        dataset="synthetic",
        reconstructor="optical_conditioned_linear",
        forward_mode="depth_spectral_coded",
        use_optical_feature_maps=True,
    )

    assert result["dataset"]["dataset_family"] == "synthetic"
    assert result["reconstruction"]["network_type"] == "optical_conditioned_linear"
    assert result["metrics"]["PSNR"] > 0.0
    assert result["run_memory"]["current_status"]
    assert "dataset_manifest.json" in result["artifact_names"]

