"""Test HSI pipeline with optical-sensitive defaults."""

from optiresearch.runtime.hsi_pipeline import run_hsi_reconstruction_flow


def test_hsi_pipeline_optical_sensitive_mock(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    result = run_hsi_reconstruction_flow(
        "Evaluate optical-sensitive synthetic HSI reconstruction",
        backend="mock_deeplens",
        encoder_type="controlled_chromatic_edof",
        workspace_id="hsi_opt_test",
        forward_mode="depth_spectral_coded",
        reconstructor_type="optical_conditioned_linear",
        dataset_pattern="mixed_materials",
    )

    assert result["run_id"]
    assert result["metrics"]["PSNR"] > 0.0
    assert "SAM" in result["metrics"]
    assert "reconstruction_metrics.json" in result["artifact_names"]
    assert "optical_features.json" in result["artifact_names"]
    assert "forward_model_manifest.json" in result["artifact_names"]
    assert result["optical_features"]["depth_stability_score"] is not None
    assert result["claims"]


def test_hsi_pipeline_backward_compat_simple_sum(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    result = run_hsi_reconstruction_flow(
        "Evaluate backward compat HSI",
        backend="mock_deeplens",
        encoder_type="conventional",
        workspace_id="hsi_compat_test",
        forward_mode="simple_sum",
        reconstructor_type="linear_baseline",
        dataset_pattern="smooth_low_rank",
    )

    assert result["run_id"]
    assert result["metrics"]["PSNR"] > 0.0
    assert "reconstruction_metrics.json" in result["artifact_names"]
