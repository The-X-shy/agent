from optiresearch.runtime.hsi_baselines import run_hsi_encoder_baselines


def test_hsi_encoder_baselines_mock(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_BASELINE_ROOT", str(tmp_path / "hsi_baselines"))

    report = run_hsi_encoder_baselines(backend="mock_deeplens", objective="HSI baselines")

    assert len(report["runs"]) == 5
    assert "best_reconstruction" in report
    assert (tmp_path / "hsi_baselines" / "mock_deeplens" / "hsi_baseline_comparison.json").exists()
    assert (tmp_path / "hsi_baselines" / "mock_deeplens" / "hsi_baseline_comparison.md").exists()
