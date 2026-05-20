"""Test HSI baselines with optical-sensitive defaults."""

from optiresearch.runtime.hsi_baselines import run_hsi_encoder_baselines


def test_hsi_baselines_optical_sensitive(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_BASELINE_ROOT", str(tmp_path / "hsi_baselines"))

    report = run_hsi_encoder_baselines(
        backend="mock_deeplens",
        objective="HSI optical-sensitive baselines",
        forward_mode="depth_spectral_coded",
        reconstructor_type="optical_conditioned_linear",
        dataset_pattern="mixed_materials",
    )

    assert len(report["runs"]) == 5
    assert "best_reconstruction" in report
    assert report["forward_mode"] == "depth_spectral_coded"

    # At least 2 encoders should have different SAM or PSNR
    sam_values = {r["SAM"] for r in report["runs"] if r["SAM"] is not None}
    psnr_values = {r["PSNR"] for r in report["runs"] if r["PSNR"] is not None}
    assert len(sam_values) > 1 or len(psnr_values) > 1, (
        f"Expected encoder metrics to differ, got SAM={sam_values}, PSNR={psnr_values}"
    )

    # controlled_chromatic_edof should not be worse than conventional
    controlled = next(r for r in report["runs"] if r["encoder_type"] == "controlled_chromatic_edof")
    conventional = next(r for r in report["runs"] if r["encoder_type"] == "conventional")
    assert controlled["reconstruction_score"] >= conventional["reconstruction_score"], (
        f"controlled_chromatic_edof score {controlled['reconstruction_score']} < conventional {conventional['reconstruction_score']}"
    )

    # Check optical feature columns exist
    for run in report["runs"]:
        assert "coding_strength" in run
        assert "depth_stability_score" in run
        assert "spectral_separability_score" in run
        assert "ranking" in run

    assert (tmp_path / "hsi_baselines" / "mock_deeplens" / "hsi_baseline_comparison.json").exists()
    assert (tmp_path / "hsi_baselines" / "mock_deeplens" / "hsi_baseline_comparison.md").exists()


def test_hsi_baselines_backward_compat(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_BASELINE_ROOT", str(tmp_path / "hsi_baselines_compat"))

    report = run_hsi_encoder_baselines(
        backend="mock_deeplens",
        objective="HSI backward compat baselines",
        forward_mode="simple_sum",
        reconstructor_type="linear_baseline",
        dataset_pattern="smooth_low_rank",
    )

    assert len(report["runs"]) == 5
