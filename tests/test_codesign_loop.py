"""Test co-design optimization loop."""
import json
from pathlib import Path
import numpy as np
from optiresearch.schemas.optimization import build_default_optimization_spec
from optiresearch.runtime.codesign_loop import run_codesign_loop


def test_codesign_loop_runs_with_mock_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    spec = build_default_optimization_spec(
        target_metrics=["PSNR"],
        backend="mock_deeplens",
        objective="Test co-design optimization",
    )
    spec.max_iterations = 2
    spec.llm_provider = "mock"

    result = run_codesign_loop(spec)

    assert result["loop_id"]
    assert result["total_iterations"] >= 1
    assert result["total_iterations"] <= 2
    assert isinstance(result["best_params"], dict)
    assert len(result["best_params"]) >= 1
    assert isinstance(result["trajectory"], list)
    assert len(result["trajectory"]) == result["total_iterations"]


def test_codesign_loop_creates_output_files(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    spec = build_default_optimization_spec(
        target_metrics=["PSNR"],
        backend="mock_deeplens",
        objective="Test output files",
    )
    spec.max_iterations = 1
    spec.llm_provider = "mock"

    result = run_codesign_loop(spec)
    output_dir = Path(result["output_dir"])

    assert output_dir.exists()
    assert (output_dir / "optimization_spec.json").exists()
    assert (output_dir / "codesign_loop_summary.json").exists()
    assert (output_dir / "codesign_loop_report.md").exists()
    # Check PSF files
    psf_files = list(output_dir.glob("iteration_*_psf_cube.npz"))
    assert len(psf_files) >= 1


def test_codesign_loop_psf_files_are_valid(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    spec = build_default_optimization_spec(
        target_metrics=["PSNR"],
        backend="mock_deeplens",
        objective="Test PSF files",
    )
    spec.max_iterations = 1
    spec.llm_provider = "mock"

    result = run_codesign_loop(spec)
    output_dir = Path(result["output_dir"])

    for psf_file in output_dir.glob("iteration_*_psf_cube.npz"):
        data = np.load(psf_file)
        assert "psf_cube" in data
        cube = data["psf_cube"]
        assert cube.ndim == 4
        assert np.all(np.isfinite(cube))


def test_codesign_loop_trajectory_has_expected_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    spec = build_default_optimization_spec(
        target_metrics=["PSNR"],
        backend="mock_deeplens",
        objective="Test trajectory",
    )
    spec.max_iterations = 2
    spec.llm_provider = "mock"

    result = run_codesign_loop(spec)

    for t in result["trajectory"]:
        for field in ["iteration", "optical_vars", "score", "loss"]:
            assert field in t, f"Missing trajectory field: {field}"


def test_codesign_loop_caveats_are_present(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    spec = build_default_optimization_spec(
        target_metrics=["PSNR"],
        backend="mock_deeplens",
        objective="Test caveats",
    )
    spec.max_iterations = 1
    spec.llm_provider = "mock"

    result = run_codesign_loop(spec)

    assert isinstance(result["caveats"], list)
    assert len(result["caveats"]) >= 1
    assert any("mock" in c.lower() or "synthetic" in c.lower() for c in result["caveats"])
