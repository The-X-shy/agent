"""Test codesign loop with different psf_source values."""
from optiresearch.schemas.optimization import build_default_optimization_spec
from optiresearch.runtime.codesign_loop import run_codesign_loop


def test_parameterized_mock_source_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    spec = build_default_optimization_spec(["PSNR"], backend="mock_deeplens", objective="test_mock")
    spec.max_iterations = 1
    spec.psf_source = "parameterized_mock"
    spec.llm_provider = "mock"

    result = run_codesign_loop(spec)
    assert result["psf_source"] == "parameterized_mock"
    assert result["total_iterations"] >= 1
    assert not result.get("fallback_used_any", True)


def test_deeplens_parameterized_source_falls_back_to_mock(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    spec = build_default_optimization_spec(["PSNR"], backend="deeplens", objective="test_dl_fallback")
    spec.max_iterations = 1
    spec.psf_source = "deeplens_parameterized"
    spec.fallback_policy = "fallback_to_mock"
    spec.llm_provider = "mock"

    result = run_codesign_loop(spec)
    assert result["psf_source"] == "deeplens_parameterized"
    # With fallback_to_mock, should still produce results
    assert result["total_iterations"] >= 1


def test_trajectory_includes_psf_source_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    spec = build_default_optimization_spec(["PSNR"], backend="mock_deeplens", objective="test_meta")
    spec.max_iterations = 2
    spec.psf_source = "parameterized_mock"
    spec.llm_provider = "mock"

    result = run_codesign_loop(spec)
    for t in result["trajectory"]:
        assert "psf_source" in t
        assert "backend" in t
        assert "fallback_used" in t
        assert "differentiable" in t
        assert "native_parameter_update" in t
        assert t["differentiable"] is False
        assert t["native_parameter_update"] is False


def test_caveats_mention_psf_source(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    spec = build_default_optimization_spec(["PSNR"], backend="mock_deeplens", objective="test_caveats")
    spec.max_iterations = 1
    spec.psf_source = "parameterized_mock"
    spec.llm_provider = "mock"

    result = run_codesign_loop(spec)
    assert isinstance(result["caveats"], list)
    assert any("parameterized" in c.lower() or "black-box" in c.lower() or "agent-driven" in c.lower() for c in result["caveats"])
