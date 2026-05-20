"""Test autonomous loop execution with mock provider."""
from optiresearch.schemas.autonomous import AutonomousLoopConfig
from optiresearch.runtime.autonomous_loop import run_autonomous_research_loop


def test_execution_produces_iteration_results(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    config = AutonomousLoopConfig(
        objective="Test execution produces results",
        max_iterations=2,
        llm_provider="mock",
        backend="mock_deeplens",
        dataset="synthetic",
        allowed_encoders=["conventional", "controlled_chromatic_edof"],
        allowed_reconstructors=["optical_conditioned_linear"],
    )
    summary = run_autonomous_research_loop(config)

    for it in summary.iterations:
        assert it.iteration_id >= 1
        assert it.status in ("succeeded", "failed", "validation_rejected")
        assert isinstance(it.metrics, dict)


def test_execution_includes_baseline_comparison(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    config = AutonomousLoopConfig(
        objective="Test baseline comparison",
        max_iterations=1,
        llm_provider="mock",
        backend="mock_deeplens",
        dataset="synthetic",
        allowed_encoders=["conventional", "controlled_chromatic_edof"],
        allowed_reconstructors=["optical_conditioned_linear"],
    )
    summary = run_autonomous_research_loop(config)

    assert isinstance(summary.baseline_metrics, dict)
    assert summary.best_iteration >= -1


def test_execution_stops_at_max_iterations(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    config = AutonomousLoopConfig(
        objective="Test max iterations",
        max_iterations=2,
        llm_provider="mock",
        backend="mock_deeplens",
        dataset="synthetic",
        allowed_encoders=["conventional", "controlled_chromatic_edof"],
        allowed_reconstructors=["optical_conditioned_linear"],
    )
    summary = run_autonomous_research_loop(config)
    assert summary.total_iterations <= config.max_iterations
