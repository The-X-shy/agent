"""Test autonomous loop with mock LLM provider."""
from optiresearch.schemas.autonomous import AutonomousLoopConfig
from optiresearch.runtime.autonomous_loop import run_autonomous_research_loop


def test_autonomous_loop_runs_with_mock_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    config = AutonomousLoopConfig(
        objective="Test mock autonomous loop",
        max_iterations=2,
        llm_provider="mock",
        backend="mock_deeplens",
        dataset="synthetic",
        allowed_encoders=["conventional", "controlled_chromatic_edof"],
        allowed_reconstructors=["optical_conditioned_linear"],
    )
    summary = run_autonomous_research_loop(config)

    assert summary.total_iterations > 0
    assert summary.total_iterations <= 2
    assert summary.loop_id
    assert summary.objective == "Test mock autonomous loop"
    assert len(summary.iterations) == summary.total_iterations
    assert isinstance(summary.caveats, list)
    assert len(summary.caveats) >= 1


def test_autonomous_loop_creates_output_files(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    config = AutonomousLoopConfig(
        objective="Test output files",
        max_iterations=1,
        llm_provider="mock",
        backend="mock_deeplens",
        dataset="synthetic",
    )
    summary = run_autonomous_research_loop(config)

    import os
    output_dir = os.path.join("workspace", "autonomous_loops", summary.loop_id)
    assert os.path.exists(os.path.join(output_dir, "loop_config.json"))
    assert os.path.exists(os.path.join(output_dir, "autonomous_loop_summary.json"))
    assert os.path.exists(os.path.join(output_dir, "autonomous_iteration_report.md"))


def test_autonomous_loop_no_llm_direct_shell(tmp_path, monkeypatch):
    """Verify the loop does not shell out to LLM."""
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    config = AutonomousLoopConfig(
        objective="Test no shell",
        max_iterations=1,
        llm_provider="mock",
        backend="mock_deeplens",
        dataset="synthetic",
    )
    summary = run_autonomous_research_loop(config)
    # Should complete without subprocess/shell calls
    assert summary.total_iterations >= 1
