"""Test autonomous loop trace and audit."""
from optiresearch.memory.meta_trace import MetaTraceWriter
from optiresearch.schemas.autonomous import AutonomousLoopConfig
from optiresearch.runtime.autonomous_loop import run_autonomous_research_loop


def test_traces_written_for_each_iteration(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    config = AutonomousLoopConfig(
        objective="Test trace audit",
        max_iterations=2,
        llm_provider="mock",
        backend="mock_deeplens",
        dataset="synthetic",
        allowed_encoders=["conventional", "controlled_chromatic_edof"],
        allowed_reconstructors=["optical_conditioned_linear"],
    )
    summary = run_autonomous_research_loop(config)

    traces = MetaTraceWriter().list_traces(run_id=summary.loop_id)
    assert len(traces) >= 1, "Expected at least one trace entry"

    phases = {t.phase for t in traces}
    assert "Explore" in phases or "Execute" in phases or "Review" in phases


def test_trace_metadata_includes_llm_info(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    config = AutonomousLoopConfig(
        objective="Test trace metadata",
        max_iterations=1,
        llm_provider="mock",
        backend="mock_deeplens",
        dataset="synthetic",
        allowed_encoders=["controlled_chromatic_edof"],
        allowed_reconstructors=["optical_conditioned_linear"],
    )
    summary = run_autonomous_research_loop(config)

    traces = MetaTraceWriter().list_traces(run_id=summary.loop_id)
    for t in traces:
        assert "llm_used" in t.metadata or "llm_provider" in t.metadata or t.phase in ("Execute",)


def test_trace_does_not_contain_shell_commands(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    config = AutonomousLoopConfig(
        objective="Test no shell in traces",
        max_iterations=1,
        llm_provider="mock",
        backend="mock_deeplens",
        dataset="synthetic",
    )
    summary = run_autonomous_research_loop(config)

    traces = MetaTraceWriter().list_traces(run_id=summary.loop_id)
    for t in traces:
        task_lower = t.task.lower()
        assert "shell" not in task_lower
        assert "subprocess" not in task_lower
        assert "bash" not in task_lower
