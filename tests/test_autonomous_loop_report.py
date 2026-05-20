"""Test autonomous loop report generation."""
import json
from pathlib import Path
from optiresearch.schemas.autonomous import (
    AutonomousLoopConfig,
    AutonomousLoopSummary,
    ResearchIterationResult,
)
from optiresearch.runtime.autonomous_loop import run_autonomous_research_loop


def test_report_contains_required_sections(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    config = AutonomousLoopConfig(
        objective="Test report sections",
        max_iterations=1,
        llm_provider="mock",
        backend="mock_deeplens",
        dataset="synthetic",
        allowed_encoders=["controlled_chromatic_edof"],
        allowed_reconstructors=["optical_conditioned_linear"],
    )
    summary = run_autonomous_research_loop(config)

    import os
    report_path = Path("workspace/autonomous_loops") / summary.loop_id / "autonomous_iteration_report.md"
    assert report_path.exists()

    content = report_path.read_text(encoding="utf-8")
    for section in [
        "Autonomous Research Loop Report",
        "Objective",
        "Baseline Metrics",
        "Iteration Plans and Results",
        "Metric Trajectory",
        "Best Result",
        "Claims Supported",
        "Claims Rejected",
        "Evidence Caveats",
    ]:
        assert section in content, f"Missing section: {section}"


def test_report_mentions_limitations(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    config = AutonomousLoopConfig(
        objective="Test limitations in report",
        max_iterations=1,
        llm_provider="mock",
        backend="mock_deeplens",
        dataset="synthetic",
        allowed_encoders=["controlled_chromatic_edof"],
        allowed_reconstructors=["optical_conditioned_linear"],
    )
    summary = run_autonomous_research_loop(config)

    import os
    report_path = Path("workspace/autonomous_loops") / summary.loop_id / "autonomous_iteration_report.md"
    content = report_path.read_text(encoding="utf-8")

    assert "mock" in content.lower()
    assert "not real" in content.lower() or "native" in content.lower()
