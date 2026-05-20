"""Test autonomous loop schemas."""
from optiresearch.schemas.autonomous import (
    AutonomousLoopConfig,
    ResearchIterationPlan,
    ResearchIterationResult,
    AutonomousLoopSummary,
)


def test_autonomous_loop_config_defaults():
    config = AutonomousLoopConfig(objective="Test objective")
    assert config.objective == "Test objective"
    assert config.max_iterations == 3
    assert config.llm_provider == "mock"
    assert config.backend == "mock_deeplens"
    assert "controlled_chromatic_edof" in config.allowed_encoders
    assert "optical_conditioned_linear" in config.allowed_reconstructors


def test_research_iteration_plan_serializes():
    plan = ResearchIterationPlan(
        iteration_id=1,
        hypothesis="Test hypothesis",
        selected_encoder="controlled_chromatic_edof",
        selected_reconstructor="optical_conditioned_linear",
    )
    data = plan.model_dump()
    assert data["iteration_id"] == 1
    assert data["selected_encoder"] == "controlled_chromatic_edof"


def test_research_iteration_result_defaults():
    result = ResearchIterationResult(iteration_id=1)
    assert result.status == "failed"
    assert result.metrics == {}
    assert result.claims == []


def test_autonomous_loop_summary_structure():
    summary = AutonomousLoopSummary(
        objective="Test",
        loop_id="test_loop",
        total_iterations=2,
        stopped_reason="max_iterations",
        caveats=["mock only"],
    )
    data = summary.model_dump()
    assert data["loop_id"] == "test_loop"
    assert data["total_iterations"] == 2
    assert "mock only" in data["caveats"]
