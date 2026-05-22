"""Phase 25 autonomous research loop dry run tests."""

import json
from pathlib import Path

from optiresearch.schemas.autonomous_loop import AutonomousLoopSpec
from optiresearch.runtime.autonomous_research_loop import run_autonomous_research_loop


def test_dry_run_returns_correct_structure():
    spec = AutonomousLoopSpec(
        objective="Test dry run",
        max_iterations=2,
        execution_mode="dry_run",
    )
    result = run_autonomous_research_loop(spec)
    assert result.status == "dry_run_only"
    assert len(result.iterations) == 2
    assert result.objective == "Test dry run"


def test_dry_run_iterations_have_strategy():
    spec = AutonomousLoopSpec(
        objective="Test dry run",
        max_iterations=1,
        execution_mode="dry_run",
    )
    result = run_autonomous_research_loop(spec)
    it = result.iterations[0]
    assert "recommended_action" in it.strategy_recommendation
    assert it.next_action in ("continue", "stop")
    assert it.stop_reason == "dry_run_no_execution"


def test_dry_run_respects_max_iterations():
    spec = AutonomousLoopSpec(
        objective="Test",
        max_iterations=3,
        execution_mode="dry_run",
    )
    result = run_autonomous_research_loop(spec)
    assert len(result.iterations) == 3


def test_dry_run_outputs_loop_spec_file():
    spec = AutonomousLoopSpec(
        objective="Test output files",
        max_iterations=1,
        execution_mode="dry_run",
    )
    result = run_autonomous_research_loop(spec)
    loop_dir = Path("workspace/autonomous_loops_v2") / result.loop_id
    assert (loop_dir / "loop_spec.json").exists()
    assert (loop_dir / "loop_result.json").exists()


def test_dry_run_iterations_have_experiment_spec():
    spec = AutonomousLoopSpec(
        objective="Test",
        max_iterations=1,
        execution_mode="dry_run",
    )
    result = run_autonomous_research_loop(spec)
    it = result.iterations[0]
    assert "spec_id" in it.experiment_spec or "task_type" in it.experiment_spec or it.experiment_spec == {}


def test_dry_run_loop_id_is_unique():
    spec1 = AutonomousLoopSpec(objective="Test A", max_iterations=1, execution_mode="dry_run")
    spec2 = AutonomousLoopSpec(objective="Test B", max_iterations=1, execution_mode="dry_run")
    r1 = run_autonomous_research_loop(spec1)
    r2 = run_autonomous_research_loop(spec2)
    assert r1.loop_id != r2.loop_id
