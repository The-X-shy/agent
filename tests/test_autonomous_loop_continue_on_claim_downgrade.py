"""Test that autonomous loop continues after claim downgrade with valid metrics."""

import pytest
from optiresearch.schemas.autonomous_loop import AutonomousLoopSpec
from optiresearch.runtime.autonomous_research_loop import run_autonomous_research_loop


def _run_loop(monkeypatch, tmp_path, **overrides):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir()
    spec_kwargs = {
        "objective": "test multi-iteration local loop with claim downgrade",
        "max_iterations": 3,
        "min_iterations_before_stop": 2,
        "no_improvement_patience": 2,
        "execution_mode": "local",
        "allowed_backends": ["phase_to_fft_proxy"],
        "allowed_task_types": ["stable_lens_hsi_codesign"],
        "planner_mode": "rule_based",
        "prefer_executable_actions": True,
        "report": False,
    }
    spec_kwargs.update(overrides)
    spec = AutonomousLoopSpec(**spec_kwargs)
    return run_autonomous_research_loop(spec)


def test_loop_runs_at_least_min_iterations(tmp_path, monkeypatch):
    result = _run_loop(monkeypatch, tmp_path)
    assert len(result.iterations) >= 2, (
        f"Expected >= 2 iterations, got {len(result.iterations)}. "
        f"Stop reason: {result.trajectory_report_path}"
    )


def test_loop_iterations_have_execution_results(tmp_path, monkeypatch):
    result = _run_loop(monkeypatch, tmp_path)
    for it in result.iterations:
        assert it.execution_result, f"Iteration {it.iteration_id} has no execution result"
        status = it.execution_result.get("status", "")
        assert status in ("succeeded", "claim_downgraded", "skipped"), (
            f"Iteration {it.iteration_id} unexpected status: {status}"
        )


def test_loop_produces_metric_trajectory(tmp_path, monkeypatch):
    result = _run_loop(monkeypatch, tmp_path)
    metrics = []
    for it in result.iterations:
        payload = it.execution_result.get("result_payload") or it.metrics_snapshot or {}
        loss = payload.get("reconstruction_loss_after")
        if loss is not None:
            metrics.append(loss)
    assert len(metrics) >= 1, f"Expected at least 1 metric value, got {len(metrics)}"


def test_loop_with_max_iterations_stops_at_limit(tmp_path, monkeypatch):
    result = _run_loop(monkeypatch, tmp_path, max_iterations=2)
    assert len(result.iterations) <= 2


def test_loop_each_iteration_has_claim_decision(tmp_path, monkeypatch):
    result = _run_loop(monkeypatch, tmp_path)
    for it in result.iterations:
        assert it.claim_gate_decision is not None, (
            f"Iteration {it.iteration_id} missing claim gate decision"
        )
        assert "decision" in it.claim_gate_decision


def test_loop_result_has_metrics_snapshot(tmp_path, monkeypatch):
    result = _run_loop(monkeypatch, tmp_path)
    for it in result.iterations:
        snapshot = it.metrics_snapshot or {}
        # Each iteration should have at least the metrics_valid or claim_downgraded flag
        has_metrics = "metrics_valid" in snapshot or "claim_downgraded" in snapshot
        has_payload = bool(it.execution_result.get("result_payload"))
        assert has_metrics or has_payload, (
            f"Iteration {it.iteration_id} has no metrics in snapshot or payload"
        )
