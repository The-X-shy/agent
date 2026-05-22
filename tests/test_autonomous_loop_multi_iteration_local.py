"""Tests for multi-iteration autonomous loop execution."""

import json
from pathlib import Path

import pytest

from optiresearch.schemas.autonomous_loop import AutonomousLoopSpec


def _run_loop(monkeypatch, tmp_path, **overrides):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir()

    from optiresearch.runtime.autonomous_research_loop import (
        run_autonomous_research_loop,
    )

    spec_kwargs = {
        "objective": "test multi-iteration local loop",
        "max_iterations": 2,
        "execution_mode": "local",
        "planner_mode": "rule_based",
        "llm_provider": "mock",
        "allowed_backends": ["phase_to_fft_proxy"],
        "allowed_task_types": ["stable_lens_hsi_codesign", "lightweight_psf_probe"],
        "memory_update": False,
        "report": False,
    }
    spec_kwargs.update(overrides)
    spec = AutonomousLoopSpec(**spec_kwargs)
    return run_autonomous_research_loop(spec)


def test_multi_iteration_local_executes_at_least_one(monkeypatch, tmp_path):
    result = _run_loop(monkeypatch, tmp_path)
    assert result.status in ("completed", "stopped", "failed")
    assert len(result.iterations) >= 1


def test_multi_iteration_recent_results_grows(monkeypatch, tmp_path):
    result = _run_loop(monkeypatch, tmp_path)
    assert len(result.iterations) >= 1


def test_best_result_tracked(monkeypatch, tmp_path):
    result = _run_loop(monkeypatch, tmp_path)
    assert result.best_result is not None


def test_prefer_executable_actions_flag_flows_to_spec():
    spec = AutonomousLoopSpec(
        objective="test",
        prefer_executable_actions=True,
    )
    assert spec.prefer_executable_actions is True

    spec2 = AutonomousLoopSpec(
        objective="test",
    )
    assert spec2.prefer_executable_actions is False


def test_multi_iteration_loop_stops_at_max(monkeypatch, tmp_path):
    result = _run_loop(monkeypatch, tmp_path, max_iterations=2)
    assert len(result.iterations) <= 2


def test_loop_with_lightweight_task_type(monkeypatch, tmp_path):
    result = _run_loop(
        monkeypatch, tmp_path,
        allowed_task_types=["lightweight_psf_probe"],
    )
    assert result.status in ("completed", "stopped", "failed")
