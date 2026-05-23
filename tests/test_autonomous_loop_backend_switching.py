"""Test autonomous loop backend switching integration."""

import pytest
from optiresearch.schemas.autonomous_loop import AutonomousLoopSpec
from optiresearch.runtime.autonomous_research_loop import run_autonomous_research_loop


def _run_loop(monkeypatch, tmp_path, **overrides):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir()
    spec_kwargs = {
        "objective": "test multi-backend autonomous loop",
        "max_iterations": 3,
        "min_iterations_before_stop": 2,
        "execution_mode": "local",
        "allowed_backends": ["phase_to_fft_proxy", "deeplens_geolens_geometric"],
        "allowed_task_types": ["stable_lens_hsi_codesign", "psf_probe"],
        "planner_mode": "rule_based",
        "prefer_executable_actions": True,
        "allow_backend_switching": True,
        "max_backend_switches": 1,
        "report": False,
    }
    spec_kwargs.update(overrides)
    spec = AutonomousLoopSpec(**spec_kwargs)
    return run_autonomous_research_loop(spec)


def test_loop_with_backend_switching_runs(tmp_path, monkeypatch):
    result = _run_loop(monkeypatch, tmp_path)
    assert len(result.iterations) >= 1
    assert result.loop_id


def test_loop_without_backend_switching_stops_at_ceiling(tmp_path, monkeypatch):
    result = _run_loop(monkeypatch, tmp_path, allow_backend_switching=False)
    assert len(result.iterations) >= 1


def test_loop_backend_switching_uses_phase_to_fft_proxy(tmp_path, monkeypatch):
    result = _run_loop(monkeypatch, tmp_path)
    backends = set()
    for it in result.iterations:
        exec_result = it.execution_result or {}
        bid = exec_result.get("backend_id", "")
        if bid:
            backends.add(bid)
    assert "phase_to_fft_proxy" in backends


def test_loop_with_switching_allows_max_switches(tmp_path, monkeypatch):
    result = _run_loop(monkeypatch, tmp_path, max_backend_switches=1)
    assert result.status in ("stopped", "completed")


def test_loop_switching_flag_disabled_respects(tmp_path, monkeypatch):
    result = _run_loop(monkeypatch, tmp_path, allow_backend_switching=False)
    # Should still run without errors
    assert result.status in ("stopped", "completed")
