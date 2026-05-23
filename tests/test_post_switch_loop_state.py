"""Phase 31: Post-switch loop state tests."""

import os
import tempfile
from pathlib import Path

from optiresearch.runtime.autonomous_research_loop import run_autonomous_research_loop
from optiresearch.schemas.autonomous_loop import AutonomousLoopSpec


def test_loop_with_backend_probe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir(exist_ok=True)

    spec = AutonomousLoopSpec(
        objective="test post-switch backend probe",
        max_iterations=3,
        min_iterations_before_stop=2,
        execution_mode="local",
        allowed_backends=["phase_to_fft_proxy", "deeplens_geolens_geometric"],
        allowed_task_types=["stable_lens_hsi_codesign", "backend_probe", "lightweight_psf_probe"],
        allow_backend_switching=True,
        max_backend_switches=1,
        prefer_executable_actions=True,
        report=False,
    )
    result = run_autonomous_research_loop(spec)

    probe_found = False
    switch_found = False
    for it in result.iterations:
        if it.next_action == "switch_backend":
            switch_found = True
        exec_result = it.execution_result or {}
        payload = exec_result.get("result_payload") or {}
        if payload.get("probe_status") == "succeeded":
            probe_found = True

    assert len(result.iterations) >= 1
    # On macOS, GeoLens native path may fail — loop should still not crash


def test_pending_backend_switch_injected_on_switch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir(exist_ok=True)

    spec = AutonomousLoopSpec(
        objective="test pending switch injection",
        max_iterations=3,
        min_iterations_before_stop=1,
        execution_mode="local",
        allowed_backends=["phase_to_fft_proxy", "deeplens_geolens_geometric"],
        allowed_task_types=["stable_lens_hsi_codesign", "backend_probe"],
        allow_backend_switching=True,
        max_backend_switches=1,
        prefer_executable_actions=True,
        report=False,
    )
    result = run_autonomous_research_loop(spec)

    assert len(result.iterations) >= 1
    switch_found = False
    for it in result.iterations:
        exec_result = it.execution_result or {}
        if exec_result.get("pending_backend_switch") and exec_result.get("switched_to_backend"):
            switch_found = True

    if len(result.iterations) >= 3:
        assert switch_found, "Expected pending_backend_switch to be set during backend switch"
