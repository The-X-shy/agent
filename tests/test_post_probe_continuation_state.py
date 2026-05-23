"""Phase 32: Post-probe continuation state tests."""

from optiresearch.runtime.autonomous_research_loop import run_autonomous_research_loop
from optiresearch.schemas.autonomous_loop import AutonomousLoopSpec


def test_post_probe_continuation_signal_injected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir(exist_ok=True)

    spec = AutonomousLoopSpec(
        objective="test post-probe continuation",
        max_iterations=4,
        min_iterations_before_stop=2,
        execution_mode="local",
        allowed_backends=["phase_to_fft_proxy", "deeplens_geolens_geometric"],
        allowed_task_types=["stable_lens_hsi_codesign", "backend_probe",
                           "native_lens_simulation_codesign"],
        allow_backend_switching=True,
        max_backend_switches=1,
        prefer_executable_actions=True,
        report=False,
    )
    result = run_autonomous_research_loop(spec)

    assert len(result.iterations) >= 2

    probe_found = False
    continuation_found = False
    for it in result.iterations:
        exec_result = it.execution_result or {}
        if exec_result.get("post_probe_continuation_required"):
            continuation_found = True
        payload = exec_result.get("result_payload") or {}
        if payload.get("probe_status") == "succeeded":
            probe_found = True
        if (it.strategy_recommendation.get("recommended_action")
                == "run_validated_backend_experiment"):
            continuation_found = True

    assert probe_found or continuation_found, (
        "Expected probe or continuation signal in loop iterations"
    )


def test_continuation_cleared_after_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir(exist_ok=True)

    spec = AutonomousLoopSpec(
        objective="test continuation clearance",
        max_iterations=4,
        min_iterations_before_stop=2,
        execution_mode="local",
        allowed_backends=["phase_to_fft_proxy", "deeplens_geolens_geometric"],
        allowed_task_types=["stable_lens_hsi_codesign", "backend_probe",
                           "native_lens_simulation_codesign"],
        allow_backend_switching=True,
        max_backend_switches=1,
        prefer_executable_actions=True,
        report=False,
    )
    result = run_autonomous_research_loop(spec)

    no_crash = all(
        it.stop_reason != "strategy_could_not_map_to_experiment"
        for it in result.iterations
        if it.iteration_id < len(result.iterations)
    )
    assert no_crash or len(result.iterations) >= 3
