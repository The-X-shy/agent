"""Phase 32: Autonomous loop post-probe continuation integration tests."""

from optiresearch.runtime.autonomous_research_loop import run_autonomous_research_loop
from optiresearch.schemas.autonomous_loop import AutonomousLoopSpec


def test_full_probe_continuation_cycle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir(exist_ok=True)

    spec = AutonomousLoopSpec(
        objective="full probe-continuation cycle",
        max_iterations=5,
        min_iterations_before_stop=2,
        execution_mode="local",
        allowed_backends=["phase_to_fft_proxy", "deeplens_geolens_geometric"],
        allowed_task_types=[
            "stable_lens_hsi_codesign", "backend_probe",
            "native_lens_simulation_codesign", "lightweight_psf_probe",
        ],
        allow_backend_switching=True,
        max_backend_switches=1,
        prefer_executable_actions=True,
        report=False,
    )
    result = run_autonomous_research_loop(spec)

    assert result.status in ("stopped", "completed")
    assert len(result.iterations) >= 2

    actions = [
        it.strategy_recommendation.get("recommended_action", "")
        for it in result.iterations
    ]
    probe_seen = any(a == "probe_new_backend" for a in actions)
    continuation_seen = any(a == "run_validated_backend_experiment" for a in actions)

    assert probe_seen or continuation_seen, (
        f"Expected probe or continuation action in {actions}"
    )

    for it in result.iterations:
        if it.stop_reason == "strategy_could_not_map_to_experiment":
            assert it.iteration_id == len(result.iterations), (
                f"strategy_could_not_map_to_experiment at iter {it.iteration_id}"
                f" (last iter is {len(result.iterations)})"
            )
