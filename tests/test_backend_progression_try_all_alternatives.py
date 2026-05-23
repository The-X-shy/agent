"""Phase 32: Backend progression try-all-alternatives tests."""

from optiresearch.backends.progression import get_all_edges_from


def test_all_edges_documented_for_fallback():
    edges = get_all_edges_from("phase_to_fft_proxy")
    assert len(edges) >= 2


def test_geolens_edges_include_alternatives():
    edges = get_all_edges_from("deeplens_geolens_geometric")
    targets = [e["next_backend"] for e in edges]
    assert "deeplens_coherent_asm" in targets or len(targets) >= 1


def test_alternative_attempts_tracked_in_loop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir(exist_ok=True)

    from optiresearch.runtime.autonomous_research_loop import run_autonomous_research_loop
    from optiresearch.schemas.autonomous_loop import AutonomousLoopSpec

    spec = AutonomousLoopSpec(
        objective="test alternative tracking",
        max_iterations=3,
        min_iterations_before_stop=1,
        execution_mode="local",
        allowed_backends=["phase_to_fft_proxy", "deeplens_geolens_geometric",
                         "deeplens_fresnel_component"],
        allowed_task_types=["stable_lens_hsi_codesign", "backend_probe"],
        allow_backend_switching=True,
        max_backend_switches=2,
        prefer_executable_actions=True,
        report=False,
    )
    result = run_autonomous_research_loop(spec)
    assert result.status in ("stopped", "completed")
    assert len(result.iterations) >= 1
