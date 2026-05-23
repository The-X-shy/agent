"""Phase 31: TrajectoryEvaluator backend probe tests."""

from optiresearch.agents.trajectory_evaluator import evaluate_trajectory, TrajectoryEvaluation
from optiresearch.schemas.autonomous_loop import AutonomousLoopSpec, AutonomousLoopIteration


def _make_iter(iteration_id, backend_id, execution_result=None):
    return AutonomousLoopIteration(
        iteration_id=iteration_id,
        execution_result=execution_result or {},
    )


def _spec():
    return AutonomousLoopSpec(
        objective="test",
        max_iterations=3,
        min_iterations_before_stop=1,
    )


def test_switch_triggered_detected():
    iterations = [
        _make_iter(1, "phase_to_fft_proxy", {"backend_id": "phase_to_fft_proxy"}),
        _make_iter(2, "phase_to_fft_proxy", {
            "backend_id": "phase_to_fft_proxy",
            "switched_from_backend": "phase_to_fft_proxy",
            "switched_to_backend": "deeplens_geolens_geometric",
        }),
    ]
    result = evaluate_trajectory(iterations, _spec())
    assert result.backend_switch_triggered is True


def test_switch_validated_detected():
    iterations = [
        _make_iter(1, "phase_to_fft_proxy", {"backend_id": "phase_to_fft_proxy"}),
        _make_iter(2, "deeplens_geolens_geometric", {
            "backend_id": "deeplens_geolens_geometric",
            "backend_switch_validated": True,
            "result_payload": {"probe_status": "succeeded"},
        }),
    ]
    result = evaluate_trajectory(iterations, _spec())
    assert result.backend_switch_validated is True
    assert result.backend_probe_success is True


def test_no_switch_no_validation():
    iterations = [
        _make_iter(1, "phase_to_fft_proxy", {"backend_id": "phase_to_fft_proxy"}),
        _make_iter(2, "phase_to_fft_proxy", {"backend_id": "phase_to_fft_proxy"}),
    ]
    result = evaluate_trajectory(iterations, _spec())
    assert result.backend_switch_triggered is False
    assert result.backend_switch_validated is False


def test_probe_unavailable_detected():
    iterations = [
        _make_iter(1, "phase_to_fft_proxy", {"backend_id": "phase_to_fft_proxy"}),
        _make_iter(2, "deeplens_geolens_geometric", {
            "backend_id": "deeplens_geolens_geometric",
            "result_payload": {"probe_status": "unavailable"},
        }),
    ]
    result = evaluate_trajectory(iterations, _spec())
    assert result.backend_probe_unavailable is True


def test_evidence_gain_after_switch():
    iterations = [
        _make_iter(1, "phase_to_fft_proxy", {
            "backend_id": "phase_to_fft_proxy",
            "evidence_level": "native_full_reconstruction_proxy",
        }),
        _make_iter(2, "phase_to_fft_proxy", {
            "backend_id": "phase_to_fft_proxy",
            "evidence_level": "native_full_reconstruction_proxy",
            "switched_from_backend": "phase_to_fft_proxy",
            "switched_to_backend": "deeplens_geolens_geometric",
        }),
        _make_iter(3, "deeplens_geolens_geometric", {
            "backend_id": "deeplens_geolens_geometric",
            "evidence_level": "native_lens_simulation",
            "backend_switch_validated": True,
        }),
    ]
    result = evaluate_trajectory(iterations, _spec())
    assert result.backend_switch_validated is True
    assert result.evidence_gain_after_switch is True
