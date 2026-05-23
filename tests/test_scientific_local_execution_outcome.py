"""Test lightweight scientific execution outcome fields and evidence level."""

from optiresearch.runtime.agent_plan_execution_loop import (
    _execute_design,
    _lightweight_result_to_execution_result,
)
from optiresearch.agents.experiment_design_generator import ExperimentDesignCandidate
from optiresearch.runtime.lightweight_experiments import run_lightweight_mse_only_hsi


def _candidate(design_id: str, spec_payload: dict | None = None):
    return ExperimentDesignCandidate(
        design_id=design_id,
        objective=design_id,
        backend_id="deeplens_geolens_geometric",
        task_type="stable_lens_hsi_codesign",
        spec_payload=spec_payload or {},
    )


def test_lightweight_mse_only_hsi_returns_scientific_evidence_level():
    result = run_lightweight_mse_only_hsi(max_steps=5)
    assert result.status == "succeeded"
    assert result.evidence_level == "lightweight_scientific_execution"
    payload = result.result_payload
    assert payload is not None
    assert payload["synthetic_data"] is True
    assert payload["physical_backend"] is False
    assert payload["mse_only_objective"] is True
    assert payload["deepens_used"] is False
    assert payload["psf_generation_method"] == "fft_fraunhofer"
    assert payload["metrics_valid"] is True


def test_lightweight_mse_only_hsi_produces_all_required_metrics():
    result = run_lightweight_mse_only_hsi(max_steps=5)
    payload = result.result_payload
    for key in (
        "reconstruction_loss_before",
        "reconstruction_loss_after",
        "best_reconstruction_loss",
        "mse_before",
        "mse_after",
        "psnr_before",
        "psnr_after",
        "improvement_detected",
        "metrics_valid",
        "execution_time_sec",
        "synthetic_data",
        "physical_backend",
        "claim_ceiling",
    ):
        assert key in payload, f"Missing key: {key}"


def test_lightweight_mse_only_hsi_completes_quickly():
    import time
    start = time.perf_counter()
    result = run_lightweight_mse_only_hsi(max_steps=10)
    elapsed = time.perf_counter() - start
    assert elapsed < 60.0, f"Took {elapsed:.1f}s, expected <60s"
    assert result.status == "succeeded"


def test_objective_redesign_mse_only_design_returns_scientific_execution():
    result = _execute_design(
        _candidate(
            "objective_redesign_simpler_metric_mse_only",
            spec_payload={
                "loss_weights": {"mse": 1.0, "spectral_angle": 0.0, "measurement_consistency": 0.0},
                "max_steps": 5,
                "optical_lr": 1e-6,
            },
        )
    )
    assert result["status"] == "completed"
    assert result["evidence_level"] == "lightweight_scientific_execution"
    assert result["outcome"] == "lightweight_scientific_execution"
    metrics = result["metrics"]
    assert metrics["synthetic_data"] is True
    assert metrics["physical_backend"] is False
    assert metrics["mse_only_objective"] is True
    assert "reconstruction_loss_before" in metrics
    assert "reconstruction_loss_after" in metrics
    assert "mse_before" in metrics
    assert "mse_after" in metrics


def test_lightweight_result_converter_maps_all_fields():
    from optiresearch.runtime.experiment_controller_v2 import ControllerResult
    from optiresearch.memory.schemas import make_deterministic_id

    d = _candidate("test_design")
    result = ControllerResult(
        spec_id="test_spec",
        status="succeeded",
        backend_id="phase_to_fft_proxy",
        run_id=make_deterministic_id("test", "backend"),
        evidence_level="lightweight_scientific_execution",
        result_payload={
            "reconstruction_loss_before": 0.1,
            "reconstruction_loss_after": 0.05,
            "best_reconstruction_loss": 0.04,
            "mse_before": 0.1,
            "mse_after": 0.05,
            "psnr_before": 10.0,
            "psnr_after": 13.0,
            "improvement_detected": True,
            "metrics_valid": True,
            "accepted_update_count": 3,
            "execution_time_sec": 1.5,
            "synthetic_data": True,
            "physical_backend": False,
            "mse_only_objective": True,
            "backend_id": "phase_to_fft_proxy",
        },
    )
    converted = _lightweight_result_to_execution_result(d, result)
    assert converted["status"] == "completed"
    assert converted["evidence_level"] == "lightweight_scientific_execution"
    assert converted["metrics"]["improvement_detected"] is True
    assert converted["metrics"]["metrics_valid"] is True
    assert converted["caveats"]
    assert converted["metadata"]["synthetic_data"] is True


def test_lightweight_mse_only_hsi_improvement_detected_is_bool():
    result = run_lightweight_mse_only_hsi(max_steps=10)
    payload = result.result_payload
    assert isinstance(payload["improvement_detected"], bool)
    assert payload["reconstruction_loss_before"] is not None
    assert payload["reconstruction_loss_after"] is not None
