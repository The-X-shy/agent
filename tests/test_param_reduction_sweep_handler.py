"""Test param reduction sweep handler produces valid metrics."""

from optiresearch.runtime.local_scientific_handlers import (
    run_param_reduction_sweep_lightweight,
)


def test_param_reduction_returns_succeeded():
    result = run_param_reduction_sweep_lightweight(max_steps=2)
    assert result.status == "succeeded"


def test_param_reduction_evidence_level():
    result = run_param_reduction_sweep_lightweight(max_steps=2)
    assert result.evidence_level == "lightweight_scientific_execution"


def test_param_reduction_produces_metrics():
    result = run_param_reduction_sweep_lightweight(max_steps=2)
    payload = result.result_payload
    for key in (
        "configs_tested",
        "best_k",
        "reconstruction_loss_before",
        "reconstruction_loss_after",
        "best_reconstruction_loss",
        "mse_before",
        "mse_after",
        "improvement_detected",
        "metrics_valid",
    ):
        assert key in payload, f"Missing key: {key}"


def test_param_reduction_configs_tested_is_3():
    result = run_param_reduction_sweep_lightweight(max_steps=2)
    payload = result.result_payload
    assert payload["configs_tested"] == 3  # k=1,2,3


def test_param_reduction_best_k_is_valid():
    result = run_param_reduction_sweep_lightweight(max_steps=2)
    payload = result.result_payload
    assert payload["best_k"] in (1, 2, 3)


def test_param_reduction_metadata():
    result = run_param_reduction_sweep_lightweight(max_steps=2)
    payload = result.result_payload
    assert payload["synthetic_data"] is True
    assert payload["physical_backend"] is False
    assert payload["native_backend"] is False
    assert payload["handler_id"] == "param_reduction_sweep"


def test_param_reduction_completes_quickly():
    import time
    start = time.perf_counter()
    result = run_param_reduction_sweep_lightweight(max_steps=2)
    elapsed = time.perf_counter() - start
    assert elapsed < 30.0, f"Took {elapsed:.1f}s, expected <30s"
    assert result.status == "succeeded"


def test_param_reduction_metrics_valid():
    result = run_param_reduction_sweep_lightweight(max_steps=3)
    payload = result.result_payload
    assert payload["metrics_valid"] is True
