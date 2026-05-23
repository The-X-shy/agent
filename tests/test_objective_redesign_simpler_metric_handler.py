"""Test objective_redesign_simpler_metric_mse_only handler behavior."""

from optiresearch.runtime.lightweight_experiments import run_lightweight_mse_only_hsi


def test_mse_only_handler_uses_mse_objective():
    """The handler should use MSE-only loss — verified by mse_only_objective flag."""
    result = run_lightweight_mse_only_hsi(max_steps=5)
    payload = result.result_payload
    assert payload["mse_only_objective"] is True


def test_mse_only_handler_produces_metrics_consistent_with_loss():
    """MSE and reconstruction loss should be the same (MSE-only objective)."""
    result = run_lightweight_mse_only_hsi(max_steps=5)
    payload = result.result_payload
    assert payload["mse_before"] == payload["reconstruction_loss_before"]
    assert payload["mse_after"] == payload["reconstruction_loss_after"]


def test_mse_only_handler_does_not_use_deeplens():
    result = run_lightweight_mse_only_hsi(max_steps=3)
    payload = result.result_payload
    assert payload["deepens_used"] is False
    assert payload["physical_backend"] is False


def test_mse_only_handler_uses_synthetic_data():
    result = run_lightweight_mse_only_hsi(max_steps=3)
    payload = result.result_payload
    assert payload["synthetic_data"] is True


def test_mse_only_handler_best_loss_leq_final_loss():
    result = run_lightweight_mse_only_hsi(max_steps=10)
    payload = result.result_payload
    assert payload["best_reconstruction_loss"] <= payload["reconstruction_loss_after"] + 1e-9


def test_mse_only_handler_with_custom_steps():
    result = run_lightweight_mse_only_hsi(max_steps=2)
    assert result.status == "succeeded"
    payload = result.result_payload
    assert payload["reconstruction_loss_before"] is not None
    assert payload["reconstruction_loss_after"] is not None


def test_mse_only_handler_psnr_valid():
    result = run_lightweight_mse_only_hsi(max_steps=5)
    payload = result.result_payload
    psnr_before = payload["psnr_before"]
    psnr_after = payload["psnr_after"]
    if payload.get("improvement_detected"):
        assert psnr_after is not None
        assert psnr_before is not None
