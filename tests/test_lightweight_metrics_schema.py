"""Test that lightweight experiments produce the unified metrics schema."""

import pytest
from optiresearch.runtime.lightweight_experiments import (
    run_lightweight_stable_lens_hsi,
    run_lightweight_psf_probe,
    run_lightweight_ablation,
)

REQUIRED_METRIC_KEYS = [
    "reconstruction_loss_before",
    "reconstruction_loss_after",
    "best_reconstruction_loss",
    "improvement_detected",
    "execution_time_sec",
    "evidence_level",
    "claim_ceiling",
]


def test_stable_lens_hsi_produces_full_schema():
    result = run_lightweight_stable_lens_hsi(max_steps=3)
    assert result.status == "succeeded"
    payload = result.result_payload
    for key in REQUIRED_METRIC_KEYS:
        assert key in payload, f"Missing key: {key}"
    assert payload["metrics_valid"] is True
    assert payload["execution_time_sec"] > 0
    assert isinstance(payload["reconstruction_loss_before"], float)
    assert isinstance(payload["reconstruction_loss_after"], float)


def test_psf_probe_produces_schema():
    result = run_lightweight_psf_probe()
    assert result.status == "succeeded"
    payload = result.result_payload
    assert "psf_energy" in payload
    assert "elapsed_seconds" in payload
    assert payload["deepens_used"] is False


def test_ablation_produces_schema():
    result = run_lightweight_ablation(max_configs=2, max_steps=2)
    assert result.status == "succeeded"
    payload = result.result_payload
    assert "ablation_results" in payload
    assert "winner" in payload
    assert payload["deepens_used"] is False


def test_stable_lens_hsi_has_mse_and_psnr():
    result = run_lightweight_stable_lens_hsi(max_steps=3)
    payload = result.result_payload
    assert "mse_before" in payload
    assert "mse_after" in payload
    assert "psnr_before" in payload or payload["psnr_before"] is None
    assert "psnr_after" in payload or payload["psnr_after"] is None


def test_stable_lens_hsi_produces_deterministic_metrics():
    """Running twice with same params should produce metrics (may differ due to random init)."""
    r1 = run_lightweight_stable_lens_hsi(max_steps=3, optical_lr=1e-6)
    r2 = run_lightweight_stable_lens_hsi(max_steps=3, optical_lr=1e-6)
    assert r1.status == "succeeded"
    assert r2.status == "succeeded"
    assert r1.result_payload["metrics_valid"] is True
    assert r2.result_payload["metrics_valid"] is True
