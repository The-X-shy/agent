"""Test trajectory evaluator cross-backend support."""

import pytest
from optiresearch.agents.trajectory_evaluator import evaluate_trajectory


class _FakeIter:
    def __init__(self, iteration_id, status="succeeded", metrics=None,
                 claim_dec=None, backend_id="phase_to_fft_proxy"):
        self.iteration_id = iteration_id
        self.execution_result = {
            "status": status,
            "result_payload": metrics or {},
            "backend_id": backend_id,
        }
        self.claim_gate_decision = claim_dec or {}
        self.metrics_snapshot = {}


class _FakeSpec:
    def __init__(self, max_iterations=3):
        self.max_iterations = max_iterations


def test_backend_history_tracked():
    iterations = [
        _FakeIter(1, "succeeded",
                  {"reconstruction_loss_after": 0.10},
                  {"max_allowed_claim": "native_full_reconstruction_proxy"},
                  "phase_to_fft_proxy"),
        _FakeIter(2, "succeeded",
                  {"reconstruction_loss_after": 0.05},
                  {"max_allowed_claim": "native_lens_simulation"},
                  "deeplens_geolens_geometric"),
    ]
    spec = _FakeSpec()
    result = evaluate_trajectory(iterations, spec)
    assert len(result.backend_history) == 2
    assert result.backend_switch_count == 1


def test_claim_ceiling_not_reached_with_different_backends():
    iterations = [
        _FakeIter(1, "succeeded",
                  {"reconstruction_loss_after": 0.10},
                  {"max_allowed_claim": "native_full_reconstruction_proxy"},
                  "phase_to_fft_proxy"),
        _FakeIter(2, "succeeded",
                  {"reconstruction_loss_after": 0.05},
                  {"max_allowed_claim": "native_lens_simulation"},
                  "deeplens_geolens_geometric"),
    ]
    spec = _FakeSpec()
    result = evaluate_trajectory(iterations, spec)
    assert not result.claim_ceiling_reached


def test_evidence_level_progression_detected():
    iterations = [
        _FakeIter(1, "succeeded",
                  {"reconstruction_loss_after": 0.10},
                  {"max_allowed_claim": "native_full_reconstruction_proxy"},
                  "phase_to_fft_proxy"),
        _FakeIter(2, "succeeded",
                  {"reconstruction_loss_after": 0.05},
                  {"max_allowed_claim": "native_lens_simulation"},
                  "deeplens_geolens_geometric"),
    ]
    spec = _FakeSpec()
    result = evaluate_trajectory(iterations, spec)
    assert result.evidence_level_progression is True


def test_same_backend_same_ceiling_still_ceiling_reached():
    iterations = [
        _FakeIter(1, "succeeded",
                  {"reconstruction_loss_after": 0.10},
                  {"max_allowed_claim": "native_full_reconstruction_proxy"},
                  "phase_to_fft_proxy"),
        _FakeIter(2, "succeeded",
                  {"reconstruction_loss_after": 0.05},
                  {"max_allowed_claim": "native_full_reconstruction_proxy"},
                  "phase_to_fft_proxy"),
    ]
    spec = _FakeSpec()
    result = evaluate_trajectory(iterations, spec)
    assert result.claim_ceiling_reached


def test_no_switch_with_single_backend():
    iterations = [
        _FakeIter(1, "succeeded",
                  {"reconstruction_loss_after": 0.10},
                  {"max_allowed_claim": "native_full_reconstruction_proxy"},
                  "phase_to_fft_proxy"),
    ]
    spec = _FakeSpec()
    result = evaluate_trajectory(iterations, spec)
    assert result.backend_switch_count == 0
    assert len(result.backend_history) == 1
