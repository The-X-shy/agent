"""Test strategy engine backend switching recommendations."""

import pytest
from optiresearch.agents.strategy_engine import StrategyEngine


def test_claim_ceiling_reached_recommends_backend_switch():
    engine = StrategyEngine()
    result = {
        "status": "succeeded",
        "stop_reason": "claim_ceiling_reached",
        "backend_id": "phase_to_fft_proxy",
    }
    rec = engine.recommend(result, "phase_to_fft_proxy")
    assert rec.recommended_action in (
        "switch_backend_after_claim_ceiling",
        "switch_backend",
    )


def test_claim_ceiling_not_reached_no_switch():
    engine = StrategyEngine()
    result = {
        "status": "succeeded",
        "reconstruction_loss_after": 0.03,
        "stop_reason": "",
        "backend_id": "phase_to_fft_proxy",
    }
    rec = engine.recommend(result, "phase_to_fft_proxy")
    assert rec.recommended_action != "switch_backend_after_claim_ceiling"


def test_geometric_ceiling_with_probe():
    engine = StrategyEngine()
    result = {
        "status": "succeeded",
        "stop_reason": "claim_ceiling_reached",
        "backend_id": "deeplens_geolens_geometric",
    }
    rec = engine.recommend(result, "deeplens_geolens_geometric")
    assert rec.recommended_action in (
        "switch_backend_after_claim_ceiling",
        "probe_waveoptics_path",
        "stop_and_report",
    )


def test_no_stop_reason_does_default():
    engine = StrategyEngine()
    result = {
        "status": "succeeded",
        "stable_training_succeeded": True,
        "loss_after": 0.02,
        "loss_before": 0.05,
    }
    rec = engine.recommend(result, "phase_to_fft_proxy")
    assert rec.recommended_action in (
        "run_remote_validation",
        "stop_and_report",
        "retry_with_smaller_lr",
        "enable_rollback",
    )
