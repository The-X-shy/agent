"""Phase 31: StrategyEngine probe_new_backend tests."""

from optiresearch.agents.strategy_engine import StrategyEngine


def test_pending_backend_switch_triggers_probe():
    engine = StrategyEngine()
    result = {
        "pending_backend_switch": True,
        "switched_from_backend": "phase_to_fft_proxy",
        "switched_to_backend": "deeplens_geolens_geometric",
        "backend_switch_count": 1,
    }
    rec = engine.recommend(result, "deeplens_geolens_geometric")
    assert rec.recommended_action == "probe_new_backend"


def test_probe_takes_priority_over_claim_ceiling():
    engine = StrategyEngine()
    result = {
        "pending_backend_switch": True,
        "stop_reason": "claim_ceiling_reached",
        "switched_from_backend": "phase_to_fft_proxy",
        "switched_to_backend": "deeplens_geolens_geometric",
    }
    rec = engine.recommend(result, "deeplens_geolens_geometric")
    assert rec.recommended_action == "probe_new_backend"


def test_without_pending_switch_existing_rules_work():
    engine = StrategyEngine()
    result = {
        "stop_reason": "claim_ceiling_reached",
    }
    rec = engine.recommend(result, "phase_to_fft_proxy")
    assert rec.recommended_action == "switch_backend_after_claim_ceiling"


def test_pending_switch_default_fallback():
    engine = StrategyEngine()
    result = {
        "pending_backend_switch": True,
        "switched_from_backend": "mock_deeplens",
        "switched_to_backend": "phase_to_fft_proxy",
    }
    rec = engine.recommend(result, "phase_to_fft_proxy")
    assert rec.recommended_action == "probe_new_backend"
    assert "pending" in rec.rationale.lower() or "validating" in rec.rationale.lower()
