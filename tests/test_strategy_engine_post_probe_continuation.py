"""Phase 32: StrategyEngine post-probe continuation tests."""

from optiresearch.agents.strategy_engine import StrategyEngine


def test_post_probe_continuation_triggers_new_action():
    engine = StrategyEngine()
    result = {
        "post_probe_continuation_required": True,
        "validated_backend_id": "deeplens_geolens_geometric",
        "validated_backend_evidence_level": "native_lens_simulation",
    }
    rec = engine.recommend(result, "deeplens_geolens_geometric")
    assert rec.recommended_action == "run_validated_backend_experiment"


def test_continuation_has_rationale():
    engine = StrategyEngine()
    result = {
        "post_probe_continuation_required": True,
        "validated_backend_id": "deeplens_geolens_geometric",
    }
    rec = engine.recommend(result, "deeplens_geolens_geometric")
    assert "validated" in rec.rationale.lower() or "continuation" in rec.rationale.lower()


def test_continuation_without_post_probe_does_not_fire():
    engine = StrategyEngine()
    result = {
        "backend_switch_validated": True,
    }
    rec = engine.recommend(result, "deeplens_geolens_geometric")
    assert rec.recommended_action != "run_validated_backend_experiment"
