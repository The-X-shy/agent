"""Tests for StrategyEngine."""

from optiresearch.agents.strategy_engine import StrategyEngine, StrategyRecommendation


def test_large_gradient_reduces_lr():
    engine = StrategyEngine()
    rec = engine.recommend(
        {"optical_gradient_norm": 500, "max_steps": 10},
        "deeplens_geolens_geometric",
    )
    assert rec.recommended_action == "retry_with_smaller_lr"
    assert rec.risk_level == "low"
    assert len(rec.proposed_cli_commands) > 0


def test_high_rollback_ratio_freezes():
    engine = StrategyEngine()
    rec = engine.recommend(
        {"rollback_count": 8, "max_steps": 10},
        "deeplens_geolens_geometric",
    )
    assert rec.recommended_action == "switch_backend"


def test_zero_grad_with_optimizer_step_audits():
    engine = StrategyEngine()
    rec = engine.recommend(
        {"optical_gradient_norm": 0, "optimizer_step_executed": True},
        "deeplens_geolens_geometric",
    )
    assert rec.recommended_action == "run_ablation"


def test_loss_increase_no_rollback_enables_rollback():
    engine = StrategyEngine()
    rec = engine.recommend(
        {"loss_after": 2.0, "loss_before": 1.0, "rollback_protected": False, "rollback_count": 0},
        "deeplens_geolens_geometric",
    )
    assert rec.recommended_action == "enable_rollback"


def test_gradient_clip_required():
    engine = StrategyEngine()
    rec = engine.recommend(
        {"optical_gradient_norm_max": 50},
        "deeplens_geolens_geometric",
    )
    assert rec.recommended_action == "retry_with_smaller_lr"


def test_recon_loss_increase():
    engine = StrategyEngine()
    rec = engine.recommend(
        {"reconstruction_loss_before": 1.0, "reconstruction_loss_after": 1.5},
        "deeplens_geolens_geometric",
    )
    assert rec.recommended_action == "retry_with_smaller_lr"


def test_psf_energy_delta_large():
    engine = StrategyEngine()
    rec = engine.recommend(
        {"psf_energy_delta": 1.0},
        "deeplens_geolens_geometric",
    )
    assert rec.recommended_action == "run_ablation"


def test_default_recommendation_when_no_rules_match():
    engine = StrategyEngine()
    rec = engine.recommend(
        {"optical_gradient_norm": 1.0, "max_steps": 5},
        "deeplens_geolens_geometric",
    )
    assert rec is not None
    assert isinstance(rec, StrategyRecommendation)


def test_strategy_recommendation_has_all_fields():
    engine = StrategyEngine()
    rec = engine.recommend(
        {"optical_gradient_norm": 500, "max_steps": 10},
        "deeplens_geolens_geometric",
    )
    assert rec.recommended_action
    assert rec.rationale
    assert rec.risk_level in ("low", "medium", "high")
    assert isinstance(rec.required_evidence, list)
    assert isinstance(rec.proposed_cli_commands, list)


def test_stable_success_recommends_remote():
    engine = StrategyEngine()
    rec = engine.recommend(
        {"stable_training_succeeded": True},
        "deeplens_geolens_geometric",
    )
    assert "remote" in rec.recommended_action.lower()


def test_loss_decreased_recommends_remote():
    engine = StrategyEngine()
    rec = engine.recommend(
        {"reconstruction_loss_before": 2.0, "reconstruction_loss_after": 1.0},
        "deeplens_geolens_geometric",
    )
    assert "remote" in rec.recommended_action.lower()


def test_first_rule_fires_in_priority_order():
    """When multiple rules match, the first (highest priority) should fire."""
    engine = StrategyEngine()
    rec = engine.recommend(
        {
            "optical_gradient_norm": 500,  # Rule 1
            "optical_gradient_norm_max": 50,  # Rule 7
            "max_steps": 10,
        },
        "deeplens_geolens_geometric",
    )
    # Rule 1 (large gradient) should fire before Rule 7 (gradient clip required)
    assert rec.recommended_action == "retry_with_smaller_lr"
