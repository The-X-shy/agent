"""Phase 25 strategy-to-ExperimentSpec compiler tests."""

from optiresearch.agents.strategy_to_spec import compile_experiment_spec
from optiresearch.agents.strategy_engine import StrategyRecommendation


def test_retry_with_smaller_lr_maps_to_stable_lens():
    rec = StrategyRecommendation(
        recommended_action="retry_with_smaller_lr",
        rationale="Large optical gradients",
        risk_level="low",
    )
    spec = compile_experiment_spec(rec, "deeplens_geolens_geometric")
    assert spec is not None
    assert spec.task_type == "stable_lens_hsi_codesign"
    assert spec.backend_id == "deeplens_geolens_geometric"
    assert spec.spec_payload.get("rollback_on_loss_increase") is True
    assert spec.spec_payload.get("optical_lr") == 1e-6


def test_enable_rollback_maps_to_stable_lens():
    rec = StrategyRecommendation(
        recommended_action="enable_rollback",
        rationale="Loss increased without protection",
        risk_level="low",
    )
    spec = compile_experiment_spec(rec, "deeplens_geolens_geometric")
    assert spec is not None
    assert spec.task_type == "stable_lens_hsi_codesign"
    assert spec.spec_payload.get("rollback_on_loss_increase") is True


def test_switch_backend_maps_to_stable_lens():
    rec = StrategyRecommendation(
        recommended_action="switch_backend",
        rationale="Too many rollbacks",
        risk_level="medium",
    )
    spec = compile_experiment_spec(rec, "phase_to_fft_proxy")
    assert spec is not None
    assert spec.backend_id == "phase_to_fft_proxy"


def test_run_ablation_maps_to_stable_lens():
    rec = StrategyRecommendation(
        recommended_action="run_ablation",
        rationale="Need ablation study",
        risk_level="medium",
    )
    spec = compile_experiment_spec(rec, "deeplens_geolens_geometric")
    assert spec is not None
    assert spec.task_type == "stable_lens_hsi_codesign"


def test_stop_and_report_returns_none():
    rec = StrategyRecommendation(
        recommended_action="stop_and_report",
        rationale="No improvement path found",
        risk_level="medium",
    )
    spec = compile_experiment_spec(rec, "deeplens_geolens_geometric")
    assert spec is None


def test_downgrade_claim_returns_none():
    rec = StrategyRecommendation(
        recommended_action="downgrade_claim",
        rationale="Claim ceiling reached",
        risk_level="low",
    )
    spec = compile_experiment_spec(rec, "deeplens_geolens_geometric")
    assert spec is None


def test_spec_id_is_deterministic():
    rec = StrategyRecommendation(
        recommended_action="retry_with_smaller_lr",
        rationale="Test",
        risk_level="low",
    )
    s1 = compile_experiment_spec(rec, "deeplens_geolens_geometric")
    s2 = compile_experiment_spec(rec, "deeplens_geolens_geometric")
    assert s1 is not None and s2 is not None
    assert s1.spec_id == s2.spec_id


def test_run_remote_validation_maps_to_stable_lens():
    rec = StrategyRecommendation(
        recommended_action="run_remote_validation",
        rationale="Local success needs remote validation",
        risk_level="low",
    )
    spec = compile_experiment_spec(rec, "deeplens_geolens_geometric")
    assert spec is not None
    assert spec.task_type == "stable_lens_hsi_codesign"
    assert spec.spec_payload.get("rollback_on_loss_increase") is True
