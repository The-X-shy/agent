"""Tests for executable strategy-to-spec compilation."""

import pytest

from optiresearch.agents.strategy_engine import StrategyRecommendation
from optiresearch.agents.strategy_to_spec import (
    MappingError,
    compile_experiment_spec,
    is_mapping_error,
)


def _make_rec(action: str, **kwargs) -> StrategyRecommendation:
    defaults = {
        "recommended_action": action,
        "rationale": "Test rationale.",
        "expected_claim_gain": "test_gain",
        "risk_level": "low",
        "required_evidence": [],
        "proposed_cli_commands": [],
        "proposed_experiment_spec": {},
    }
    defaults.update(kwargs)
    return StrategyRecommendation(**defaults)


def test_probe_waveoptics_maps_to_lightweight_psf_probe():
    rec = _make_rec("probe_waveoptics_path")
    spec = compile_experiment_spec(rec, "phase_to_fft_proxy")
    assert spec is not None
    assert spec.task_type == "lightweight_psf_probe"
    assert spec.spec_payload["max_steps"] == 3


def test_retry_with_prefer_executable_uses_small_max_steps():
    rec = _make_rec("retry_with_smaller_lr")
    spec = compile_experiment_spec(rec, "phase_to_fft_proxy", prefer_executable=True)
    assert spec is not None
    assert spec.spec_payload["max_steps"] <= 5


def test_retry_without_prefer_executable_uses_default_max_steps():
    rec = _make_rec("retry_with_smaller_lr")
    spec = compile_experiment_spec(rec, "phase_to_fft_proxy", prefer_executable=False)
    assert spec is not None
    assert spec.spec_payload["max_steps"] > 5


def test_experiment_spec_patch_overrides_defaults():
    rec = _make_rec("retry_with_smaller_lr")
    spec = compile_experiment_spec(
        rec, "phase_to_fft_proxy",
        prefer_executable=True,
        spec_patch={"optical_lr": 1e-7, "custom_field": "value"},
    )
    assert spec is not None
    assert spec.spec_payload["optical_lr"] == 1e-7
    assert spec.spec_payload["custom_field"] == "value"
    assert spec.spec_payload["max_steps"] <= 5  # base field preserved


def test_mapping_error_for_unmappable_action_with_prefer_executable():
    rec = _make_rec("unknown_action")
    result = compile_experiment_spec(
        rec, "phase_to_fft_proxy", prefer_executable=True
    )
    assert is_mapping_error(result)
    assert isinstance(result, MappingError)
    assert result.action == "unknown_action"


def test_stop_and_report_returns_none():
    rec = _make_rec("stop_and_report")
    result = compile_experiment_spec(rec, "phase_to_fft_proxy")
    assert result is None


def test_stop_and_report_returns_mapping_error_when_prefer_executable():
    rec = _make_rec("stop_and_report")
    result = compile_experiment_spec(
        rec, "phase_to_fft_proxy", prefer_executable=True
    )
    assert is_mapping_error(result)
    assert "stop_and_report" in result.reason


def test_enable_rollback_prefer_executable():
    rec = _make_rec("enable_rollback")
    spec = compile_experiment_spec(rec, "phase_to_fft_proxy", prefer_executable=True)
    assert spec is not None
    assert spec.spec_payload["max_steps"] <= 5
    assert spec.spec_payload["rollback_on_loss_increase"] is True
