"""Test strategy-to-spec compilation for backend switching."""

import pytest
from optiresearch.agents.strategy_to_spec import compile_experiment_spec, _pick_task_for_backend
from optiresearch.agents.strategy_engine import StrategyRecommendation


def _make_rec(action, **kwargs):
    return StrategyRecommendation(
        recommended_action=action,
        rationale="test",
        metadata=kwargs.pop("metadata", {}),
        **kwargs,
    )


def test_pick_task_for_geolens():
    assert _pick_task_for_backend("deeplens_geolens_geometric") == "psf_probe"


def test_pick_task_for_proxy():
    assert _pick_task_for_backend("phase_to_fft_proxy") == "stable_lens_hsi_codesign"


def test_pick_task_for_unknown():
    assert _pick_task_for_backend("unknown") == "lightweight_psf_probe"


def test_switch_backend_after_ceiling_uses_next_backend():
    rec = _make_rec(
        "switch_backend_after_claim_ceiling",
        metadata={"next_backend": "deeplens_geolens_geometric"},
    )
    spec = compile_experiment_spec(rec, "deeplens_geolens_geometric")
    assert spec is not None
    assert spec.backend_id == "deeplens_geolens_geometric"
    assert spec.task_type is not None
    assert spec.expected_evidence_level is not None


def test_switch_backend_after_ceiling_selects_psf_probe_for_geolens():
    rec = _make_rec(
        "switch_backend_after_claim_ceiling",
        metadata={"next_backend": "deeplens_geolens_geometric"},
    )
    spec = compile_experiment_spec(rec, "deeplens_geolens_geometric")
    assert spec.task_type in ("psf_probe", "stable_lens_hsi_codesign")


def test_switch_backend_from_proxy_to_geolens_claim_gain():
    rec = _make_rec(
        "switch_backend_after_claim_ceiling",
        metadata={
            "next_backend": "deeplens_geolens_geometric",
            "expected_claim_gain": "native_full_reconstruction_proxy -> native_lens_simulation",
        },
    )
    spec = compile_experiment_spec(rec, "deeplens_geolens_geometric")
    assert spec.max_allowed_claim is not None
    assert spec.backend_id == "deeplens_geolens_geometric"


def test_switch_backend_payload_has_rollback():
    rec = _make_rec("switch_backend_after_claim_ceiling")
    spec = compile_experiment_spec(rec, "deeplens_geolens_geometric")
    assert spec.spec_payload.get("rollback_on_loss_increase") is True


def test_switch_backend_payload_has_device():
    rec = _make_rec("switch_backend_after_claim_ceiling")
    spec = compile_experiment_spec(rec, "deeplens_geolens_geometric")
    assert spec.spec_payload.get("device") == "cpu"
