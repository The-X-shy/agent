"""Test experiment_spec_patch validation and end-to-end integration."""

import pytest
from optiresearch.agents.strategy_to_spec import compile_experiment_spec
from optiresearch.agents.strategy_engine import StrategyRecommendation


def _make_rec(action, **kwargs):
    return StrategyRecommendation(
        recommended_action=action,
        rationale="test",
        **kwargs,
    )


def test_spec_patch_overrides_optical_lr():
    rec = _make_rec("enable_rollback")
    spec = compile_experiment_spec(
        rec, "phase_to_fft_proxy",
        spec_patch={"optical_lr": 1e-7, "max_steps": 8},
    )
    assert spec.spec_payload["optical_lr"] == 1e-7
    assert spec.spec_payload["max_steps"] == 8


def test_spec_patch_rejects_backend_id_override():
    rec = _make_rec("enable_rollback")
    spec = compile_experiment_spec(
        rec, "phase_to_fft_proxy",
        spec_patch={"backend_id": "deeplens_geolens_geometric"},
    )
    assert spec.backend_id == "phase_to_fft_proxy"


def test_spec_patch_rejects_execution_target_override():
    rec = _make_rec("enable_rollback")
    spec = compile_experiment_spec(
        rec, "phase_to_fft_proxy",
        spec_patch={"execution_target": "remote"},
    )
    assert spec.execution_target == "local"


def test_spec_includes_evidence_level():
    rec = _make_rec("enable_rollback")
    spec = compile_experiment_spec(rec, "phase_to_fft_proxy")
    assert spec.expected_evidence_level == "native_full_reconstruction_proxy"


def test_spec_includes_max_allowed_claim():
    rec = _make_rec("enable_rollback")
    spec = compile_experiment_spec(rec, "phase_to_fft_proxy")
    assert spec.max_allowed_claim == "native_full_reconstruction_proxy"


def test_spec_patch_rejects_claim_ceiling_override():
    rec = _make_rec("enable_rollback")
    spec = compile_experiment_spec(
        rec, "phase_to_fft_proxy",
        spec_patch={"claim_ceiling": "real_hsi_performance"},
    )
    assert "claim_ceiling" not in spec.spec_payload


def test_spec_patch_rejects_shell_command():
    rec = _make_rec("enable_rollback")
    spec = compile_experiment_spec(
        rec, "phase_to_fft_proxy",
        spec_patch={"shell_command": "rm -rf /"},
    )
    assert "shell_command" not in spec.spec_payload


def test_spec_patch_rejects_file_path():
    rec = _make_rec("enable_rollback")
    spec = compile_experiment_spec(
        rec, "phase_to_fft_proxy",
        spec_patch={"file_path": "/etc/passwd"},
    )
    assert "file_path" not in spec.spec_payload


def test_spec_patch_allows_safe_overrides():
    rec = _make_rec("enable_rollback")
    spec = compile_experiment_spec(
        rec, "phase_to_fft_proxy",
        spec_patch={
            "optical_lr": 1e-7,
            "recon_lr": 1e-4,
            "rollback_on_loss_increase": False,
            "lightweight_mode": True,
            "objective_profile": "fast_proxy",
        },
    )
    assert spec.spec_payload["optical_lr"] == 1e-7
    assert spec.spec_payload["recon_lr"] == 1e-4
    assert spec.spec_payload["rollback_on_loss_increase"] is False


def test_geolens_spec_has_correct_evidence_level():
    rec = _make_rec("enable_rollback")
    spec = compile_experiment_spec(rec, "deeplens_geolens_geometric")
    assert spec.expected_evidence_level == "native_lens_simulation"
    assert spec.max_allowed_claim == "native_lens_simulation"
