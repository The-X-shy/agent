"""Tests for ClaimGate with stabilized native GeoLens HSI evidence."""

from __future__ import annotations

import pytest

from optiresearch.memory.claim_gate_v2 import ClaimGateV2


def _make_result(**kwargs):
    defaults = {
        "evidence_level": "stable_native_lens_hsi_codesign",
        "execution_fidelity": "deeplens_native_geometric",
        "deeplens_native_psf_path": "geolens.psf_geometric",
        "full_wave_optics": False,
        "phase_to_fft_proxy_used": False,
        "mse_before": 0.5,
        "mse_after": 0.4,
        "sam_before": 1.0,
        "sam_after": 0.9,
        "claim_ceiling": "native_lens_simulation",
        "backend_id": "deeplens_geolens_geometric",
    }
    defaults.update(kwargs)
    return defaults


def test_native_geolens_geometric_training_claim_allowed():
    gate = ClaimGateV2()
    result = _make_result()
    decision = gate.check_claim(
        "native GeoLens geometric HSI co-design path is trainable via "
        "DeepLens native parameter API",
        backend_id="deeplens_geolens_geometric",
        experiment_result=result,
    )
    assert decision.decision in ("needs_followup", "qualified", "supported")


def test_claim_blocked_when_wave_optics_mentioned():
    gate = ClaimGateV2()
    result = _make_result()
    decision = gate.check_claim(
        "full wave-optics HSI co-design achieved with GeoLens",
        backend_id="deeplens_geolens_geometric",
        experiment_result=result,
    )
    assert decision.decision == "unsupported"


def test_sam_worsening_triggers_violation():
    gate = ClaimGateV2()
    result = _make_result(
        mse_before=0.5, mse_after=0.4,
        sam_before=1.0, sam_after=1.2,  # SAM worsened
    )
    decision = gate.check_claim(
        "stable multi-metric improvement achieved across all metrics",
        backend_id="deeplens_geolens_geometric",
        experiment_result=result,
    )
    assert decision.decision == "qualified"
    assert "sam" in str(decision.applicable_caveats).lower() or "SAM" in str(decision.applicable_caveats)


def test_sam_improvement_no_violation():
    gate = ClaimGateV2()
    result = _make_result(
        mse_before=0.5, mse_after=0.4,
        sam_before=1.0, sam_after=0.8,  # SAM improved
    )
    decision = gate.check_claim(
        "MSE and SAM both improved under native GeoLens simulation",
        backend_id="deeplens_geolens_geometric",
        experiment_result=result,
    )
    assert decision.decision != "unsupported"


def test_real_hsi_claim_blocked():
    gate = ClaimGateV2()
    result = _make_result()
    decision = gate.check_claim(
        "real HSI performance validated on GeoLens",
        backend_id="deeplens_geolens_geometric",
        experiment_result=result,
    )
    assert decision.decision == "unsupported"
