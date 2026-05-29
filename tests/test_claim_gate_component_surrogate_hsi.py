"""ClaimGate tests for component surrogate HSI co-design claims."""

from optiresearch.memory.claim_gate_v2 import ClaimGateV2


def _result():
    return {
        "status": "completed",
        "evidence_level": "component_surrogate_hsi_codesign",
        "handler_id": "component_surrogate_hsi_codesign",
        "claim_ceiling": "component_surrogate_hsi_codesign",
        "synthetic_data": True,
        "physical_backend": False,
        "native_backend": False,
        "phase_to_fft_proxy_used": True,
        "full_wave_optics": False,
        "metrics": {
            "component_grad_norm_max": 0.1,
            "component_parameter_changed": True,
            "reconstruction_loss_before": 1.0,
            "reconstruction_loss_after": 0.9,
        },
    }


def test_component_surrogate_hsi_claim_is_supported():
    decision = ClaimGateV2().check_claim(
        "Component-level surrogate PSF can be optimized through synthetic HSI reconstruction loss",
        "component_surrogate_psf",
        experiment_result=_result(),
        handler_id="component_surrogate_hsi_codesign",
    )

    assert decision.decision == "supported"
    assert decision.final_claim_ceiling == "component_surrogate_hsi_codesign"


def test_component_surrogate_rejects_full_geolens_claim():
    decision = ClaimGateV2().check_claim(
        "Full GeoLens lens-level optimization succeeded",
        "component_surrogate_psf",
        experiment_result=_result(),
        handler_id="component_surrogate_hsi_codesign",
    )

    assert decision.decision == "unsupported"
    assert decision.violation_type == "component_surrogate_as_full_geolens"


def test_component_surrogate_rejects_real_hsi_performance_claim():
    decision = ClaimGateV2().check_claim(
        "Real HSI performance improved with physical camera validation",
        "component_surrogate_psf",
        experiment_result=_result(),
        handler_id="component_surrogate_hsi_codesign",
    )

    assert decision.decision == "unsupported"
    assert decision.violation_type in ("synthetic_metric_as_real_hsi", "handler_capability_exceeded")
