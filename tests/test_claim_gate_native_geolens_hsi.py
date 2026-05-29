"""Phase 33: ClaimGate native GeoLens HSI tests."""

from optiresearch.memory.claim_gate_v2 import ClaimGateV2


def test_geometric_as_coherent_still_works():
    gate = ClaimGateV2()
    d = gate.check_claim("Full wave-optics co-design", "deeplens_geolens_geometric")
    assert d.decision == "unsupported"
    assert d.violation_type == "geometric_as_coherent"
    assert len(d.safe_wording) > 0


def test_proxy_as_native_geolens_detected():
    gate = ClaimGateV2()
    d = gate.check_claim(
        "Native lens simulation via GeoLens geometric PSF",
        "deeplens_geolens_geometric",
        experiment_result={
            "execution_fidelity": "lightweight_proxy",
            "phase_to_fft_proxy_used": True,
        },
    )
    assert d.decision == "unsupported"
    assert d.violation_type == "proxy_as_native_geolens"


def test_native_geolens_no_violation():
    gate = ClaimGateV2()
    d = gate.check_claim(
        "Native lens simulation via GeoLens geometric PSF",
        "deeplens_geolens_geometric",
        experiment_result={
            "execution_fidelity": "deeplens_native_geometric",
            "phase_to_fft_proxy_used": False,
        },
    )
    assert d.decision == "supported"


def test_native_geolens_geometric_training_claim_allowed_with_connected_audit():
    gate = ClaimGateV2()
    d = gate.check_claim(
        "Full GeoLens geometric PSF native simulation parameters receive gradients from synthetic HSI loss",
        "deeplens_geolens_geometric",
        experiment_result={
            "evidence_level": "native_lens_simulation",
            "claim_ceiling": "native_lens_simulation",
            "execution_fidelity": "deeplens_native_geometric",
            "deeplens_native_psf_path": "geolens.psf_geometric",
            "phase_to_fft_proxy_used": False,
            "psf_requires_grad": True,
            "loss_requires_grad": True,
            "graph_connected": True,
            "trainable_param_count": 14,
            "params_with_grad": 14,
            "synthetic_data": True,
        },
        handler_id="deeplens_native_geolens_hsi_codesign",
    )

    assert d.decision == "supported"
    assert d.final_claim_ceiling == "native_lens_simulation"


def test_native_geolens_geometric_training_does_not_allow_real_hsi_claim():
    gate = ClaimGateV2()
    d = gate.check_claim(
        "Full GeoLens geometric PSF native simulation proves real HSI performance",
        "deeplens_geolens_geometric",
        experiment_result={
            "evidence_level": "native_lens_simulation",
            "claim_ceiling": "native_lens_simulation",
            "execution_fidelity": "deeplens_native_geometric",
            "phase_to_fft_proxy_used": False,
            "synthetic_data": True,
        },
        handler_id="deeplens_native_geolens_hsi_codesign",
    )

    assert d.decision == "unsupported"
    assert d.violation_type in ("synthetic_metric_as_real_hsi", "handler_capability_exceeded")


def test_proxy_backend_no_geolens_claim():
    gate = ClaimGateV2()
    d = gate.check_claim(
        "Differentiable HSI co-design experiment",
        "phase_to_fft_proxy",
    )
    assert d.decision == "supported"
