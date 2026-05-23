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


def test_proxy_backend_no_geolens_claim():
    gate = ClaimGateV2()
    d = gate.check_claim(
        "Differentiable HSI co-design experiment",
        "phase_to_fft_proxy",
    )
    assert d.decision == "supported"
