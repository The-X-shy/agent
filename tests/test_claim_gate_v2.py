"""Tests for ClaimGateV2."""

from optiresearch.memory.claim_gate_v2 import ClaimGateV2


def test_proxy_as_waveoptics_detected():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Phase-to-FFT proxy supports wave-optics co-design",
        "phase_to_fft_proxy",
    )
    assert decision.decision == "unsupported"
    assert decision.violation_type == "proxy_as_waveoptics"


def test_geometric_as_coherent_detected():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "GeoLens geometric PSF supports coherent wave-optics",
        "deeplens_geolens_geometric",
    )
    assert decision.decision == "unsupported"
    assert decision.violation_type == "geometric_as_coherent"


def test_geometric_wave_optics_detected():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Full DeepLens wave-optics native HSI co-design is supported",
        "deeplens_geolens_geometric",
    )
    assert decision.decision == "unsupported"
    assert decision.violation_type == "geometric_as_coherent"


def test_synthetic_as_real_detected():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Synthetic HSI achieves real-world performance",
        "local_synthetic_hsi",
    )
    assert decision.decision == "unsupported"
    assert decision.violation_type == "synthetic_as_real"


def test_differentiable_as_improves_detected():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Differentiable optimization improves HSI reconstruction",
        "deeplens_geolens_geometric",
        experiment_result={
            "reconstruction_loss_before": 1.0,
            "reconstruction_loss_after": 1.5,
        },
    )
    assert decision.decision == "qualified"
    assert decision.violation_type == "differentiable_as_improves"


def test_rollback_as_improvement_detected():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Rollback improves optical performance",
        "deeplens_geolens_geometric",
        experiment_result={"accepted_update_count": 0},
    )
    assert decision.decision == "qualified"
    assert decision.violation_type == "rollback_protection_as_improvement"


def test_black_box_as_native_detected():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Black-box PSF supports native gradient optimization",
        "deeplens_blackbox_source_psf",
    )
    assert decision.decision == "unsupported"
    assert decision.violation_type == "black_box_as_native"


def test_clean_claim_passes():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Native lens simulation HSI co-design is supported",
        "deeplens_geolens_geometric",
    )
    assert decision.decision == "supported"
    assert decision.violation_type is None


def test_safe_wording_generated():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Full DeepLens wave-optics native HSI co-design is supported",
        "deeplens_geolens_geometric",
    )
    assert "geometric" in decision.safe_wording.lower()
    assert decision.safe_wording != "Full DeepLens wave-optics native HSI co-design is supported"


def test_max_allowed_claim_from_backend():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Some claim",
        "deeplens_geolens_geometric",
    )
    assert decision.max_allowed_claim == "native_lens_simulation"


def test_caveats_provided_for_violations():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Synthetic data shows real HSI performance",
        "local_synthetic_hsi",
    )
    assert len(decision.applicable_caveats) > 0


def test_unsupported_path_as_supported():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Coherent ASM HSI co-design is supported",
        "deeplens_coherent_asm",
    )
    assert decision.decision == "unsupported"


def test_local_only_as_robust_detected():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Local-only execution demonstrates robust training",
        "deeplens_geolens_geometric",
        evidence_scope={"execution_target": "local"},
    )
    assert decision.decision == "needs_followup"
    assert decision.violation_type == "local_only_as_robust"


def test_check_claim_without_violation_and_rollback_accepted():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Rollback improves training stability",
        "deeplens_geolens_geometric",
        experiment_result={"accepted_update_count": 3},
    )
    # With accepted updates, rollback+improve should not trigger
    assert decision.violation_type != "rollback_protection_as_improvement"
