"""Test ClaimGateV2 handling of lightweight scientific execution evidence."""

from optiresearch.memory.claim_gate_v2 import ClaimGateV2


def test_lightweight_scientific_passes_modest_claim():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Lightweight scientific HSI co-design completed with MSE-only objective",
        "phase_to_fft_proxy",
        experiment_result={
            "status": "completed",
            "evidence_level": "lightweight_scientific_execution",
            "synthetic_data": True,
            "physical_backend": False,
            "mse_only_objective": True,
        },
        evidence_scope={"execution_target": "local"},
    )
    assert decision.decision == "supported"
    assert decision.max_allowed_claim is not None


def test_lightweight_as_native_physical_detected():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Native DeepLens simulation confirms optical improvement",
        "phase_to_fft_proxy",
        experiment_result={
            "status": "completed",
            "evidence_level": "lightweight_scientific_execution",
            "synthetic_data": True,
            "physical_backend": False,
        },
    )
    assert decision.decision == "unsupported"
    assert decision.violation_type == "lightweight_as_native_physical"


def test_lightweight_as_native_deeplens_claim_detected():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "DeepLens native GeoLens simulation validated with physical optics",
        "phase_to_fft_proxy",
        experiment_result={
            "status": "completed",
            "evidence_level": "lightweight_scientific_execution",
            "synthetic_data": True,
        },
    )
    assert decision.decision == "unsupported"
    assert decision.violation_type == "lightweight_as_native_physical"


def test_synthetic_metric_as_real_hsi_detected():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Real HSI performance demonstrated by synthetic metrics",
        "phase_to_fft_proxy",
        experiment_result={
            "status": "completed",
            "evidence_level": "lightweight_scientific_execution",
            "synthetic_data": True,
        },
    )
    assert decision.decision == "unsupported"
    assert decision.violation_type == "synthetic_metric_as_real_hsi"


def test_lightweight_scientific_allows_synthetic_improvement_claim():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "MSE-only optimization improves HSI reconstruction in synthetic experiment",
        "phase_to_fft_proxy",
        experiment_result={
            "status": "completed",
            "evidence_level": "lightweight_scientific_execution",
            "synthetic_data": True,
            "reconstruction_loss_after": 0.05,
            "reconstruction_loss_before": 0.1,
        },
    )
    # Improvement claim about synthetic experiment is fine — no native/physical wording
    assert decision.decision == "supported"


def test_synthetic_metric_as_real_measurement_hsi_detected():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Real measurement HSI data confirms the optical design works",
        "phase_to_fft_proxy",
        experiment_result={
            "status": "completed",
            "evidence_level": "lightweight_scientific_execution",
            "synthetic_data": True,
        },
    )
    assert decision.decision == "unsupported"
    assert decision.violation_type == "synthetic_metric_as_real_hsi"


def test_lightweight_execution_caveats_present():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Native physical optical performance validated",
        "phase_to_fft_proxy",
        experiment_result={
            "status": "completed",
            "evidence_level": "lightweight_scientific_execution",
            "synthetic_data": True,
        },
    )
    assert decision.decision == "unsupported"
    assert decision.applicable_caveats


def test_lightweight_claim_ceiling_is_correct():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Lightweight scientific HSI co-design completed successfully",
        "phase_to_fft_proxy",
        experiment_result={
            "status": "completed",
            "evidence_level": "lightweight_scientific_execution",
            "synthetic_data": True,
            "physical_backend": False,
        },
        evidence_scope={"execution_target": "local"},
    )
    assert decision.decision == "supported"
    # claim_ceiling should come from the backend registry for phase_to_fft_proxy
    assert decision.max_allowed_claim is not None
