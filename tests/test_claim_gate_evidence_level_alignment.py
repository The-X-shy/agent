"""Test ClaimGate evidence level overestimate detection."""

from optiresearch.memory.claim_gate_v2 import ClaimGateV2


def test_evidence_overestimated_detected():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Native lens simulation completed successfully",
        "phase_to_fft_proxy",
        experiment_result={
            "status": "completed",
            "evidence_level": "lightweight_scientific_execution",
            "expected_evidence_level": "native_lens_simulation",
            "actual_handler_evidence_level": "lightweight_scientific_execution",
        },
    )
    # Should have some violation — either evidence_level_overestimated or lightweight_as_native_physical
    assert decision.violation_type is not None


def test_aligned_evidence_no_overestimate():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Lightweight scientific HSI co-design completed with MSE-only objective",
        "phase_to_fft_proxy",
        experiment_result={
            "status": "completed",
            "evidence_level": "lightweight_scientific_execution",
            "expected_evidence_level": "lightweight_scientific_execution",
            "actual_handler_evidence_level": "lightweight_scientific_execution",
        },
        evidence_scope={"execution_target": "local"},
    )
    assert decision.decision == "supported"


def test_handler_capability_exceeded_triggers():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Native lens simulation with real HSI performance demonstrated",
        "phase_to_fft_proxy",
        experiment_result={
            "status": "completed",
            "evidence_level": "lightweight_scientific_execution",
            "claim_ceiling": "lightweight_scientific_execution",
            "synthetic_data": True,
        },
    )
    # "real HSI" in claim + lightweight ceiling = violation
    assert decision.violation_type is not None
