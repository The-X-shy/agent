"""Test that lightweight handler cannot claim native DeepLens simulation."""

from optiresearch.memory.claim_gate_v2 import ClaimGateV2
from optiresearch.memory.claim_ceiling_resolver import resolve_claim_ceiling


def test_lightweight_handler_ceiling_is_not_native():
    """Even with deeplens_geolens_geometric backend, lightweight handler ceiling is limited."""
    result = resolve_claim_ceiling(
        handler_id="objective_redesign_simpler_metric",
        backend_id="deeplens_geolens_geometric",
        dataset="synthetic",
        execution_fidelity="lightweight_proxy",
        synthetic_data=True,
        physical_backend=False,
        native_backend=False,
        phase_to_fft_proxy_used=True,
    )
    assert result.final_claim_ceiling == "lightweight_scientific_execution"
    assert result.final_claim_ceiling != "native_lens_simulation"
    assert result.backend_claim_ceiling == "native_lens_simulation"
    assert "no_physical_backend" in result.ceiling_source or "no_native_backend" in result.ceiling_source or result.ceiling_source in ("handler", "dataset", "execution_fidelity")


def test_claim_gate_prevents_native_claim_for_lightweight():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Native DeepLens GeoLens simulation confirms optical improvement",
        "deeplens_geolens_geometric",
        experiment_result={
            "status": "completed",
            "evidence_level": "lightweight_scientific_execution",
            "handler_id": "objective_redesign_simpler_metric",
            "synthetic_data": True,
            "physical_backend": False,
            "native_backend": False,
        },
        handler_id="objective_redesign_simpler_metric",
    )
    # Should be unsupported — lightweight handler cannot claim native
    assert decision.decision in ("unsupported", "qualified", "needs_followup")
    # Final ceiling should be lightweight, not native
    assert "native" not in decision.final_claim_ceiling.lower() or decision.final_claim_ceiling == ""


def test_claim_gate_allows_lightweight_claim():
    gate = ClaimGateV2()
    decision = gate.check_claim(
        "Lightweight scientific HSI co-design completed with synthetic metrics",
        "deeplens_geolens_geometric",
        experiment_result={
            "status": "completed",
            "evidence_level": "lightweight_scientific_execution",
            "handler_id": "objective_redesign_simpler_metric",
            "synthetic_data": True,
            "physical_backend": False,
            "native_backend": False,
        },
        handler_id="objective_redesign_simpler_metric",
        evidence_scope={"execution_target": "local"},
    )
    assert decision.decision == "supported"
    assert decision.final_claim_ceiling == "lightweight_scientific_execution"
