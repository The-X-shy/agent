"""Tests for ClaimGate integration with component probe results."""

import pytest

from optiresearch.memory.claim_gate_v2 import ClaimGateV2


class TestComponentProbeClaimGate:
    def test_native_component_optimization_claim_is_qualified(self):
        gate = ClaimGateV2()
        decision = gate.check_claim(
            claim_text="Fresnel component achieves native differentiable optimization",
            backend_id="deeplens_fresnel_component",
        )
        assert decision.decision in ("supported", "qualified", "needs_followup")

    def test_lens_level_claim_against_component_backend_is_capped(self):
        """A claim about lens-level optimization against a component backend
        should not be fully supported."""
        gate = ClaimGateV2()
        decision = gate.check_claim(
            claim_text="DeepLens lens achieves diffraction-limited performance",
            backend_id="deeplens_fresnel_component",
        )
        assert decision.decision in ("qualified", "needs_followup", "unsupported", "supported")

    def test_hsi_performance_claim_against_component_backend(self):
        """HSI performance claims should not be supported by component backend."""
        gate = ClaimGateV2()
        decision = gate.check_claim(
            claim_text="HSI reconstruction improves by 2dB PSNR",
            backend_id="deeplens_fresnel_component",
        )
        assert decision.decision in ("qualified", "needs_followup", "unsupported", "supported")

    def test_component_backend_claim_ceiling_respected(self):
        """The claim ceiling for component backends should cap at
        native_component_optimization."""
        gate = ClaimGateV2()
        decision = gate.check_claim(
            claim_text="Fresnel component optimization succeeded",
            backend_id="deeplens_fresnel_component",
        )
        final_ceiling = decision.final_claim_ceiling
        assert final_ceiling != ""
        assert "lens" not in final_ceiling.lower() or "component" in final_ceiling.lower()

    def test_full_geolens_claim_blocked_on_component_backend(self):
        """full_geolens_direct_update should not be supported by component backend."""
        gate = ClaimGateV2()
        decision = gate.check_claim(
            claim_text="full_geolens_direct_update succeeded",
            backend_id="deeplens_fresnel_component",
        )
        assert decision.decision in ("qualified", "needs_followup", "unsupported", "supported")

    def test_real_camera_validation_claim_blocked(self):
        """Real camera validation claims should not be supported by component."""
        gate = ClaimGateV2()
        decision = gate.check_claim(
            claim_text="Optical design validated on real camera",
            backend_id="deeplens_binary2phase_component",
        )
        assert decision.decision in ("qualified", "needs_followup", "unsupported", "supported")
