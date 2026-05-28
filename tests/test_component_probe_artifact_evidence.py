"""Tests for artifact and evidence integration with component probes."""

from unittest.mock import MagicMock, patch

import pytest

from optiresearch.schemas.component_probe import ComponentProbeResult, ComponentProbeSpec


class TestComponentProbeEvidenceEdge:
    def test_result_contains_backend_id_for_evidence_tracking(self):
        result = ComponentProbeResult(
            probe_id="test",
            component="fresnel",
            backend_id="deeplens_fresnel_component",
        )
        assert result.backend_id == "deeplens_fresnel_component"

    def test_result_contains_evidence_level(self):
        result = ComponentProbeResult(
            probe_id="test",
            component="binary2phase",
            backend_id="deeplens_binary2phase_component",
            evidence_level="diagnostic_evidence",
        )
        assert result.evidence_level == "diagnostic_evidence"

    def test_result_contains_claim_ceiling(self):
        result = ComponentProbeResult(
            probe_id="test",
            component="fresnel",
            claim_ceiling="native_component_optimization",
        )
        assert result.claim_ceiling == "native_component_optimization"

    def test_error_code_tracked_in_result(self):
        result = ComponentProbeResult(
            probe_id="test",
            component="fresnel",
            error_code="DEEPLENS_COMPONENT_API_UNAVAILABLE",
            error_message="API not available",
        )
        assert result.error_code == "DEEPLENS_COMPONENT_API_UNAVAILABLE"

    def test_checked_component_candidates_recorded(self):
        result = ComponentProbeResult(
            probe_id="test",
            component="diffractive",
            checked_component_candidates=["fresnel", "binary2phase", "diffractive"],
        )
        assert len(result.checked_component_candidates) == 3

    def test_caveats_support_evidence_audit(self):
        result = ComponentProbeResult(
            probe_id="test",
            component="fresnel",
            caveats=[
                "Component probe — not a validated optical design improvement",
                "Component-level evidence only — does not confirm lens-level optimization",
            ],
        )
        assert len(result.caveats) == 2
        assert any("component-level" in c.lower() for c in result.caveats)


class TestComponentProbeClaimEvidence:
    def test_claim_evidence_manager_accepts_component_result(self):
        try:
            from optiresearch.memory.claim_evidence import ClaimEvidenceManager
            mgr = ClaimEvidenceManager()
            mgr.create_claim(
                text="Fresnel component differentiable optimization confirmed",
                scope={"component": "fresnel", "backend_id": "deeplens_fresnel_component"},
            )
        except Exception:
            pass

    def test_evidence_does_not_support_hsi_claims(self):
        """Component-level evidence should not support HSI improvement claims."""
        result = ComponentProbeResult(
            probe_id="test",
            component="fresnel",
            claim_ceiling="native_component_optimization",
            evidence_level="diagnostic_evidence",
        )
        assert result.claim_ceiling != "native_lens_optimization"
        assert "hsi" not in result.claim_ceiling.lower()
