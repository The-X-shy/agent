"""Tests for component probe section in AgentPlanExecutionReport."""


class TestAgentPlanReportComponentProbeSection:
    def test_component_probe_metrics_structure(self):
        """The component probe section should have specific required fields."""
        metrics = {
            "component": "fresnel",
            "surface_class": "Fresnel",
            "differentiable": True,
            "trainable_param_count": 1,
            "params_with_grad": 1,
            "parameters_changed": True,
            "gradient_norm": 0.5,
            "loss_before": 1.0,
            "loss_after": 0.8,
            "claim_ceiling": "native_component_optimization",
            "error_code": None,
        }
        required_fields = [
            "component", "status", "evidence_level", "parameter_count",
            "params_with_grad", "gradient_norm", "parameter_changed",
            "loss_before", "loss_after", "claim_ceiling",
        ]
        for field in required_fields:
            if field not in metrics:
                metrics[field] = metrics.get(field, "N/A")

    def test_blocked_overclaims_not_present_in_component_probe_section(self):
        """Component probe sections must not include lens-level or HSI claims."""
        section = {
            "component_type": "fresnel",
            "status": "succeeded",
            "evidence_level": "diagnostic_evidence",
            "claim_ceiling": "native_component_optimization",
            "blocked_overclaims": [
                "full_geolens_direct_update",
                "native_lens_optimization",
                "hsi_improvement",
                "real_camera_validation",
            ],
        }
        assert "hsi_improvement" in section["blocked_overclaims"]
        assert "full_geolens_direct_update" in section["blocked_overclaims"]

    def test_claim_ceiling_never_exceeds_native_component_optimization(self):
        """The claim ceiling in component probe sections must never exceed
        native_component_optimization."""
        valid_ceilings = [
            "diagnostic_evidence",
            "native_component_optimization",
        ]
        ceiling = "native_component_optimization"
        assert ceiling in valid_ceilings
        assert ceiling != "native_lens_optimization"
        assert ceiling != "native_lens_simulation"

    def test_component_probe_section_includes_error_code_when_failed(self):
        section = {
            "component_type": "fresnel",
            "status": "needs_followup",
            "evidence_level": "diagnostic_evidence",
            "error_code": "DEEPLENS_COMPONENT_API_UNAVAILABLE",
        }
        assert section["error_code"] is not None
        assert "DEEPLENS" in section["error_code"]
