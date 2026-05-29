"""Test failure taxonomy with GeoLens autograd diagnostic findings."""

from optiresearch.agent_system.failure_taxonomy import FailureClassifier


class TestGeoLensAutogradFailureModes:
    def setup_method(self):
        self.fc = FailureClassifier()

    def test_no_standard_trainable_parameters(self):
        result = {"parameter_count": 0, "trainable_param_count": 0, "status": "succeeded"}
        fm = self.fc.classify(result)
        assert fm is not None
        assert fm.failure_id == "no_standard_trainable_parameters"
        assert fm.category == "autograd_break"
        assert fm.severity == "high"
        assert "component_first_fresnel_probe" in fm.recommended_recoveries

    def test_autograd_graph_disconnected(self):
        result = {
            "graph_connected": False, "psf_requires_grad": False,
            "loss_requires_grad": False, "status": "succeeded",
        }
        fm = self.fc.classify(result)
        assert fm is not None
        assert fm.failure_id == "autograd_graph_disconnected"
        assert fm.category == "autograd_break"
        assert "component_first_binary2phase_probe" in fm.recommended_recoveries

    def test_non_differentiable_geolens_psf_path(self):
        result = {"graph_connected": False, "parameter_count": 0, "status": "succeeded"}
        fm = self.fc.classify(result)
        assert fm is not None
        assert fm.failure_id == "non_differentiable_geolens_psf_path"
        assert fm.category == "gradient_instability"
        assert fm.severity == "critical"
        assert "blocked unless a native optimizer audit passes" in fm.claim_impact

    def test_phase60_autograd_audit_result_matches_all_three(self):
        """The Phase 60 autograd audit result should match multiple failure modes."""
        result = {
            "status": "succeeded",
            "graph_connected": False,
            "psf_requires_grad": False,
            "loss_requires_grad": False,
            "parameter_count": 0,
            "trainable_param_count": 0,
            "params_with_grad": 0,
            "grad_norm_max": 0.0,
        }
        fm = self.fc.classify(result)
        assert fm is not None
        # First match wins — should be no_standard_trainable_parameters
        assert fm.failure_id == "no_standard_trainable_parameters"

    def test_unstable_native_geolens_update_still_works(self):
        result = {
            "optical_gradient_norm": 2000.0,
            "accepted_update_count": 0,
            "rollback_count": 3,
            "proxy_fallback_used": False,
            "configs_tested": 6,
        }
        fm = self.fc.classify(result)
        assert fm is not None
        assert fm.failure_id == "unstable_native_geolens_update"

    def test_normal_result_does_not_match(self):
        result = {"status": "succeeded", "loss_improved": True, "parameter_count": 100}
        fm = self.fc.classify(result)
        assert fm is None

    def test_classify_by_id(self):
        fm = self.fc.classify_by_id("autograd_graph_disconnected")
        assert fm is not None
        assert fm.failure_id == "autograd_graph_disconnected"
        assert fm.severity == "high"

    def test_total_failure_modes(self):
        modes = self.fc.list_all()
        assert len(modes) >= 11
