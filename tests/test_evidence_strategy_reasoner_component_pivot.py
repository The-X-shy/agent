"""Test EvidenceStrategyReasoner component pivot from GeoLens findings."""

from optiresearch.agents.evidence_strategy_reasoner import EvidenceStrategyReasoner


class TestComponentPivotStrategies:
    def setup_method(self):
        self.reasoner = EvidenceStrategyReasoner()

    def test_no_standard_trainable_params_generates_component_first(self):
        diag = {
            "status": "diagnosed",
            "diagnosis_id": "test",
            "failure_modes": ["no_standard_trainable_parameters"],
            "likely_causes": [],
            "recommended_recoveries": [],
            "severity": "high",
        }
        strategies = self.reasoner.reason_from_diagnosis(diag, "test")
        strategy_ids = [s.strategy_id for s in strategies]
        assert "component_first_fresnel_probe" in strategy_ids

    def test_autograd_disconnected_generates_binary2phase_and_surrogate(self):
        diag = {
            "status": "diagnosed",
            "diagnosis_id": "test",
            "failure_modes": ["autograd_graph_disconnected"],
            "likely_causes": [],
            "recommended_recoveries": [],
            "severity": "high",
        }
        strategies = self.reasoner.reason_from_diagnosis(diag, "test")
        strategy_ids = [s.strategy_id for s in strategies]
        assert "component_first_binary2phase_probe" in strategy_ids
        assert "differentiable_surrogate_psf" in strategy_ids

    def test_non_differentiable_geolens_generates_surface_adapter(self):
        diag = {
            "status": "diagnosed",
            "diagnosis_id": "test",
            "failure_modes": ["non_differentiable_geolens_psf_path"],
            "likely_causes": [],
            "recommended_recoveries": [],
            "severity": "critical",
        }
        strategies = self.reasoner.reason_from_diagnosis(diag, "test")
        strategy_ids = [s.strategy_id for s in strategies]
        assert "surface_parameter_adapter" in strategy_ids

    def test_component_strategies_have_blocked_route(self):
        diag = {
            "status": "diagnosed",
            "diagnosis_id": "test",
            "failure_modes": ["no_standard_trainable_parameters"],
            "likely_causes": [],
            "recommended_recoveries": [],
            "severity": "high",
        }
        strategies = self.reasoner.reason_from_diagnosis(diag, "test")
        component_strategies = [s for s in strategies if s.strategy_type == "component_first"]
        assert len(component_strategies) > 0
        for s in component_strategies:
            assert s.blocked_route == "full_geolens_direct_update"
            assert s.pivot_reason
            assert s.required_component_backend

    def test_new_strategy_types_are_valid(self):
        diag = {
            "status": "diagnosed",
            "diagnosis_id": "test",
            "failure_modes": ["no_standard_trainable_parameters", "autograd_graph_disconnected"],
            "likely_causes": [],
            "recommended_recoveries": [],
            "severity": "high",
        }
        strategies = self.reasoner.reason_from_diagnosis(diag, "test")
        for s in strategies:
            assert s.strategy_type in (
                "alternative_parameterization", "objective_redesign", "backend_switch",
                "waveoptics_probe", "real_data_request", "report_negative_result",
                "optimizer_change", "parameter_reduction", "autograd_audit",
                "parameter_inspection", "component_inspection", "run_probe_only",
                "component_first", "surrogate_parameterization",
            ), f"Unknown strategy_type: {s.strategy_type}"

    def test_geolens_audit_success_allows_native_geometric_direct_update(self):
        diag = {
            "status": "diagnosed",
            "diagnosis_id": "geolens-audit-success",
            "failure_modes": [],
            "likely_causes": [],
            "recommended_recoveries": [],
            "severity": "medium",
            "trainable_param_count": 14,
            "params_with_grad": 14,
            "psf_requires_grad": True,
            "loss_requires_grad": True,
            "graph_connected": True,
        }

        strategies = self.reasoner.reason_from_diagnosis(diag, "test")
        direct = [s for s in strategies if s.strategy_id == "full_geolens_geometric_direct_update"]

        assert len(direct) == 1
        assert direct[0].blocked_route == ""
        assert direct[0].claim_ceiling == "native_lens_simulation"
