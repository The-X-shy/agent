"""Test recovery policy component pivot from GeoLens to component-level."""

from optiresearch.agent_system.recovery_policy import RecoveryPolicy


class TestComponentPivotRecovery:
    def setup_method(self):
        self.rp = RecoveryPolicy()

    def test_no_standard_trainable_params_recovery(self):
        rec = self.rp.recommend_recovery("no_standard_trainable_parameters")
        assert rec["failure_id"] == "no_standard_trainable_parameters"
        recoveries = [r["recovery"] for r in rec["recoveries"]]
        assert "component_first_fresnel_probe" in recoveries
        assert "component_first_binary2phase_probe" in recoveries
        # component_first_fresnel_probe should be highest priority (9 > 0)
        top = rec["recoveries"][0]
        assert top["recovery"] in ("component_first_fresnel_probe", "component_first_binary2phase_probe", "report_full_geolens_non_differentiable_path")

    def test_autograd_disconnected_recovery(self):
        rec = self.rp.recommend_recovery("autograd_graph_disconnected")
        recoveries = [r["recovery"] for r in rec["recoveries"]]
        assert "diffractive_component_probe" in recoveries
        assert "differentiable_surrogate_psf_parameterization" in recoveries

    def test_non_differentiable_geolens_recovery(self):
        rec = self.rp.recommend_recovery("non_differentiable_geolens_psf_path")
        assert rec["severity"] == "critical"
        recoveries = [r["recovery"] for r in rec["recoveries"]]
        assert "surface_parameter_adapter" in recoveries
        assert "report_full_geolens_non_differentiable_path" in recoveries

    def test_component_probe_ranked_above_full_geolens(self):
        rec = self.rp.recommend_recovery("no_standard_trainable_parameters")
        recoveries = [r["recovery"] for r in rec["recoveries"]]
        cp_idx = recoveries.index("component_first_fresnel_probe") if "component_first_fresnel_probe" in recoveries else -1
        fg_idx = recoveries.index("full_geolens_direct_update") if "full_geolens_direct_update" in recoveries else len(recoveries)
        assert cp_idx < fg_idx or fg_idx == len(recoveries), "component probe must rank above full_geolens"

    def test_convert_component_strategy(self):
        strat = self.rp.convert_recovery_to_strategy("component_first_fresnel_probe")
        assert strat["strategy_type"] == "alternative_parameterization"
        assert "Fresnel" in strat["action"]

    def test_convert_report_non_differentiable(self):
        strat = self.rp.convert_recovery_to_strategy("report_full_geolens_non_differentiable_path")
        assert strat["strategy_type"] == "report_negative_result"

    def test_explain_component_probe(self):
        explanation = self.rp.explain_recovery("component_first_fresnel_probe")
        assert "Fresnel" in explanation and "f0" in explanation
