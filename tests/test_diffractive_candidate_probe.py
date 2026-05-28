"""Tests for diffractive candidate probe."""

from unittest.mock import MagicMock, patch

from optiresearch.schemas.component_probe import ComponentProbeSpec, make_component_probe_id
from optiresearch.runtime.deeplens_component_first_probe import run_deeplens_component_probe


class TestDiffractiveCandidateProbe:
    @patch("optiresearch.runtime.deeplens_surface_optimization_probe.run_surface_optimization_probe")
    def test_diffractive_probe_availability_check(self, mock_probe):
        mock = MagicMock()
        mock.status = "succeeded"
        mock.differentiable = True
        mock.autograd_graph_exists = True
        mock.parameters_changed = True
        mock.can_instantiate = True
        mock.has_get_optimizer = True
        mock.has_get_optimizer_params = True
        mock.module_path = "deeplens.diffractive_surface.fresnel"
        mock.trainable_params = ["f0"]
        mock.loss_before = 1.0
        mock.loss_after = 0.9
        mock.gradient_norm = 0.3
        mock.parameter_norm_before = 50.0
        mock.parameter_norm_after = 49.7
        mock.optimizer_class = "Adam"
        mock.error_code = None
        mock.error_message = None
        mock.caveats = []
        mock.objective = "minimize_phase_variance"
        mock.metadata = {"per_parameter_grad_norm": {"f0": 0.3}}
        mock_probe.return_value = mock

        spec = ComponentProbeSpec(
            probe_id=make_component_probe_id("diffractive"),
            component="diffractive",
        )
        result = run_deeplens_component_probe(spec)
        assert result.component == "diffractive"
        assert result.surface_class == "Fresnel"
        assert result.module_path is not None

    @patch("optiresearch.runtime.deeplens_surface_optimization_probe.run_surface_optimization_probe")
    def test_diffractive_probe_needs_followup_when_unavailable(self, mock_probe):
        mock = MagicMock()
        mock.status = "unsupported"
        mock.differentiable = False
        mock.autograd_graph_exists = False
        mock.parameters_changed = False
        mock.can_instantiate = False
        mock.has_get_optimizer = False
        mock.has_get_optimizer_params = False
        mock.module_path = None
        mock.trainable_params = []
        mock.loss_before = None
        mock.loss_after = None
        mock.gradient_norm = None
        mock.parameter_norm_before = None
        mock.parameter_norm_after = None
        mock.optimizer_class = None
        mock.error_code = "IMPORT_FAILED"
        mock.error_message = "Module not found"
        mock.caveats = ["Import failed"]
        mock.objective = "minimize_phase_variance"
        mock.metadata = {}
        mock_probe.return_value = mock

        spec = ComponentProbeSpec(probe_id="test", component="diffractive")
        result = run_deeplens_component_probe(spec)
        assert result.status == "structured_unavailable"
        assert result.evidence_level == "diagnostic_evidence"

    def test_diffractive_checked_candidates(self):
        spec = ComponentProbeSpec(
            probe_id=make_component_probe_id("diffractive"),
            component="diffractive",
        )
        result = run_deeplens_component_probe(spec)
        if result.status == "needs_followup":
            assert len(result.checked_component_candidates) > 0
