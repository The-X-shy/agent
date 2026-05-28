"""Tests for Fresnel component probe."""

from unittest.mock import MagicMock, patch

import pytest

from optiresearch.schemas.component_probe import ComponentProbeSpec, make_component_probe_id
from optiresearch.runtime.deeplens_component_first_probe import run_deeplens_component_probe


class TestFresnelComponentProbe:
    @patch("optiresearch.runtime.deeplens_surface_optimization_probe.run_surface_optimization_probe")
    def test_fresnel_probe_parameter_count(self, mock_probe):
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
        mock.loss_after = 0.85
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
            probe_id=make_component_probe_id("fresnel"),
            component="fresnel",
        )
        result = run_deeplens_component_probe(spec)
        assert result.parameter_count == 1
        assert result.trainable_param_count == 1
        assert result.params_with_grad == 1

    @patch("optiresearch.runtime.deeplens_surface_optimization_probe.run_surface_optimization_probe")
    def test_fresnel_probe_gradient_norm(self, mock_probe):
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
        mock.loss_after = 0.8
        mock.gradient_norm = 0.25
        mock.parameter_norm_before = 50.0
        mock.parameter_norm_after = 49.75
        mock.optimizer_class = "Adam"
        mock.error_code = None
        mock.error_message = None
        mock.caveats = []
        mock.objective = "minimize_phase_variance"
        mock.metadata = {"per_parameter_grad_norm": {"f0": 0.25}}
        mock_probe.return_value = mock

        spec = ComponentProbeSpec(probe_id="test", component="fresnel")
        result = run_deeplens_component_probe(spec)
        assert result.gradient_norm == 0.25
        assert result.gradient_norm > 0

    @patch("optiresearch.runtime.deeplens_surface_optimization_probe.run_surface_optimization_probe")
    def test_fresnel_probe_loss_recording(self, mock_probe):
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
        mock.loss_before = 2.5
        mock.loss_after = 2.0
        mock.gradient_norm = 0.4
        mock.parameter_norm_before = 50.0
        mock.parameter_norm_after = 49.5
        mock.optimizer_class = "Adam"
        mock.error_code = None
        mock.error_message = None
        mock.caveats = []
        mock.objective = "minimize_phase_variance"
        mock.metadata = {"per_parameter_grad_norm": {"f0": 0.4}}
        mock_probe.return_value = mock

        spec = ComponentProbeSpec(probe_id="test", component="fresnel")
        result = run_deeplens_component_probe(spec)
        assert result.loss_before == 2.5
        assert result.loss_after == 2.0
        assert result.loss_after < result.loss_before

    @patch("optiresearch.runtime.deeplens_surface_optimization_probe.run_surface_optimization_probe")
    def test_fresnel_probe_native_component_optimization_evidence(self, mock_probe):
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
        mock.gradient_norm = 0.5
        mock.parameter_norm_before = 50.0
        mock.parameter_norm_after = 49.5
        mock.optimizer_class = "Adam"
        mock.error_code = None
        mock.error_message = None
        mock.caveats = []
        mock.objective = "minimize_phase_variance"
        mock.metadata = {"per_parameter_grad_norm": {"f0": 0.5}}
        mock_probe.return_value = mock

        spec = ComponentProbeSpec(probe_id="test", component="fresnel")
        result = run_deeplens_component_probe(spec)
        assert result.claim_ceiling == "native_component_optimization"
        assert result.evidence_level == "diagnostic_evidence"

    @patch("optiresearch.runtime.deeplens_surface_optimization_probe.run_surface_optimization_probe")
    def test_fresnel_probe_no_trainable_params_error(self, mock_probe):
        mock = MagicMock()
        mock.status = "unsupported"
        mock.differentiable = False
        mock.autograd_graph_exists = False
        mock.parameters_changed = False
        mock.can_instantiate = True
        mock.has_get_optimizer = False
        mock.has_get_optimizer_params = False
        mock.module_path = "deeplens.diffractive_surface.fresnel"
        mock.trainable_params = []
        mock.loss_before = None
        mock.loss_after = None
        mock.gradient_norm = None
        mock.parameter_norm_before = None
        mock.parameter_norm_after = None
        mock.optimizer_class = None
        mock.error_code = "NO_TRAINABLE_COMPONENT_PARAMETERS"
        mock.error_message = "No trainable params"
        mock.caveats = ["No params"]
        mock.objective = "minimize_phase_variance"
        mock.metadata = {}
        mock_probe.return_value = mock

        spec = ComponentProbeSpec(probe_id="test", component="fresnel")
        result = run_deeplens_component_probe(spec)
        assert result.status == "structured_unavailable"
        assert result.parameter_count == 0
