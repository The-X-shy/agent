"""Tests for Binary2Phase component probe."""

from unittest.mock import MagicMock, patch

from optiresearch.schemas.component_probe import ComponentProbeSpec, make_component_probe_id
from optiresearch.runtime.deeplens_component_first_probe import run_deeplens_component_probe


class TestBinary2PhaseComponentProbe:
    @patch("optiresearch.runtime.deeplens_surface_optimization_probe.run_surface_optimization_probe")
    def test_binary2phase_parameter_count(self, mock_probe):
        mock = MagicMock()
        mock.status = "succeeded"
        mock.differentiable = True
        mock.autograd_graph_exists = True
        mock.parameters_changed = True
        mock.can_instantiate = True
        mock.has_get_optimizer = True
        mock.has_get_optimizer_params = True
        mock.module_path = "deeplens.phase_surface.binary2"
        mock.trainable_params = ["d", "order2", "order4", "order6", "order8", "order10", "order12"]
        mock.loss_before = 2.0
        mock.loss_after = 1.5
        mock.gradient_norm = 0.9
        mock.parameter_norm_before = 10.0
        mock.parameter_norm_after = 8.5
        mock.optimizer_class = "Adam"
        mock.error_code = None
        mock.error_message = None
        mock.caveats = []
        mock.objective = "minimize_phase_variance"
        mock.metadata = {"per_parameter_grad_norm": {
            "d": 0.05, "order2": 0.4, "order4": 0.3,
            "order6": 0.2, "order8": 0.0, "order10": 0.0, "order12": 0.0,
        }}
        mock_probe.return_value = mock

        spec = ComponentProbeSpec(
            probe_id=make_component_probe_id("binary2phase"),
            component="binary2phase",
        )
        result = run_deeplens_component_probe(spec)
        assert result.parameter_count == 7
        assert result.trainable_param_count == 7

    @patch("optiresearch.runtime.deeplens_surface_optimization_probe.run_surface_optimization_probe")
    def test_binary2phase_gradient_across_orders(self, mock_probe):
        mock = MagicMock()
        mock.status = "succeeded"
        mock.differentiable = True
        mock.autograd_graph_exists = True
        mock.parameters_changed = True
        mock.can_instantiate = True
        mock.has_get_optimizer = True
        mock.has_get_optimizer_params = True
        mock.module_path = "deeplens.phase_surface.binary2"
        mock.trainable_params = ["d", "order2", "order4", "order6"]
        mock.loss_before = 3.0
        mock.loss_after = 2.2
        mock.gradient_norm = 1.2
        mock.parameter_norm_before = 10.0
        mock.parameter_norm_after = 8.0
        mock.optimizer_class = "Adam"
        mock.error_code = None
        mock.error_message = None
        mock.caveats = []
        mock.objective = "minimize_phase_variance"
        mock.metadata = {
            "per_parameter_grad_norm": {"d": 0.1, "order2": 0.6, "order4": 0.5, "order6": 0.2},
        }
        mock_probe.return_value = mock

        spec = ComponentProbeSpec(probe_id="test", component="binary2phase")
        result = run_deeplens_component_probe(spec)
        assert result.gradient_norm == 1.2
        assert result.params_with_grad == 4

    @patch("optiresearch.runtime.deeplens_surface_optimization_probe.run_surface_optimization_probe")
    def test_binary2phase_zero_gradient_detection(self, mock_probe):
        mock = MagicMock()
        mock.status = "succeeded"
        mock.differentiable = True
        mock.autograd_graph_exists = True
        mock.parameters_changed = True
        mock.can_instantiate = True
        mock.has_get_optimizer = True
        mock.has_get_optimizer_params = True
        mock.module_path = "deeplens.phase_surface.binary2"
        mock.trainable_params = ["d", "order2", "order4", "order6", "order8", "order10", "order12"]
        mock.loss_before = 2.0
        mock.loss_after = 1.5
        mock.gradient_norm = 0.8
        mock.parameter_norm_before = 10.0
        mock.parameter_norm_after = 9.0
        mock.optimizer_class = "Adam"
        mock.error_code = None
        mock.error_message = None
        mock.caveats = []
        mock.objective = "minimize_phase_variance"
        mock.metadata = {"per_parameter_grad_norm": {
            "d": 0.05, "order2": 0.4, "order4": 0.3,
            "order6": 0.1, "order8": None, "order10": None, "order12": None,
        }}
        mock_probe.return_value = mock

        spec = ComponentProbeSpec(probe_id="test", component="binary2phase")
        result = run_deeplens_component_probe(spec)
        assert len(result.zero_gradient_parameters) > 0
        assert "order8" in result.zero_gradient_parameters or any(
            "order" in z for z in result.zero_gradient_parameters
        )

    @patch("optiresearch.runtime.deeplens_surface_optimization_probe.run_surface_optimization_probe")
    def test_binary2phase_native_component_optimization(self, mock_probe):
        mock = MagicMock()
        mock.status = "succeeded"
        mock.differentiable = True
        mock.autograd_graph_exists = True
        mock.parameters_changed = True
        mock.can_instantiate = True
        mock.has_get_optimizer = True
        mock.has_get_optimizer_params = True
        mock.module_path = "deeplens.phase_surface.binary2"
        mock.trainable_params = ["d", "order2", "order4"]
        mock.loss_before = 1.5
        mock.loss_after = 1.0
        mock.gradient_norm = 0.7
        mock.parameter_norm_before = 10.0
        mock.parameter_norm_after = 8.0
        mock.optimizer_class = "Adam"
        mock.error_code = None
        mock.error_message = None
        mock.caveats = []
        mock.objective = "minimize_phase_variance"
        mock.metadata = {"per_parameter_grad_norm": {"d": 0.1, "order2": 0.5, "order4": 0.4}}
        mock_probe.return_value = mock

        spec = ComponentProbeSpec(probe_id="test", component="binary2phase")
        result = run_deeplens_component_probe(spec)
        assert result.claim_ceiling == "native_component_optimization"
        assert result.parameters_changed is True

    @patch("optiresearch.runtime.deeplens_surface_optimization_probe.run_surface_optimization_probe")
    def test_binary2phase_constructor_incompatible(self, mock_probe):
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
        mock.error_code = "FRESNEL_CONSTRUCTOR_INCOMPATIBLE"
        mock.error_message = "Constructor not compatible"
        mock.caveats = ["Incompatible constructor"]
        mock.objective = "minimize_phase_variance"
        mock.metadata = {}
        mock_probe.return_value = mock

        spec = ComponentProbeSpec(probe_id="test", component="binary2phase")
        result = run_deeplens_component_probe(spec)
        assert result.status == "structured_unavailable"
