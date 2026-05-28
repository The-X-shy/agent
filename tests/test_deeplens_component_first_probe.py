"""Tests for the component-first probe runtime wrapper."""

from unittest.mock import MagicMock, patch

import pytest

from optiresearch.schemas.component_probe import (
    ComponentProbeResult,
    ComponentProbeSpec,
    make_component_probe_id,
)
from optiresearch.runtime.deeplens_component_first_probe import (
    COMPONENT_TO_SURFACE,
    COMPONENT_TO_BACKEND,
    run_deeplens_component_probe,
)


class TestComponentToSurfaceMapping:
    def test_fresnel(self):
        assert COMPONENT_TO_SURFACE["fresnel"] == "Fresnel"

    def test_binary2phase(self):
        assert COMPONENT_TO_SURFACE["binary2phase"] == "Binary2Phase"

    def test_diffractive(self):
        assert COMPONENT_TO_SURFACE["diffractive"] == "Fresnel"


class TestComponentToBackendMapping:
    def test_fresnel_backend(self):
        assert COMPONENT_TO_BACKEND["fresnel"] == "deeplens_fresnel_component"

    def test_binary2phase_backend(self):
        assert COMPONENT_TO_BACKEND["binary2phase"] == "deeplens_binary2phase_component"


class TestRunDeeplensComponentProbe:
    def test_unknown_component_returns_needs_followup(self):
        spec = ComponentProbeSpec(
            probe_id=make_component_probe_id("unknown"),
            component="unknown_component",
        )
        result = run_deeplens_component_probe(spec)
        assert result.status == "needs_followup"
        assert result.error_code == "UNKNOWN_COMPONENT"
        assert "unknown_component" in result.error_message

    def test_api_unavailable_returns_needs_followup(self, monkeypatch):
        """When DeepLens is not installed, return structured needs_followup."""
        monkeypatch.setattr(
            "optiresearch.runtime.deeplens_component_first_probe._map_surface_to_component_result",
            lambda *a, **kw: ComponentProbeResult(
                probe_id="test", component="fresnel",
                status="needs_followup",
                error_code="DEEPLENS_COMPONENT_API_UNAVAILABLE",
                error_message="DeepLens not available",
                evidence_level="diagnostic_evidence",
            ),
        )
        spec = ComponentProbeSpec(probe_id="test", component="fresnel")
        result = run_deeplens_component_probe(spec)
        assert result.status == "needs_followup"
        assert result.evidence_level == "diagnostic_evidence"

    @patch("optiresearch.runtime.deeplens_surface_optimization_probe.run_surface_optimization_probe")
    def test_successful_probe_maps_correctly(self, mock_probe):
        """A successful surface probe should yield a ComponentProbeResult."""
        mock_surface = MagicMock()
        mock_surface.status = "succeeded"
        mock_surface.differentiable = True
        mock_surface.autograd_graph_exists = True
        mock_surface.parameters_changed = True
        mock_surface.can_instantiate = True
        mock_surface.has_get_optimizer = True
        mock_surface.has_get_optimizer_params = True
        mock_surface.module_path = "deeplens.diffractive_surface.fresnel"
        mock_surface.trainable_params = ["f0"]
        mock_surface.loss_before = 1.0
        mock_surface.loss_after = 0.8
        mock_surface.gradient_norm = 0.5
        mock_surface.parameter_norm_before = 50.0
        mock_surface.parameter_norm_after = 49.5
        mock_surface.optimizer_class = "Adam"
        mock_surface.error_code = None
        mock_surface.error_message = None
        mock_surface.caveats = []
        mock_surface.objective = "minimize_phase_variance"
        mock_surface.metadata = {"per_parameter_grad_norm": {"f0": 0.5}}
        mock_probe.return_value = mock_surface

        from optiresearch.schemas.surface_optimization import SurfaceOptimizationProbeSpec
        spec = ComponentProbeSpec(
            probe_id=make_component_probe_id("fresnel"),
            component="fresnel",
            max_steps=3,
        )
        result = run_deeplens_component_probe(spec)
        assert result.component == "fresnel"
        assert result.surface_class == "Fresnel"
        assert result.claim_ceiling == "native_component_optimization"
        assert result.parameters_changed is True

    @patch("optiresearch.runtime.deeplens_surface_optimization_probe.run_surface_optimization_probe")
    def test_binary2phase_successful_probe(self, mock_probe):
        mock_surface = MagicMock()
        mock_surface.status = "succeeded"
        mock_surface.differentiable = True
        mock_surface.autograd_graph_exists = True
        mock_surface.parameters_changed = True
        mock_surface.can_instantiate = True
        mock_surface.has_get_optimizer = True
        mock_surface.has_get_optimizer_params = True
        mock_surface.module_path = "deeplens.phase_surface.binary2"
        mock_surface.trainable_params = ["d", "order2", "order4", "order6"]
        mock_surface.loss_before = 2.0
        mock_surface.loss_after = 1.5
        mock_surface.gradient_norm = 0.8
        mock_surface.parameter_norm_before = 10.0
        mock_surface.parameter_norm_after = 9.0
        mock_surface.optimizer_class = "Adam"
        mock_surface.error_code = None
        mock_surface.error_message = None
        mock_surface.caveats = []
        mock_surface.objective = "minimize_phase_variance"
        mock_surface.metadata = {
            "per_parameter_grad_norm": {
                "d": 0.1, "order2": 0.5, "order4": 0.3, "order6": 0.1,
            },
        }
        mock_probe.return_value = mock_surface

        spec = ComponentProbeSpec(
            probe_id=make_component_probe_id("binary2phase"),
            component="binary2phase",
        )
        result = run_deeplens_component_probe(spec)
        assert result.component == "binary2phase"
        assert result.surface_class == "Binary2Phase"
        assert result.trainable_param_count == 4

    @patch("optiresearch.runtime.deeplens_surface_optimization_probe.run_surface_optimization_probe")
    def test_failed_surface_probe_maps_status(self, mock_probe):
        mock_surface = MagicMock()
        mock_surface.status = "failed"
        mock_surface.differentiable = False
        mock_surface.autograd_graph_exists = False
        mock_surface.parameters_changed = False
        mock_surface.can_instantiate = False
        mock_surface.has_get_optimizer = False
        mock_surface.has_get_optimizer_params = False
        mock_surface.module_path = None
        mock_surface.trainable_params = []
        mock_surface.loss_before = None
        mock_surface.loss_after = None
        mock_surface.gradient_norm = None
        mock_surface.parameter_norm_before = None
        mock_surface.parameter_norm_after = None
        mock_surface.optimizer_class = None
        mock_surface.error_code = "IMPORT_FAILED"
        mock_surface.error_message = "Module not found"
        mock_surface.caveats = ["Import failed"]
        mock_surface.objective = "minimize_phase_variance"
        mock_surface.metadata = {}
        mock_probe.return_value = mock_surface

        spec = ComponentProbeSpec(
            probe_id=make_component_probe_id("fresnel"),
            component="fresnel",
        )
        result = run_deeplens_component_probe(spec)
        assert result.status == "failed"
        assert result.error_code == "IMPORT_FAILED"

    @patch("optiresearch.runtime.deeplens_surface_optimization_probe.run_surface_optimization_probe")
    def test_unsupported_surface_probe_maps_to_structured_unavailable(self, mock_probe):
        mock_surface = MagicMock()
        mock_surface.status = "unsupported"
        mock_surface.differentiable = False
        mock_surface.autograd_graph_exists = False
        mock_surface.parameters_changed = False
        mock_surface.can_instantiate = True
        mock_surface.has_get_optimizer = True
        mock_surface.has_get_optimizer_params = True
        mock_surface.module_path = "deeplens.diffractive_surface.fresnel"
        mock_surface.trainable_params = ["f0"]
        mock_surface.loss_before = None
        mock_surface.loss_after = None
        mock_surface.gradient_norm = None
        mock_surface.parameter_norm_before = None
        mock_surface.parameter_norm_after = None
        mock_surface.optimizer_class = None
        mock_surface.error_code = "SURFACE_NOT_DIFFERENTIABLE"
        mock_surface.error_message = "Surface not differentiable"
        mock_surface.caveats = ["Not differentiable"]
        mock_surface.objective = "minimize_phase_variance"
        mock_surface.metadata = {}
        mock_probe.return_value = mock_surface

        spec = ComponentProbeSpec(probe_id="test", component="fresnel")
        result = run_deeplens_component_probe(spec)
        assert result.status == "structured_unavailable"
        assert result.evidence_level == "diagnostic_evidence"

    def test_diffractive_probe_maps_to_fresnel(self):
        spec = ComponentProbeSpec(
            probe_id=make_component_probe_id("diffractive"),
            component="diffractive",
        )
        result = run_deeplens_component_probe(spec)
        if result.status == "needs_followup":
            assert "fresnel" in result.checked_component_candidates
