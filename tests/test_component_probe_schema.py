"""Tests for ComponentProbeSpec and ComponentProbeResult schemas."""

import json

import pytest

from optiresearch.schemas.component_probe import (
    ComponentProbeResult,
    ComponentProbeSpec,
    make_component_probe_id,
)


class TestComponentProbeSpec:
    def test_valid_spec_construction(self):
        spec = ComponentProbeSpec(
            probe_id="test_probe_01",
            component="fresnel",
            objective="parameter_sanity_check",
            max_steps=5,
            device="cpu",
        )
        assert spec.component == "fresnel"
        assert spec.max_steps == 5
        assert spec.learning_rate == 1e-3

    def test_default_objective(self):
        spec = ComponentProbeSpec(probe_id="p1", component="binary2phase")
        assert spec.objective == "parameter_sanity_check"

    def test_invalid_component_allowed_by_schema(self):
        """Schema validates min_length only — component validation is at runtime."""
        spec = ComponentProbeSpec(probe_id="p1", component="unknown_component")
        assert spec.component == "unknown_component"

    def test_max_steps_bounds(self):
        with pytest.raises(Exception):
            ComponentProbeSpec(probe_id="p1", component="fresnel", max_steps=0)
        with pytest.raises(Exception):
            ComponentProbeSpec(probe_id="p1", component="fresnel", max_steps=101)

    def test_learning_rate_positive(self):
        with pytest.raises(Exception):
            ComponentProbeSpec(probe_id="p1", component="fresnel", learning_rate=0)
        with pytest.raises(Exception):
            ComponentProbeSpec(probe_id="p1", component="fresnel", learning_rate=-0.1)

    def test_serialization_roundtrip(self):
        spec = ComponentProbeSpec(
            probe_id="test_probe_02",
            component="diffractive",
            objective="minimize_phase_variance",
            max_steps=3,
            device="cpu",
        )
        data = spec.model_dump(mode="json")
        reloaded = ComponentProbeSpec(**data)
        assert reloaded.probe_id == spec.probe_id
        assert reloaded.component == spec.component

    def test_default_values_populated(self):
        spec = ComponentProbeSpec(probe_id="p1", component="fresnel")
        assert spec.device == "cpu"
        assert spec.max_steps == 5
        assert spec.save_artifacts is True
        assert spec.metadata == {}


class TestComponentProbeResult:
    def test_default_construction(self):
        result = ComponentProbeResult(probe_id="p1", component="fresnel")
        assert result.status == "failed"
        assert result.evidence_level == "diagnostic_evidence"
        assert result.trainable_param_count == 0

    def test_succeeded_result(self):
        result = ComponentProbeResult(
            probe_id="p1",
            component="fresnel",
            status="succeeded",
            surface_class="Fresnel",
            backend_id="deeplens_fresnel_component",
            differentiable=True,
            autograd_graph_exists=True,
            parameters_changed=True,
            parameter_count=1,
            trainable_param_count=1,
            params_with_grad=1,
            gradient_norm=0.5,
            loss_before=1.0,
            loss_after=0.8,
            evidence_level="diagnostic_evidence",
            claim_ceiling="native_component_optimization",
        )
        assert result.claim_ceiling == "native_component_optimization"
        assert result.differentiable is True

    def test_serialization(self):
        result = ComponentProbeResult(
            probe_id="p1",
            component="binary2phase",
            status="needs_followup",
            surface_class="Binary2Phase",
            error_code="NO_TRAINABLE_COMPONENT_PARAMETERS",
            error_message="No trainable parameters found",
            checked_component_candidates=["fresnel", "binary2phase", "diffractive"],
        )
        data = result.model_dump(mode="json")
        assert data["error_code"] == "NO_TRAINABLE_COMPONENT_PARAMETERS"
        assert len(data["checked_component_candidates"]) == 3

    def test_trainable_param_names_serialization(self):
        result = ComponentProbeResult(
            probe_id="p1",
            component="binary2phase",
            trainable_param_names=["d", "order2", "order4"],
        )
        data = json.loads(json.dumps(result.model_dump(mode="json")))
        assert "d" in data["trainable_param_names"]

    def test_zero_gradient_parameters(self):
        result = ComponentProbeResult(
            probe_id="p1",
            component="binary2phase",
            zero_gradient_parameters=["d", "order12"],
        )
        assert "d" in result.zero_gradient_parameters


class TestMakeComponentProbeId:
    def test_deterministic(self):
        id1 = make_component_probe_id("fresnel", "parameter_sanity_check")
        id2 = make_component_probe_id("fresnel", "parameter_sanity_check")
        assert id1 == id2

    def test_different_components_yield_different_ids(self):
        id1 = make_component_probe_id("fresnel")
        id2 = make_component_probe_id("binary2phase")
        assert id1 != id2

    def test_id_format(self):
        pid = make_component_probe_id("fresnel")
        assert pid.startswith("comp_probe_")
