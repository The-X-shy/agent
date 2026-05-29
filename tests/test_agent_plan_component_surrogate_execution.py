"""Agent plan execution tests for component surrogate HSI co-design."""

from optiresearch.agents.experiment_design_generator import ExperimentDesignCandidate
from optiresearch.runtime.agent_plan_execution_loop import _execute_design, _run_claim_gate


def _design(component="fresnel"):
    return ExperimentDesignCandidate(
        design_id=f"component_surrogate_{component}_hsi_codesign_design",
        objective="Run component surrogate HSI co-design",
        backend_id="component_surrogate_psf",
        task_type="component_surrogate_hsi_codesign",
        spec_payload={
            "component": component,
            "dataset": "synthetic",
            "steps": 3,
            "bands": 4,
            "image_size": 16,
            "psf_size": 9,
        },
        expected_evidence_level="component_surrogate_hsi_codesign",
        claim_ceiling="component_surrogate_hsi_codesign",
        handler_id="component_surrogate_hsi_codesign",
    )


def test_execute_component_surrogate_design_returns_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = _execute_design(_design("fresnel"))

    assert result["status"] == "completed"
    assert result["handler_id"] == "component_surrogate_hsi_codesign"
    assert result["evidence_level"] == "component_surrogate_hsi_codesign"
    assert result["metrics"]["component_grad_norm_max"] > 0
    assert result["metrics"]["component_parameter_changed"] is True
    assert result["metrics"]["psf_requires_grad"] is True
    assert result["metrics"]["loss_requires_grad"] is True
    assert result["handler_claim_ceiling"] == "component_surrogate_hsi_codesign"
    assert result["synthetic_data"] is True
    assert result["native_backend"] is False


def test_component_surrogate_claim_gate_decision_is_bounded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    design = _design("binary2phase")
    result = _execute_design(design)

    decision = _run_claim_gate(design, result)

    assert decision["decision"] == "supported"
    assert decision["final_claim_ceiling"] == "component_surrogate_hsi_codesign"
    assert decision["max_allowed_claim"] == "component_surrogate_hsi_codesign"


def test_geolens_unavailable_degrades_to_component_surrogate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from optiresearch.runtime import agent_plan_execution_loop as loop

    design = ExperimentDesignCandidate(
        design_id="full_geolens_geometric_direct_update_design",
        objective="Run native GeoLens geometric HSI co-design",
        backend_id="deeplens_geolens_geometric",
        task_type="stable_lens_hsi_codesign",
        spec_payload={"candidate": "GeoLensCooke", "dataset": "synthetic", "max_steps": 3},
        expected_evidence_level="native_lens_simulation",
        claim_ceiling="native_lens_simulation",
        handler_id="deeplens_native_geolens_hsi_codesign",
    )
    seen = {}

    def fake_component_surrogate(fallback_design):
        seen["design"] = fallback_design
        return {
            "status": "completed",
            "design_id": fallback_design.design_id,
            "task_type": fallback_design.task_type,
            "backend_id": fallback_design.backend_id,
            "evidence_level": "component_surrogate_hsi_codesign",
            "metrics": {"component_parameter_changed": True},
        }

    monkeypatch.setattr(loop, "_deeplens_available", lambda: False)
    monkeypatch.setattr(loop, "_execute_component_surrogate_hsi_design", fake_component_surrogate)

    result = _execute_design(design)

    assert result["status"] == "completed"
    assert result["backend_id"] == "component_surrogate_psf"
    assert result["design_id"].startswith("component_surrogate_fallback_")
    assert seen["design"].spec_payload["component"] == "fresnel"
    assert seen["design"].spec_payload["steps"] == 3
