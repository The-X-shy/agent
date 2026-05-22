"""Tests for ExperimentControllerV2."""

from optiresearch.runtime.experiment_controller_v2 import (
    _CLAIM_LEVELS,
    _TASK_REQUIRED_CEILING,
    _claim_level_index,
    ControllerResult,
    ExperimentControllerV2,
    ExperimentSpecV2,
)


def test_claim_level_ordering():
    assert _claim_level_index("unsupported") == 0
    assert _claim_level_index("real_hsi_performance") == 10
    assert _claim_level_index("native_lens_simulation") < _claim_level_index("native_waveoptics")


def test_claim_level_unknown():
    assert _claim_level_index("nonexistent") == -1


def test_controller_downgrades_insufficient_backend():
    ctrl = ExperimentControllerV2()
    spec = ExperimentSpecV2(
        spec_id="test_waveoptics_on_geometric",
        task_type="native_waveoptics_codesign",
        backend_id="deeplens_geolens_geometric",
    )
    result = ctrl.run_local(spec)
    assert result.status == "claim_downgraded"
    assert result.downgraded_from == "native_waveoptics"
    assert result.downgraded_to == "native_lens_simulation"


def test_controller_valid_backend_passes_preconditions():
    ctrl = ExperimentControllerV2()
    spec = ExperimentSpecV2(
        spec_id="test_stable_on_geometric",
        task_type="stable_lens_hsi_codesign",
        backend_id="deeplens_geolens_geometric",
    )
    issues = ctrl.validate_preconditions(spec)
    assert len(issues) == 0


def test_controller_unknown_backend():
    ctrl = ExperimentControllerV2()
    spec = ExperimentSpecV2(
        spec_id="test_unknown",
        task_type="stable_lens_hsi_codesign",
        backend_id="nonexistent",
    )
    issues = ctrl.validate_preconditions(spec)
    assert len(issues) > 0
    assert any("Unknown backend" in i for i in issues)


def test_controller_plan_experiment():
    ctrl = ExperimentControllerV2()
    spec = ctrl.plan_experiment(
        "test objective",
        "deeplens_geolens_geometric",
        "stable_lens_hsi_codesign",
    )
    assert spec.backend_id == "deeplens_geolens_geometric"
    assert spec.task_type == "stable_lens_hsi_codesign"
    assert spec.spec_id.startswith("v2_")


def test_controller_collect_artifacts(tmp_path):
    ctrl = ExperimentControllerV2()
    artifacts = ctrl.collect_artifacts("nonexistent_run_id")
    assert isinstance(artifacts, list)


def test_controller_evaluate_metrics():
    ctrl = ExperimentControllerV2()
    result = ControllerResult(
        spec_id="test",
        status="succeeded",
        backend_id="deeplens_geolens_geometric",
        result_payload={
            "reconstruction_loss_before": 1.0,
            "reconstruction_loss_after": 0.5,
            "optical_gradient_norm": 10.0,
            "rollback_count": 0,
            "accepted_update_count": 5,
        },
    )
    metrics = ctrl.evaluate_metrics(result)
    assert metrics["reconstruction_loss_before"] == 1.0
    assert metrics["reconstruction_loss_after"] == 0.5


def test_controller_update_memory():
    ctrl = ExperimentControllerV2()
    result = ControllerResult(
        spec_id="test_mem",
        status="succeeded",
        backend_id="deeplens_geolens_geometric",
        evidence_level="native_lens_simulation",
    )
    # Should not raise
    ctrl.update_memory(result)


def test_controller_update_claim_evidence():
    ctrl = ExperimentControllerV2()
    result = ControllerResult(
        spec_id="test",
        status="succeeded",
        evidence_level="native_lens_simulation",
    )
    claim = ctrl.update_claim_evidence(result)
    assert "native_lens_simulation" in claim


def test_controller_recommend_next_action():
    ctrl = ExperimentControllerV2()
    result = ControllerResult(
        spec_id="test",
        status="succeeded",
        backend_id="deeplens_geolens_geometric",
        result_payload={"optical_gradient_norm": 500},
    )
    rec = ctrl.recommend_next_action(result)
    assert "action" in rec
    assert "rationale" in rec


def test_task_required_ceiling_mapping():
    assert _TASK_REQUIRED_CEILING["stable_lens_hsi_codesign"] == "native_lens_simulation"
    assert _TASK_REQUIRED_CEILING["native_waveoptics_codesign"] == "native_waveoptics"
    assert _TASK_REQUIRED_CEILING["psf_probe"] == "deeplens_integration_smoke"


def test_claim_levels_list_is_ordered():
    for i in range(len(_CLAIM_LEVELS) - 1):
        assert _claim_level_index(_CLAIM_LEVELS[i]) < _claim_level_index(_CLAIM_LEVELS[i + 1])


def test_controller_validates_waveoptics_support():
    ctrl = ExperimentControllerV2()
    spec = ExperimentSpecV2(
        spec_id="test",
        task_type="native_waveoptics_codesign",
        backend_id="deeplens_geolens_geometric",
    )
    issues = ctrl.validate_preconditions(spec)
    assert len(issues) > 0
    assert any("full wave-optics" in i for i in issues)


def test_controller_validates_remote_support():
    ctrl = ExperimentControllerV2()
    spec = ExperimentSpecV2(
        spec_id="test",
        task_type="stable_lens_hsi_codesign",
        backend_id="mock_deeplens",
        execution_target="remote",
    )
    issues = ctrl.validate_preconditions(spec)
    assert len(issues) > 0
    assert any("remote" in i.lower() for i in issues)
