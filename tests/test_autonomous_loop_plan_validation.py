"""Test autonomous loop plan validation."""
from optiresearch.schemas.autonomous import AutonomousLoopConfig, ResearchIterationPlan
from optiresearch.runtime.autonomous_loop import _validate_plan


def test_validate_plan_allows_valid_plan():
    config = AutonomousLoopConfig(
        objective="Test",
        backend="mock_deeplens",
        allowed_encoders=["conventional", "controlled_chromatic_edof"],
        allowed_reconstructors=["optical_conditioned_linear"],
        allowed_forward_modes=["depth_spectral_coded"],
    )
    plan = ResearchIterationPlan(
        iteration_id=1,
        hypothesis="Test encoder comparison",
        selected_encoder="controlled_chromatic_edof",
        selected_reconstructor="optical_conditioned_linear",
        selected_forward_mode="depth_spectral_coded",
        selected_backend="mock_deeplens",
        expected_improvement="Better reconstruction score",
    )
    error = _validate_plan(plan, config)
    assert error == ""


def test_validate_plan_rejects_disallowed_encoder():
    config = AutonomousLoopConfig(
        objective="Test",
        allowed_encoders=["conventional"],
        allowed_reconstructors=["optical_conditioned_linear"],
        allowed_forward_modes=["depth_spectral_coded"],
    )
    plan = ResearchIterationPlan(
        iteration_id=1,
        hypothesis="Test",
        selected_encoder="controlled_chromatic_edof",
        selected_reconstructor="optical_conditioned_linear",
        selected_forward_mode="depth_spectral_coded",
        selected_backend="mock_deeplens",
        expected_improvement="Test",
    )
    error = _validate_plan(plan, config)
    assert "not in allowed list" in error


def test_validate_plan_rejects_disallowed_reconstructor():
    config = AutonomousLoopConfig(
        objective="Test",
        allowed_encoders=["conventional"],
        allowed_reconstructors=["linear_baseline"],
        allowed_forward_modes=["depth_spectral_coded"],
    )
    plan = ResearchIterationPlan(
        iteration_id=1,
        hypothesis="Test",
        selected_encoder="conventional",
        selected_reconstructor="optical_conditioned_linear",
        selected_forward_mode="depth_spectral_coded",
        selected_backend="mock_deeplens",
        expected_improvement="Test",
    )
    error = _validate_plan(plan, config)
    assert "not in allowed list" in error


def test_validate_plan_rejects_native_claim_on_mock():
    config = AutonomousLoopConfig(
        objective="Test",
        backend="mock_deeplens",
        allowed_encoders=["conventional"],
        allowed_reconstructors=["optical_conditioned_linear"],
        allowed_forward_modes=["depth_spectral_coded"],
    )
    plan = ResearchIterationPlan(
        iteration_id=1,
        hypothesis="This encoder achieves native physical validation",
        selected_encoder="conventional",
        selected_reconstructor="optical_conditioned_linear",
        selected_forward_mode="depth_spectral_coded",
        selected_backend="mock_deeplens",
        expected_improvement="Test",
    )
    error = _validate_plan(plan, config)
    assert "native" in error.lower()


def test_validate_plan_rejects_real_camera_on_mock():
    config = AutonomousLoopConfig(
        objective="Test",
        backend="mock_deeplens",
        allowed_encoders=["conventional"],
        allowed_reconstructors=["optical_conditioned_linear"],
        allowed_forward_modes=["depth_spectral_coded"],
    )
    plan = ResearchIterationPlan(
        iteration_id=1,
        hypothesis="Real camera HSI shows improvement",
        selected_encoder="conventional",
        selected_reconstructor="optical_conditioned_linear",
        selected_forward_mode="depth_spectral_coded",
        selected_backend="mock_deeplens",
        expected_improvement="Test",
    )
    error = _validate_plan(plan, config)
    assert "real camera" in error.lower()
