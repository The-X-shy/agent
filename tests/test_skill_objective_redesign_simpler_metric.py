"""Test SkillRuntimeV2 dispatch for lightweight_scientific_hsi_mse_only skill."""

from optiresearch.skills.runtime_v2 import SkillRuntimeV2
from optiresearch.skills.registry_v2 import SkillRegistryV2


def test_skill_is_registered():
    registry = SkillRegistryV2()
    spec = registry.get("lightweight_scientific_hsi_mse_only")
    assert spec is not None
    assert spec.skill_id == "lightweight_scientific_hsi_mse_only"
    assert spec.required_backends == []
    assert spec.evidence_level == "lightweight_scientific_execution"
    assert spec.risk_level == "low"
    assert "lightweight_scientific_execution" in spec.claim_implications
    assert "result.json" in spec.produced_artifacts


def test_skill_executes_via_runtime():
    runtime = SkillRuntimeV2()
    result = runtime.execute_skill(
        "lightweight_scientific_hsi_mse_only",
        {"max_steps": 3, "bands": 4},
    )
    assert result.status == "succeeded"
    assert result.skill_id == "lightweight_scientific_hsi_mse_only"
    output = result.output
    assert output["status"] == "succeeded"
    assert output["evidence_level"] == "lightweight_scientific_execution"
    assert "reconstruction_loss_before" in output
    assert "reconstruction_loss_after" in output
    assert "mse_before" in output
    assert "mse_after" in output
    assert output["synthetic_data"] is True
    assert output["physical_backend"] is False


def test_skill_returns_improvement_field():
    runtime = SkillRuntimeV2()
    result = runtime.execute_skill(
        "lightweight_scientific_hsi_mse_only",
        {"max_steps": 5},
    )
    assert result.status == "succeeded"
    assert isinstance(result.output.get("improvement_detected"), bool)


def test_skill_execution_is_fast():
    import time
    runtime = SkillRuntimeV2()
    start = time.perf_counter()
    result = runtime.execute_skill(
        "lightweight_scientific_hsi_mse_only",
        {"max_steps": 5},
    )
    elapsed = time.perf_counter() - start
    assert result.status == "succeeded"
    assert elapsed < 30.0, f"Skill took {elapsed:.1f}s, expected <30s"


def test_skill_validates_inputs():
    runtime = SkillRuntimeV2()
    errors = runtime.validate_input("lightweight_scientific_hsi_mse_only", {})
    # No required inputs, so validation should pass
    assert errors == []
