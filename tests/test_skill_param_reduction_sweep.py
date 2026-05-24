"""Test skill dispatch for param_reduction_sweep."""

from optiresearch.skills.runtime_v2 import SkillRuntimeV2
from optiresearch.skills.registry_v2 import SkillRegistryV2


def test_param_reduction_skill_is_registered():
    registry = SkillRegistryV2()
    spec = registry.get("param_reduction_sweep")
    assert spec is not None
    assert spec.skill_id == "param_reduction_sweep"
    assert spec.required_backends == []
    assert spec.evidence_level == "lightweight_scientific_execution"
    assert spec.risk_level == "low"


def test_param_reduction_skill_executes_via_runtime():
    runtime = SkillRuntimeV2()
    result = runtime.execute_skill("param_reduction_sweep", {"max_steps": 2})
    assert result.status == "succeeded"
    output = result.output
    assert output["status"] == "succeeded"
    assert output["evidence_level"] == "lightweight_scientific_execution"
    assert output["configs_tested"] == 3
    assert output["best_k"] in (1, 2, 3)
    assert "reconstruction_loss_before" in output
    assert "reconstruction_loss_after" in output
