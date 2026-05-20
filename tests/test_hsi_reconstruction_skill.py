from optiresearch.skills.registry import SkillRegistry
from optiresearch.skills.router import SkillRouter


def test_hsi_reconstruction_skill_is_registered_and_routed():
    registry = SkillRegistry()
    registry.load_all()
    skill = registry.get("hsi-reconstruction")
    resolved = SkillRouter(registry=registry).resolve("SimulationExperimentalist", "run HSI reconstruction with SAM metric", intent="HSI reconstruction")

    assert skill.skill_id == "hsi-reconstruction"
    assert any(item.skill_id == "hsi-reconstruction" for item in resolved)
