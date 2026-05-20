from optiresearch.skills.loader import SkillLoader
from optiresearch.skills.registry import SkillRegistry


def test_skill_registry_scans_manifests_and_loads_skill_md():
    registry = SkillRegistry()
    registry.load_all()

    manifests = registry.list()
    deeplens = registry.get("deeplens-adapter")
    intent_matches = registry.find_by_intent("PSF optical simulation")
    skill_md = SkillLoader(registry).load_skill_md("deeplens-adapter")

    assert "deeplens-adapter" in {manifest.skill_id for manifest in manifests}
    assert deeplens.display_name == "DeepLens Adapter"
    assert "deeplens-adapter" in {manifest.skill_id for manifest in intent_matches}
    assert "MockDeepLensAdapter" in skill_md
