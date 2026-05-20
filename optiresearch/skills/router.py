"""Rule-based skill router."""

from __future__ import annotations

from optiresearch.memory.schemas import SkillManifest
from optiresearch.memory.skill_memory import SkillMemoryManager
from optiresearch.skills.registry import SkillRegistry


class SkillRouter:
    def __init__(
        self,
        registry: SkillRegistry | None = None,
        skill_memory_manager: SkillMemoryManager | None = None,
    ) -> None:
        self.registry = registry or SkillRegistry()
        self.skill_memory_manager = skill_memory_manager

    def resolve(
        self,
        role: str,
        task: str,
        intent: str | None = None,
        budget: str | None = None,
    ) -> list[SkillManifest]:
        text = f"{role} {task} {intent or ''} {budget or ''}".lower()
        skill_ids: list[str] = []
        if any(token in text for token in ("deeplens", "psf", "mtf", "光学仿真", "可微分")):
            skill_ids.append("deeplens-adapter")
        if any(token in text for token in ("hsi", "hyperspectral", "reconstruction", "sam", "ergas")):
            skill_ids.append("hsi-reconstruction")
        if any(token in text for token in ("claim", "evidence", "证据", "审查")):
            skill_ids.append("evidence-review")
        if any(token in text for token in ("paper", "writing", "论文", "报告")):
            skill_ids.append("paper-writing")
        if any(token in text for token in ("artifact", "文件", "图表", "指标")):
            skill_ids.append("artifact-inspector")
        if not skill_ids:
            skill_ids = [manifest.skill_id for manifest in self.registry.find_by_role(role)]
        manifests = [self.registry.get(skill_id) for skill_id in dict.fromkeys(skill_ids)]
        return self._rank_with_skill_memory(manifests, f"{task} {intent or ''}", role)

    def _rank_with_skill_memory(
        self,
        manifests: list[SkillManifest],
        intent: str,
        role: str,
    ) -> list[SkillManifest]:
        manager = self.skill_memory_manager or SkillMemoryManager()
        intent_lower = intent.lower()

        def score(manifest: SkillManifest) -> float:
            memory = manager.get_skill_memory(manifest.skill_id, manifest.version)
            value = 0.5
            if memory:
                value += memory.success_rate
                value -= 0.5 if memory.success_count == 0 and memory.failure_count > 0 else 0.0
                value += sum(0.25 for cue in memory.preferred_when if cue.lower() in intent_lower)
            if any(item == role for item in manifest.roles):
                value += 0.2
            return value

        return sorted(manifests, key=score, reverse=True)
