"""Progressive skill metadata loader."""

from __future__ import annotations

from pathlib import Path

from optiresearch.memory.schemas import SkillManifest
from optiresearch.skills.registry import SkillRegistry


class SkillLoader:
    """Load L0/L1 skill resources for the MVP."""

    def __init__(self, registry: SkillRegistry | None = None) -> None:
        self.registry = registry or SkillRegistry()

    def load_metadata(self, skill_id: str) -> SkillManifest:
        return self.registry.get(skill_id)

    def load_skill_md(self, skill_id: str) -> str:
        path = self.registry.skill_path(skill_id) / "SKILL.md"
        return path.read_text(encoding="utf-8")

    def load_resources(self, skill_id: str) -> dict[str, list[str]]:
        base = self.registry.skill_path(skill_id)
        resources: dict[str, list[str]] = {}
        for folder in ("config_templates", "references", "assets", "scripts"):
            path = base / folder
            if path.exists():
                resources[folder] = [item.name for item in sorted(path.iterdir())]
        return resources

    def skill_path(self, skill_id: str) -> Path:
        return self.registry.skill_path(skill_id)
