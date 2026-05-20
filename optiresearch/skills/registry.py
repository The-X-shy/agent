"""Skill manifest registry."""

from __future__ import annotations

from pathlib import Path

import yaml

from optiresearch.memory.schemas import SkillManifest


class SkillRegistry:
    """Scan skill folders and expose manifest lookup helpers."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or Path(__file__).parent)
        self._manifests: dict[str, SkillManifest] = {}
        self._paths: dict[str, Path] = {}

    def load_all(self) -> list[SkillManifest]:
        self._manifests.clear()
        self._paths.clear()
        for manifest_path in sorted(self.root.glob("*/manifest.yaml")):
            payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            manifest = SkillManifest(**payload)
            self._manifests[manifest.skill_id] = manifest
            self._paths[manifest.skill_id] = manifest_path.parent
        return self.list()

    def get(self, skill_id: str) -> SkillManifest:
        if not self._manifests:
            self.load_all()
        if skill_id not in self._manifests:
            raise KeyError(f"Unknown skill_id={skill_id}")
        return self._manifests[skill_id]

    def skill_path(self, skill_id: str) -> Path:
        if not self._paths:
            self.load_all()
        if skill_id not in self._paths:
            raise KeyError(f"Unknown skill_id={skill_id}")
        return self._paths[skill_id]

    def list(self) -> list[SkillManifest]:
        if not self._manifests:
            for manifest_path in sorted(self.root.glob("*/manifest.yaml")):
                payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
                manifest = SkillManifest(**payload)
                self._manifests[manifest.skill_id] = manifest
                self._paths[manifest.skill_id] = manifest_path.parent
        return list(self._manifests.values())

    def find_by_intent(self, intent: str) -> list[SkillManifest]:
        needle = intent.lower()
        matches: list[SkillManifest] = []
        for manifest in self.list():
            haystack = " ".join(
                [manifest.skill_id, manifest.display_name, manifest.description, *manifest.intents]
            ).lower()
            if any(token in haystack for token in needle.split()) or needle in haystack:
                matches.append(manifest)
        return matches

    def find_by_role(self, role: str) -> list[SkillManifest]:
        role_lower = role.lower()
        return [manifest for manifest in self.list() if any(item.lower() == role_lower for item in manifest.roles)]
