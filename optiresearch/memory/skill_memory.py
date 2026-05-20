"""Skill memory manager."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from optiresearch.memory.schemas import MetaTrace, SkillMemory
from optiresearch.skills.registry import SkillRegistry
from optiresearch.storage.sqlite_store import SQLiteStore


class SkillMemoryManager:
    """Compile skill usage statistics from Meta-Trace entries."""

    def __init__(self, store: Optional[SQLiteStore] = None, registry: Optional[SkillRegistry] = None) -> None:
        self.store = store or SQLiteStore()
        self.store.init_db()
        self.registry = registry or SkillRegistry()

    def update_from_trace(self, trace: MetaTrace) -> Optional[SkillMemory]:
        if not trace.skill_id:
            return None
        version = trace.skill_version or "unknown"
        memory = self.get_skill_memory(trace.skill_id, version) or self._new_memory(trace.skill_id, version)
        if trace.run_id not in memory.used_in:
            memory.used_in.append(trace.run_id)
        if trace.status == "succeeded":
            memory.success_count += 1
        elif trace.status == "failed":
            memory.failure_count += 1
        command = trace.metadata.get("command") or trace.tool
        if command and command not in memory.commands:
            memory.commands.append(str(command))
        for artifact_type in trace.metadata.get("artifact_types", []):
            if artifact_type not in memory.emitted_artifact_types:
                memory.emitted_artifact_types.append(artifact_type)
        for limitation in trace.limitations:
            if limitation not in memory.common_failures:
                memory.common_failures.append(limitation)
        total = memory.success_count + memory.failure_count
        memory.success_rate = round(memory.success_count / total, 6) if total else 0.0
        memory.last_updated = datetime.now(timezone.utc)
        self.save(memory)
        return memory

    def update_from_run(self, run_id: str) -> list[SkillMemory]:
        memories: list[SkillMemory] = []
        for payload in self.store.list("meta_traces", run_id=run_id):
            memory = self.update_from_trace(MetaTrace(**payload))
            if memory:
                memories.append(memory)
        return memories

    def get_skill_memory(self, skill_id: str, version: Optional[str] = None) -> Optional[SkillMemory]:
        if version:
            payload = self.store.get("skill_memories", f"{skill_id}:{version}")
            return SkillMemory(**payload) if payload else None
        candidates = [
            SkillMemory(**payload)
            for payload in self.store.list("skill_memories")
            if payload.get("skill_id") == skill_id
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: item.last_updated or datetime.min, reverse=True)[0]

    def list_skill_memories(self) -> list[SkillMemory]:
        return [SkillMemory(**payload) for payload in self.store.list("skill_memories")]

    def recommend_skills(self, intent: str, role: Optional[str] = None) -> list[SkillMemory]:
        memories = self.list_skill_memories()
        if not memories:
            self.registry.load_all()
            for manifest in self.registry.find_by_intent(intent):
                memories.append(self._new_memory(manifest.skill_id, manifest.version))
        intent_lower = intent.lower()
        role_lower = (role or "").lower()

        def score(memory: SkillMemory) -> float:
            manifest_bonus = 0.0
            try:
                manifest = self.registry.get(memory.skill_id)
                haystack = " ".join([manifest.description, *manifest.intents, *manifest.roles]).lower()
                manifest_bonus += sum(0.3 for token in intent_lower.split() if token in haystack)
                if role_lower and any(item.lower() == role_lower for item in manifest.roles):
                    manifest_bonus += 0.5
            except KeyError:
                pass
            preferred_bonus = sum(0.4 for cue in memory.preferred_when if cue.lower() in intent_lower)
            return memory.success_rate + manifest_bonus + preferred_bonus

        return sorted(memories, key=score, reverse=True)

    def save(self, memory: SkillMemory) -> SkillMemory:
        self.store.upsert("skill_memories", f"{memory.skill_id}:{memory.version}", memory)
        return memory

    def _new_memory(self, skill_id: str, version: str) -> SkillMemory:
        preferred = ["simulate", "psf", "mtf"] if skill_id == "deeplens-adapter" else []
        best_practices = ["register artifacts and write Meta-Trace"] if skill_id == "deeplens-adapter" else []
        return SkillMemory(
            skill_id=skill_id,
            version=version,
            used_in=[],
            success_rate=0.0,
            preferred_when=preferred,
            common_failures=[],
            best_practices=best_practices,
            last_updated=None,
            success_count=0,
            failure_count=0,
            commands=[],
            emitted_artifact_types=[],
        )


class SkillMemoryStore(SkillMemoryManager):
    """Backward-compatible alias for MVP code."""
