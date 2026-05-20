"""Rule-based memory router."""

from __future__ import annotations

from typing import Any, Optional

from optiresearch.memory.schemas import make_context_pack_id
from optiresearch.skills.registry import SkillRegistry
from optiresearch.storage.file_artifact_store import FileArtifactStore
from optiresearch.storage.sqlite_store import SQLiteStore


class MemoryRouter:
    """Route queries across memory projections without external LLM calls."""

    def __init__(
        self,
        store: Optional[SQLiteStore] = None,
        artifact_store: Optional[FileArtifactStore] = None,
        skill_registry: Optional[SkillRegistry] = None,
    ) -> None:
        self.store = store or SQLiteStore()
        self.store.init_db()
        self.artifact_store = artifact_store or FileArtifactStore(store=self.store)
        self.skill_registry = skill_registry or SkillRegistry()

    def query(
        self,
        role: str,
        intent: str,
        query: str,
        scope: Optional[dict[str, Any]],
        top_k: int = 8,
        require_evidence: bool = False,
    ) -> dict[str, Any]:
        intent_text = f"{intent} {query}".lower()
        items: list[dict[str, Any]] = []
        warnings: list[str] = []
        if any(token in intent_text for token in ("evidence", "claim", "证据")):
            items.extend(self._items("claim", "claims", query, top_k, require_evidence=require_evidence))
            items.extend(self._items("artifact", "artifacts", query, top_k))
            items.extend(self._items("trace", "meta_traces", query, top_k))
        elif any(token in intent_text for token in ("plan", "计划")):
            items.extend(self._items("plan_template", "plan_templates", query, top_k))
            items.extend(self._items("run_memory", "run_memories", query, top_k))
        elif any(token in intent_text for token in ("skill", "技能")):
            items.extend(self._items("skill_memory", "skill_memories", query, top_k))
            self.skill_registry.load_all()
            items.extend(
                self._context_item("skill_manifest", manifest.skill_id, manifest.model_dump(mode="json"), query)
                for manifest in self.skill_registry.find_by_intent(query)
            )
        else:
            items.extend(self._items("run_memory", "run_memories", query, top_k))
            items.extend(self._items("trace", "meta_traces", query, top_k))
        if require_evidence and not any(item["evidence_refs"] for item in items if item["type"] in {"claim", "artifact"}):
            warnings.append("No direct claim or artifact evidence was found.")
        if require_evidence:
            for item in items:
                if item["type"] == "claim" and not item["evidence_refs"]:
                    item["score"] = 0.0
                    warnings.append(f"Claim lacks artifact evidence: {item['id']}")
                if item["type"] == "design_rule" and not item["evidence_refs"]:
                    item["score"] = 0.0
                    warnings.append(f"Design rule lacks support: {item['id']}")
        items.sort(key=lambda item: item["score"], reverse=True)
        return {
            "context_pack_id": make_context_pack_id(role, intent, query, scope or {}),
            "items": items[:top_k],
            "warnings": list(dict.fromkeys(warnings)),
        }

    def _items(
        self,
        item_type: str,
        table: str,
        query: str,
        top_k: int,
        require_evidence: bool = False,
    ) -> list[dict[str, Any]]:
        matches = self.store.search_by_text(table, query, top_k=top_k)
        if not matches:
            matches = self.store.list(table)[-top_k:]
        items = [self._context_item(item_type, self._payload_id(item_type, payload), payload, query) for payload in matches]
        if require_evidence and item_type == "claim":
            return [item for item in items if item["evidence_refs"]]
        return items

    def _context_item(self, item_type: str, item_id: str, payload: dict[str, Any], query: str) -> dict[str, Any]:
        evidence_refs = self._evidence_refs(item_type, payload)
        return {
            "type": item_type,
            "id": item_id,
            "summary": self._summary(item_type, payload),
            "score": self._score(payload, query, evidence_refs),
            "evidence_refs": evidence_refs,
            "source": self._source(item_type),
            "payload": payload,
        }

    def _payload_id(self, item_type: str, payload: dict[str, Any]) -> str:
        keys = {
            "trace": "trace_id",
            "artifact": "artifact_id",
            "run_memory": "run_id",
            "claim": "claim_id",
            "plan_template": "template_id",
            "skill_memory": "skill_id",
            "design_rule": "rule_id",
        }
        key = keys.get(item_type, "id")
        return str(payload.get(key, "unknown"))

    def _summary(self, item_type: str, payload: dict[str, Any]) -> str:
        if item_type == "claim":
            return f"{payload.get('status')}: {payload.get('text')}"
        if item_type == "artifact":
            return f"{payload.get('metadata', {}).get('artifact_type', 'artifact')}: {payload.get('uri')}"
        if item_type == "trace":
            return f"{payload.get('actor')} {payload.get('status')}: {payload.get('task')}"
        if item_type == "run_memory":
            return f"{payload.get('current_status')}: {payload.get('objective')}"
        if item_type == "plan_template":
            return f"{payload.get('template_id')}: {payload.get('description')}"
        if item_type == "skill_memory":
            return f"{payload.get('skill_id')} success_rate={payload.get('success_rate')}"
        return str(payload)[:200]

    def _score(self, payload: dict[str, Any], query: str, evidence_refs: list[str]) -> float:
        text = str(payload).lower()
        score = sum(0.1 for token in query.lower().split() if token in text)
        score += 0.5 if evidence_refs else 0.0
        if isinstance(payload.get("support_score"), (int, float)):
            score += float(payload["support_score"])
        if isinstance(payload.get("historical_success_rate"), (int, float)):
            score += float(payload["historical_success_rate"])
        if isinstance(payload.get("success_rate"), (int, float)):
            score += float(payload["success_rate"])
        return round(score, 6)

    def _evidence_refs(self, item_type: str, payload: dict[str, Any]) -> list[str]:
        if item_type == "claim":
            refs = [edge.get("artifact_id") for edge in payload.get("support_edges", []) if edge.get("artifact_id")]
            refs.extend(edge.get("artifact_id") for edge in payload.get("contradict_edges", []) if edge.get("artifact_id"))
            return sorted(set(refs))
        if item_type == "artifact":
            return [payload["artifact_id"]] if payload.get("artifact_id") else []
        if item_type == "design_rule":
            return payload.get("supported_by", [])
        return payload.get("source_trace_ids", [])

    def _source(self, item_type: str) -> str:
        return {
            "trace": "meta_traces",
            "artifact": "artifacts",
            "run_memory": "run_memories",
            "claim": "claims",
            "plan_template": "plan_templates",
            "skill_memory": "skill_memories",
            "design_rule": "design_rules",
        }.get(item_type, "unknown")
