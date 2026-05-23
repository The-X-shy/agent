"""Skill Runtime v2 for Phase 36."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from optiresearch.agent_system.event_bus import get_event_bus
from optiresearch.agent_system.events import AgentEvent
from optiresearch.skills.contracts import SkillResult
from optiresearch.skills.registry_v2 import SkillRegistryV2


class SkillRuntimeV2:
    def __init__(self, registry: SkillRegistryV2 | None = None):
        self._registry = registry or SkillRegistryV2()
        self._event_bus = get_event_bus()

    @property
    def registry(self) -> SkillRegistryV2:
        return self._registry

    def validate_input(self, skill_id: str, inputs: dict[str, Any]) -> list[str]:
        spec = self._registry.get(skill_id)
        if spec is None:
            return [f"Unknown skill: {skill_id}"]
        errors: list[str] = []
        for key, schema in spec.input_schema.items():
            if key not in inputs and schema.get("required"):
                errors.append(f"Missing required input: {key}")
        return errors

    def execute_skill(self, skill_id: str, inputs: dict[str, Any] | None = None) -> SkillResult:
        inputs = inputs or {}
        spec = self._registry.get(skill_id)
        if spec is None:
            return SkillResult(skill_id=skill_id, status="failed",
                               errors=[f"Unknown skill: {skill_id}"])

        validation_errors = self.validate_input(skill_id, inputs)
        if validation_errors:
            return SkillResult(skill_id=skill_id, status="failed", errors=validation_errors)

        t0 = time.time()
        try:
            output = self._dispatch(skill_id, inputs)
            elapsed = time.time() - t0
            result = SkillResult(
                skill_id=skill_id, status="succeeded",
                inputs_hash=_hash_inputs(inputs),
                output=output, execution_time_sec=elapsed,
            )
            self._event_bus.publish(AgentEvent.create(
                "skill_called", "skill_runtime",
                payload={"skill_id": skill_id, "status": "succeeded", "execution_time_sec": elapsed},
            ))
            return result
        except Exception as exc:
            elapsed = time.time() - t0
            result = SkillResult(
                skill_id=skill_id, status="failed",
                inputs_hash=_hash_inputs(inputs),
                errors=[str(exc)], execution_time_sec=elapsed,
            )
            self._event_bus.publish(AgentEvent.create(
                "skill_failed", "skill_runtime",
                payload={"skill_id": skill_id, "error": str(exc)},
                severity="error",
            ))
            return result

    def audit_skill_result(self, result: SkillResult) -> list[str]:
        issues: list[str] = []
        if result.status == "failed":
            issues.append(f"Skill {result.skill_id} failed: {result.errors}")
        if result.execution_time_sec > 3600:
            issues.append(f"Skill {result.skill_id} exceeded 1h timeout")
        return issues

    def _dispatch(self, skill_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
        if skill_id == "claim_check":
            from optiresearch.memory.claim_gate_v2 import ClaimGateV2
            gate = ClaimGateV2()
            decision = gate.check_claim(
                claim_text=inputs.get("claim", ""),
                backend_id=inputs.get("backend_id", "deeplens_geolens_geometric"),
            )
            return {"decision": decision.decision, "max_allowed_claim": decision.max_allowed_claim,
                    "violation_type": decision.violation_type, "safe_wording": decision.safe_wording}
        if skill_id == "strategy_recommendation":
            from optiresearch.agents.strategy_engine import StrategyEngine
            engine = StrategyEngine()
            rec = engine.recommend(
                latest_result=inputs.get("latest_result", {}),
                backend_id=inputs.get("backend_id", "deeplens_geolens_geometric"),
            )
            return {"action": rec.recommended_action, "rationale": rec.rationale,
                    "risk_level": rec.risk_level}
        raise NotImplementedError(f"No runtime dispatch for skill: {skill_id}")


def _hash_inputs(inputs: dict[str, Any]) -> str:
    raw = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
