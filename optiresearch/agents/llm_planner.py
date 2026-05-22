"""LLM-Assisted Autonomous Research Planner.

Generates candidate research proposals using an LLM provider,
validates them through safety gates, and returns the best proposal
with rule-based fallback on failure.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from optiresearch.memory.schemas import make_deterministic_id


class LLMPlanner:
    """LLM-assisted planner for differentiable optics research.

    Generates candidate proposals via LLM, validates each through
    PlannerValidator, ranks by risk/evidence match, and selects the
    best proposal. Falls back to StrategyEngine on any failure.
    """

    def plan(
        self,
        objective: str,
        provider_name: str = "mock",
        allowed_backends: Optional[list[str]] = None,
        allowed_task_types: Optional[list[str]] = None,
        recent_results: Optional[list[dict[str, Any]]] = None,
        execution_mode: str = "dry_run",
        allow_remote: bool = False,
        max_candidate_plans: int = 3,
        prefer_executable_actions: bool = False,
    ) -> "LLMPlannerResult":
        """Generate and validate research proposals.

        Args:
            objective: Research objective.
            provider_name: LLM provider name (mock, deepseek).
            allowed_backends: Allowed backend IDs.
            allowed_task_types: Allowed task types.
            recent_results: Recent experiment results for context.
            execution_mode: dry_run, local, remote_opt_in.
            allow_remote: Whether remote execution is allowed.
            max_candidate_plans: Max proposals to generate.
            prefer_executable_actions: Prefer executable over stop_and_report.

        Returns:
            LLMPlannerResult with selected proposal or fallback.
        """
        from optiresearch.schemas.llm_planner import (
            LLMPlannerResult,
            LLMPlannerProposal,
            LLMPlannerContext,
        )

        backends = allowed_backends or [
            "phase_to_fft_proxy",
            "deeplens_geolens_geometric",
            "local_synthetic_hsi",
        ]
        tasks = allowed_task_types or [
            "stable_lens_hsi_codesign",
            "native_hsi_codesign",
            "native_hsi_reconstruction_codesign",
        ]

        run_id = make_deterministic_id("llmplan", objective, str(time.time()))

        # Build context
        context = self.build_context(
            objective=objective,
            allowed_backends=backends,
            allowed_task_types=tasks,
            recent_results=recent_results or [],
            execution_mode=execution_mode,
            max_candidate_plans=max_candidate_plans,
            prefer_executable_actions=prefer_executable_actions,
        )

        # Try LLM provider
        try:
            provider = self._get_provider(provider_name)
            if not provider.available():
                return self._fallback_result(
                    run_id, "provider_not_available", provider_name
                )

            proposals_raw = self._call_provider(provider, context, max_candidate_plans)

            # Parse and validate proposals
            proposals = self._parse_proposals(proposals_raw)
            validated = self._validate_all(
                proposals, backends, tasks, execution_mode, allow_remote
            )

            valid_proposals = [p for p, v in validated if v["valid"]]
            validation_errors = [
                {"proposal_id": p.proposal_id, "errors": v["errors"]}
                for p, v in validated
                if not v["valid"]
            ]

            if not valid_proposals:
                return self._fallback_result(
                    run_id, "no_valid_proposals", provider_name,
                    validation_errors=validation_errors,
                )

            # Rank and select
            ranked = self.rank_proposals(valid_proposals)
            selected = ranked[0]
            rejected = ranked[1:] if len(ranked) > 1 else []

            # Apply claim gate to selected proposal
            selected = self._apply_claim_gate(selected)

            # If prefer_executable and selected is stop_and_report,
            # scan rejected proposals for an executable alternative
            if (prefer_executable_actions
                    and selected.recommended_action == "stop_and_report"
                    and rejected):
                executable_alt = next(
                    (p for p in rejected
                     if p.recommended_action != "stop_and_report"
                     and p.recommended_action != "downgrade_claim"),
                    None,
                )
                if executable_alt is not None:
                    executable_alt = self._apply_claim_gate(executable_alt)
                    rejected = [
                        p for p in rejected
                        if p.proposal_id != executable_alt.proposal_id
                    ]
                    rejected.append(selected)
                    selected = executable_alt

            # Record trace
            trace_path = self._record_trace(
                run_id, context, proposals_raw, validated, selected, None
            )

            return LLMPlannerResult(
                status="succeeded",
                provider=provider_name,
                model=getattr(provider, "model", "unknown"),
                planner_run_id=run_id,
                proposals=ranked,
                selected_proposal=selected,
                rejected_proposals=rejected,
                validation_errors=validation_errors,
                planner_trace_path=trace_path,
            )

        except Exception as exc:
            return self._fallback_result(
                run_id, f"llm_error: {exc}", provider_name,
                error=str(exc),
            )

    def build_context(
        self,
        objective: str,
        allowed_backends: list[str],
        allowed_task_types: list[str],
        recent_results: list[dict[str, Any]],
        execution_mode: str = "dry_run",
        max_candidate_plans: int = 3,
        prefer_executable_actions: bool = False,
    ) -> dict[str, Any]:
        """Build planning context from Phase 24/25 components."""
        context: dict[str, Any] = {
            "objective": objective,
            "allowed_backends": allowed_backends,
            "allowed_task_types": allowed_task_types,
            "recent_results": recent_results,
            "execution_mode": execution_mode,
            "max_candidate_plans": max_candidate_plans,
            "prefer_executable_actions": prefer_executable_actions,
        }

        # Add backend registry summary
        try:
            from optiresearch.backends.registry import list_backends
            context["backend_registry_summary"] = [
                {
                    "backend_id": b.backend_id,
                    "claim_ceiling": b.claim_ceiling,
                    "differentiability_level": b.differentiability_level,
                    "known_failure_modes": b.known_failure_modes[:3],
                }
                for b in list_backends()
                if b.backend_id in allowed_backends
            ]
        except Exception:
            context["backend_registry_summary"] = []

        # Add research memory rules
        try:
            from optiresearch.memory.research_memory_v2 import ResearchMemoryV2
            mem = ResearchMemoryV2()
            rules = mem.query(min_confidence=0.8)
            context["research_memory"] = [
                {
                    "memory_type": r.memory_type,
                    "content": r.content,
                    "tags": r.tags,
                }
                for r in rules
            ]
        except Exception:
            context["research_memory"] = []

        return context

    def rank_proposals(
        self, proposals: list["LLMPlannerProposal"]
    ) -> list["LLMPlannerProposal"]:
        """Rank proposals by risk level (low > medium > high)."""
        risk_order = {"low": 0, "medium": 1, "high": 2}

        def sort_key(p):
            return risk_order.get(p.risk_level, 99)

        return sorted(proposals, key=sort_key)

    def _get_provider(self, provider_name: str):
        from optiresearch.llm.registry import get_llm_provider
        return get_llm_provider(provider_name)

    def _call_provider(
        self, provider, context: dict[str, Any], max_plans: int
    ) -> list[dict[str, Any]]:
        """Call the LLM provider and extract proposals.

        For mock provider: returns deterministic mock proposals.
        For real providers: calls structured_complete with LLMPlannerProposal schema.
        """
        if getattr(provider, "provider_name", "") == "mock":
            from optiresearch.agents.prompts.llm_planner_prompt import (
                build_mock_proposals,
            )
            return build_mock_proposals()[:max_plans]

        # Build prompt for real providers
        from optiresearch.agents.prompts.llm_planner_prompt import (
            build_planner_prompt,
        )

        messages = build_planner_prompt(context)
        response = provider.complete(messages)
        content = response.content if hasattr(response, "content") else str(response)

        # Parse JSON from response
        import json as _json
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])
        try:
            data = _json.loads(content)
        except _json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                data = _json.loads(content[start:end])
            else:
                raise ValueError(f"Could not parse JSON from LLM response: {content[:200]}")

        return data.get("proposals", [data])[:max_plans]

    def _parse_proposals(
        self, raw: list[dict[str, Any]]
    ) -> list["LLMPlannerProposal"]:
        from optiresearch.schemas.llm_planner import LLMPlannerProposal

        proposals: list[LLMPlannerProposal] = []
        for i, p in enumerate(raw):
            try:
                if not p.get("proposal_id"):
                    p["proposal_id"] = f"proposal_{i:03d}"
                proposals.append(LLMPlannerProposal(**p))
            except Exception:
                continue
        return proposals

    def _validate_all(
        self,
        proposals: list["LLMPlannerProposal"],
        allowed_backends: list[str],
        allowed_task_types: list[str],
        execution_mode: str,
        allow_remote: bool,
    ) -> list[tuple["LLMPlannerProposal", dict[str, Any]]]:
        from optiresearch.agents.planner_validator import validate_proposal

        results: list[tuple["LLMPlannerProposal", dict[str, Any]]] = []
        for p in proposals:
            v = validate_proposal(
                p, allowed_backends, allowed_task_types, execution_mode, allow_remote
            )
            results.append((p, v))
        return results

    def _apply_claim_gate(
        self, proposal: "LLMPlannerProposal"
    ) -> "LLMPlannerProposal":
        try:
            from optiresearch.memory.claim_gate_v2 import ClaimGateV2
            gate = ClaimGateV2()
            decision = gate.check_claim(
                claim_text=proposal.proposed_claim,
                backend_id=proposal.backend_id,
            )
            if decision.decision == "unsupported" and decision.safe_wording:
                proposal.safe_wording = decision.safe_wording
        except Exception:
            pass
        return proposal

    def _fallback_result(
        self,
        run_id: str,
        reason: str,
        provider_name: str,
        validation_errors: Optional[list[dict[str, Any]]] = None,
        error: Optional[str] = None,
    ) -> "LLMPlannerResult":
        from optiresearch.schemas.llm_planner import LLMPlannerResult
        from optiresearch.agents.strategy_engine import StrategyEngine

        # Get fallback from StrategyEngine
        engine = StrategyEngine()
        fallback_rec = engine.recommend({}, "deeplens_geolens_geometric")

        return LLMPlannerResult(
            status="fallback_used",
            provider=provider_name,
            planner_run_id=run_id,
            validation_errors=validation_errors or [],
            fallback_strategy={
                "reason": reason,
                "recommended_action": fallback_rec.recommended_action,
                "rationale": fallback_rec.rationale,
                "risk_level": fallback_rec.risk_level,
            },
            error=error,
        )

    def _record_trace(
        self,
        run_id: str,
        context: dict[str, Any],
        raw_response: list[dict[str, Any]],
        validated: list[tuple["LLMPlannerProposal", dict[str, Any]]],
        selected: Optional["LLMPlannerProposal"],
        fallback: Optional[dict[str, Any]],
    ) -> Optional[str]:
        try:
            from optiresearch.agents.planner_trace import (
                start_planner_trace,
                record_context,
                record_response,
                record_validation,
                record_selection,
                finalize_trace,
            )
            trace = start_planner_trace(run_id)
            record_context(trace, context)
            record_response(trace, raw_response)
            record_validation(trace, validated)
            if selected:
                record_selection(trace, selected.model_dump(mode="json"))
            if fallback:
                record_selection(trace, {"fallback": fallback})
            return finalize_trace(trace)
        except Exception:
            return None
