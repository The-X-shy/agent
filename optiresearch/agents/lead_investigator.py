"""Rule-based Lead Investigator."""

from __future__ import annotations

from typing import Any

from optiresearch.memory.schemas import make_run_id
from optiresearch.llm.audit import build_llm_trace_metadata
from optiresearch.llm.base import LLMProviderError, LLMResponse
from optiresearch.llm.registry import get_llm_provider
from optiresearch.llm.structured_output import ResearchPlanDraft
from optiresearch.skills.router import SkillRouter


class LeadInvestigator:
    """Create a first-pass research plan from a natural language objective."""

    def __init__(self, skill_router: SkillRouter | None = None) -> None:
        self.skill_router = skill_router or SkillRouter()

    def plan_with_llm(self, objective: str, memory_context: dict[str, Any] | None = None, provider: Any = None) -> ResearchPlanDraft:
        provider = provider or get_llm_provider()
        return provider.structured_complete(
            [
                {"role": "system", "content": "You are LeadInvestigator for OptiResearch. Return JSON only."},
                {"role": "user", "content": objective},
            ],
            ResearchPlanDraft,
        )

    def plan(
        self,
        objective: str,
        workspace_id: str = "default",
        backend: str = "mock_deeplens",
        use_llm: bool = False,
        provider: Any = None,
    ) -> dict[str, Any]:
        llm_metadata = {"llm_used": False, "fallback_used": False}
        draft: ResearchPlanDraft | None = None
        if use_llm:
            provider = provider or get_llm_provider()
            if getattr(provider, "available", lambda: False)():
                try:
                    draft = self.plan_with_llm(objective, {}, provider=provider)
                    response = LLMResponse(content=draft.model_dump_json(), provider=provider.provider_name, model=getattr(provider, "model", "unknown"))
                    llm_metadata = build_llm_trace_metadata(objective, response, "ResearchPlanDraft", fallback_used=False)
                except Exception as exc:
                    code = exc.error_code if isinstance(exc, LLMProviderError) else exc.__class__.__name__
                    llm_metadata = build_llm_trace_metadata(objective, None, "ResearchPlanDraft", fallback_used=True, error_code=code)
            else:
                llm_metadata = build_llm_trace_metadata(objective, None, "ResearchPlanDraft", fallback_used=True, error_code="LLM_PROVIDER_UNAVAILABLE")
        skills = self.skill_router.resolve("LeadInvestigator", objective, intent="plan deeplens evidence")
        run_id = make_run_id(workspace_id, objective)
        backend_label = "real DeepLens" if backend == "deeplens" else "mock DeepLens"
        steps = [
            "parse objective",
            f"build {backend_label} optical spec",
            f"run {backend_label} PSF simulation",
            "register artifacts",
            "compile run memory",
            "review claims against artifacts",
        ]
        if draft and draft.candidate_experiments:
            steps = [*steps, *draft.candidate_experiments[:2]]
        return {
            "run_id": run_id,
            "workspace_id": workspace_id,
            "objective": objective,
            "candidate_skills": [skill.skill_id for skill in skills],
            "steps": steps,
            "first_run": {"backend": backend, "seed": 42},
            "llm_metadata": llm_metadata,
        }
