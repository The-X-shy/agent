"""Deterministic mock LLM provider for tests and fallback demos."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from optiresearch.llm.base import LLMProvider, LLMResponse
from optiresearch.llm.structured_output import ClaimReviewDraft, ExperimentSpecDraft, PaperSectionDraft, ResearchPlanDraft
from optiresearch.schemas.autonomous import ResearchIterationPlan, ReviewerOutput


class MockLLMProvider(LLMProvider):
    provider_name = "mock"
    model = "mock-llm"

    def available(self) -> bool:
        return True

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        objective = messages[-1]["content"] if messages else "OptiResearch"
        return LLMResponse(
            content=f"Mock response for: {objective}",
            provider=self.provider_name,
            model=self.model,
            finish_reason="stop",
            usage={"total_tokens": len(objective.split())},
            latency_ms=0.0,
            raw={"mock": True},
        )

    def structured_complete(self, messages: list[dict[str, str]], schema: type[BaseModel], **kwargs: Any) -> BaseModel:
        objective = messages[-1]["content"] if messages else "OptiResearch"
        if schema is ResearchPlanDraft:
            return schema.model_validate(
                {
                    "objective": objective,
                    "hypotheses": ["depth-stable PSF can improve evidence quality"],
                    "candidate_experiments": ["run first optical baseline"],
                    "required_skills": ["deeplens-adapter", "evidence-review"],
                    "risks": ["simulation-only evidence"],
                    "next_action": "run first simulation",
                }
            )
        if schema is ResearchIterationPlan:
            return schema.model_validate(
                {
                    "iteration_id": kwargs.get("iteration_id", 1),
                    "hypothesis": "Mock hypothesis: controlled_chromatic_edof will outperform conventional under synthetic HSI.",
                    "selected_encoder": kwargs.get("selected_encoder", "controlled_chromatic_edof"),
                    "selected_reconstructor": kwargs.get("selected_reconstructor", "optical_conditioned_linear"),
                    "selected_forward_mode": "depth_spectral_coded",
                    "selected_backend": kwargs.get("backend", "mock_deeplens"),
                    "expected_improvement": "Higher reconstruction_score due to encoder PSF coding.",
                    "required_skills": ["hsi_reconstruction"],
                    "risk_notes": "Mock provider - no LLM reasoning. Synthetic only.",
                    "evidence_requirements": ["synthetic_hsi_metrics"],
                }
            )
        if schema is ReviewerOutput:
            return schema.model_validate(
                {
                    "iteration_assessment": "Mock assessment: iteration completed successfully.",
                    "improvement_detected": True,
                    "improvement_detail": "Mock reviewer detected improvement.",
                    "evidence_level": "mock",
                    "caveats": ["Mock reviewer - no LLM reasoning.", "Mock backend - not real optical validation."],
                    "supported_claim": "Mock claim: controlled_chromatic_edof shows improvement.",
                    "unsupported_claim": "",
                    "next_action": "continue",
                    "next_encoder": "",
                    "next_reconstructor": "",
                    "next_forward_mode": "",
                    "stopping_reason": "",
                    "recommendation_for_human": "Mock: Continue exploring parameter space.",
                }
            )
        if schema is ExperimentSpecDraft:
            return schema.model_validate({"objective": objective, "backend": kwargs.get("backend", "mock_deeplens"), "caveats": ["mock provider draft"]})
        if schema is ClaimReviewDraft:
            return schema.model_validate(
                {
                    "claim_text": objective,
                    "suggested_status": "partially_supported",
                    "reasoning": "Mock reviewer requires artifact support.",
                    "required_caveats": ["LLM review is advisory"],
                    "missing_evidence": [],
                    "follow_up_experiments": [],
                    "risk_level": "medium",
                }
            )
        if schema is PaperSectionDraft:
            return schema.model_validate(
                {
                    "section_title": "Experiment Summary",
                    "paragraphs": ["The section summarizes registered artifacts and reviewed claims."],
                    "cited_artifact_ids": kwargs.get("artifact_ids", []),
                    "cited_claim_ids": kwargs.get("claim_ids", []),
                    "limitations": ["LLM text is not evidence."],
                    "unsupported_claim_warnings": [],
                }
            )
        return schema.model_validate(json.loads("{}"))
