"""Pydantic schemas for structured LLM output."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from optiresearch.memory.schemas import StrictModel


class ResearchPlanDraft(StrictModel):
    objective: str = Field(min_length=1)
    hypotheses: list[str] = Field(default_factory=list)
    candidate_experiments: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    next_action: str = Field(min_length=1)


class ExperimentSpecDraft(StrictModel):
    objective: str = Field(min_length=1)
    encoder_type: str = Field(default="controlled_chromatic_edof")
    wavelength_range_nm: tuple[float, float] = (450.0, 700.0)
    wavelength_bands: int = 31
    depth_range_mm: tuple[float, float] = (-4.0, 4.0)
    depth_planes: int = 9
    primary_metric: str = "psf_depth_similarity"
    metric_thresholds: dict[str, float] = Field(default_factory=lambda: {"psf_depth_similarity": 0.8, "spectral_separability": 0.3})
    backend: str = "mock_deeplens"
    caveats: list[str] = Field(default_factory=list)


class ClaimReviewDraft(StrictModel):
    claim_text: str = Field(min_length=1)
    suggested_status: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)
    required_caveats: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    follow_up_experiments: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"]


class PaperSectionDraft(StrictModel):
    section_title: str = Field(min_length=1)
    paragraphs: list[str] = Field(default_factory=list)
    cited_artifact_ids: list[str] = Field(default_factory=list)
    cited_claim_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    unsupported_claim_warnings: list[str] = Field(default_factory=list)
