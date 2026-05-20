"""Autonomous research loop schemas.

Defines configuration, iteration plan, result, and summary models
for the LLM-driven autonomous research loop.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from optiresearch.memory.schemas import StrictModel


class AutonomousLoopConfig(StrictModel):
    """Configuration for an autonomous research loop."""

    objective: str
    max_iterations: int = 3
    llm_provider: str = "mock"
    backend: str = "mock_deeplens"
    dataset: str = "synthetic"
    allowed_encoders: list[str] = ["conventional", "achromatic", "edof", "chromatic_coded", "controlled_chromatic_edof"]
    allowed_reconstructors: list[str] = ["optical_conditioned_linear", "tiny_cnn"]
    allowed_forward_modes: list[str] = ["depth_spectral_coded"]
    budget: Optional[float] = None
    stopping_criteria: list[str] = ["max_iterations", "no_improvement"]
    evidence_policy: str = "conservative"
    metadata: dict[str, Any] = {}


class ResearchIterationPlan(StrictModel):
    """A single iteration plan proposed by the LLM (or rule fallback)."""

    iteration_id: int
    hypothesis: str
    selected_encoder: str
    selected_reconstructor: str
    selected_forward_mode: str = "depth_spectral_coded"
    selected_backend: str = "mock_deeplens"
    expected_improvement: str = ""
    required_skills: list[str] = []
    risk_notes: str = ""
    evidence_requirements: list[str] = []


class ResearchIterationResult(StrictModel):
    """Result of executing a single iteration plan."""

    iteration_id: int
    run_id: str = ""
    status: Literal["succeeded", "failed", "validation_rejected"] = "failed"
    metrics: dict[str, Any] = {}
    claims: list[dict[str, Any]] = []
    design_rules: list[str] = []
    artifacts: list[str] = []
    improvement_over_baseline: Optional[float] = None
    next_recommendation: str = ""
    error_message: str = ""


class AutonomousLoopSummary(StrictModel):
    """Summary of the complete autonomous research loop."""

    objective: str
    loop_id: str
    iterations: list[ResearchIterationResult] = []
    total_iterations: int = 0
    best_iteration: int = -1
    best_metrics: dict[str, Any] = {}
    stopped_reason: str = ""
    supported_claims: list[str] = []
    unsupported_claims: list[str] = []
    caveats: list[str] = []
    baseline_metrics: dict[str, Any] = {}
    improvement_achieved: bool = False


class ReviewerOutput(StrictModel):
    """Structured output from the autonomous reviewer LLM call."""

    iteration_assessment: str = ""
    improvement_detected: bool = False
    improvement_detail: str = ""
    evidence_level: str = "mock"
    caveats: list[str] = []
    supported_claim: str = ""
    unsupported_claim: str = ""
    next_action: str = "continue"
    next_encoder: str = ""
    next_reconstructor: str = ""
    next_forward_mode: str = ""
    stopping_reason: str = ""
    recommendation_for_human: str = ""
