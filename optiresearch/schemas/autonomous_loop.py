"""Phase 25 autonomous research loop schemas.

Separate from the existing autonomous.py schemas to avoid coupling
with the LLM-driven Phase 3-17 autonomous loop.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from optiresearch.memory.schemas import StrictModel


class AutonomousLoopSpec(StrictModel):
    """Input specification for the closed-loop autonomous research loop."""

    loop_id: str = ""
    objective: str
    seed_result_path: Optional[str] = None
    max_iterations: int = 3
    execution_mode: Literal["dry_run", "local", "remote_opt_in"] = "dry_run"
    allowed_backends: list[str] = [
        "phase_to_fft_proxy",
        "deeplens_geolens_geometric",
        "local_synthetic_hsi",
    ]
    allowed_task_types: list[str] = [
        "stable_lens_hsi_codesign",
        "native_hsi_codesign",
        "native_hsi_reconstruction_codesign",
    ]
    max_runtime_minutes_per_iter: int = 10
    allow_remote: bool = False
    remote_worker_id: Optional[str] = None
    allow_code_modification: bool = False
    strict_claim_gate: bool = True
    stop_conditions: list[str] = [
        "claim_supported",
        "no_improvement",
        "repeated_failure",
        "claim_ceiling_reached",
        "max_iterations_reached",
    ]
    memory_update: bool = True
    report: bool = True
    # Phase 26: LLM planner integration
    planner_mode: Literal["rule_based", "llm_assisted", "llm_first_with_rule_fallback"] = "rule_based"
    llm_provider: str = "mock"
    max_llm_proposals: int = 3
    require_claim_gate_for_llm: bool = True
    allow_llm_remote_plan: bool = False
    # Phase 28: executable LLM planning
    prefer_executable_actions: bool = False
    metadata: dict[str, Any] = {}


class AutonomousLoopIteration(StrictModel):
    """Record of a single iteration in the autonomous research loop."""

    iteration_id: int
    strategy_recommendation: dict[str, Any] = {}
    experiment_spec: dict[str, Any] = {}
    execution_result: dict[str, Any] = {}
    autograd_audit: Optional[dict[str, Any]] = None
    claim_gate_decision: Optional[dict[str, Any]] = None
    memory_updates: list[dict[str, Any]] = []
    next_action: str = "continue"
    stop_reason: str = ""
    metrics_snapshot: dict[str, Any] = {}


class AutonomousLoopResult(StrictModel):
    """Result of a completed (or stopped) autonomous research loop."""

    loop_id: str
    status: Literal["completed", "stopped", "failed", "dry_run_only"]
    objective: str
    iterations: list[AutonomousLoopIteration] = []
    best_result: dict[str, Any] = {}
    final_claim_decision: Optional[dict[str, Any]] = None
    final_supported_claims: list[str] = []
    final_unsupported_claims: list[str] = []
    trajectory_report_path: Optional[str] = None
    artifacts: list[str] = []
    error: Optional[str] = None
