"""Agent Plan Execution schema for Phase 37."""

from __future__ import annotations

from typing import Any, Literal, Optional

from optiresearch.memory.schemas import StrictModel


class AgentPlanExecutionSpec(StrictModel):
    execution_id: str
    objective: str = ""
    seed_result_path: Optional[str] = None
    mode: Literal["dry_run", "local", "remote_opt_in"] = "dry_run"
    max_candidate_strategies: int = 6
    max_candidate_designs: int = 6
    execute_top_k: int = 1
    require_claim_gate: bool = True
    use_llm_planner: bool = False
    use_gradient_diagnosis: bool = False
    diagnosis_source_path: Optional[str] = None
    diagnosis_id: Optional[str] = None
    use_evidence_reasoner: bool = True
    allow_remote: bool = False
    remote_worker_id: Optional[str] = None
    allow_code_modification: bool = False
    snapshot_state: bool = True
    event_logging: bool = True


class AgentPlanExecutionResult(StrictModel):
    execution_id: str
    status: Literal["completed", "dry_run_only", "failed", "stopped"]
    outcome: str = ""
    objective: str = ""
    classified_failure: Optional[str] = None
    failure_category: Optional[str] = None
    candidate_strategies_count: int = 0
    candidate_strategies: list[dict[str, Any]] = []
    candidate_designs_count: int = 0
    candidate_designs: list[dict[str, Any]] = []
    plan_scores: list[dict[str, Any]] = []
    selected_design: Optional[str] = None
    selected_design_rank: Optional[int] = None
    skipped_higher_ranked_designs: list[dict[str, Any]] = []
    executable_selection_reason: str = ""
    stop_reason: Optional[str] = None
    selected_designs: list[dict[str, Any]] = []
    attempted_designs: list[dict[str, Any]] = []
    execution_result: dict[str, Any] = {}
    execution_results: list[dict[str, Any]] = []
    claim_gate_decision: dict[str, Any] = {}
    claim_gate_decisions: list[dict[str, Any]] = []
    memory_updates: list[str] = []
    memory_updated: bool = False
    state_snapshots_count: int = 0
    state_snapshot_refs: list[str] = []
    event_count: int = 0
    event_log_path: str = ""
    report_path: str = ""
    final_recommendation: str = ""
    mode: str = "dry_run"
    executed_or_dry_run: str = "dry_run"
    selected_design_executed: bool = False
    fallback_to_report_only: bool = False
    diagnosis_id: str = ""
    diagnosis_status: str = ""
    diagnosis_failure_modes: list[str] = []
    diagnosis_used_for_planning: bool = False
    diagnosis_strategy_count: int = 0
    errors: list[str] = []
