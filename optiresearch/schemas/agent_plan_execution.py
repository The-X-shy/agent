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
    use_evidence_reasoner: bool = True
    allow_remote: bool = False
    remote_worker_id: Optional[str] = None
    allow_code_modification: bool = False
    snapshot_state: bool = True
    event_logging: bool = True


class AgentPlanExecutionResult(StrictModel):
    execution_id: str
    status: Literal["completed", "dry_run_only", "failed", "stopped"]
    objective: str = ""
    classified_failure: Optional[str] = None
    failure_category: Optional[str] = None
    candidate_strategies_count: int = 0
    candidate_strategies: list[dict[str, Any]] = []
    candidate_designs_count: int = 0
    candidate_designs: list[dict[str, Any]] = []
    plan_scores: list[dict[str, Any]] = []
    selected_designs: list[dict[str, Any]] = []
    execution_results: list[dict[str, Any]] = []
    claim_gate_decisions: list[dict[str, Any]] = []
    memory_updates: list[str] = []
    state_snapshots_count: int = 0
    event_count: int = 0
    event_log_path: str = ""
    report_path: str = ""
    final_recommendation: str = ""
    mode: str = "dry_run"
    executed_or_dry_run: str = "dry_run"
    errors: list[str] = []
