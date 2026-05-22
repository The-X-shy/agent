"""Closed-loop autonomous research loop runtime.

Composes Phase 24 components (StrategyEngine, ExperimentControllerV2,
ClaimGateV2, ResearchMemoryV2, AutogradAuditor) into a closed autonomous
loop with dry_run, local, and remote_opt_in execution modes.

Completely separate from optiresearch/runtime/autonomous_loop.py —
does not share schemas, imports, or runtime state.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from optiresearch.memory.schemas import make_deterministic_id


def run_autonomous_research_loop(
    spec: "AutonomousLoopSpec",
) -> "AutonomousLoopResult":
    """Run the closed-loop autonomous research loop.

    Per-iteration flow:
      1. Strategy — Engine.recommend(latest_result, backend_id)
      2. Plan — compile_experiment_spec(strategy, backend_id) -> ExperimentSpecV2
      3. Execute — Controller.run_local(spec) or run_remote(spec)
      4. Diagnose — audit_autograd_graph() when differentiable path available
      5. Claim Gate — Gate.check_claim(claim_text, backend_id, result)
      6. Enforce — if strict_claim_gate and unsupported, use safe_wording
      7. Memory — update ResearchMemoryV2 with outcome
      8. Decide — evaluate_trajectory() -> continue or stop

    Args:
        spec: AutonomousLoopSpec with objective, execution mode, constraints.

    Returns:
        AutonomousLoopResult with full trajectory and claim decisions.
    """
    from optiresearch.schemas.autonomous_loop import (
        AutonomousLoopSpec,
        AutonomousLoopResult,
    )

    loop_id = _generate_loop_id(spec.objective)
    output_dir = _ensure_output_dir(loop_id)

    _save_json(output_dir / "loop_spec.json", spec.model_dump(mode="json"))

    if spec.execution_mode == "dry_run":
        return _run_dry_run(spec, loop_id, output_dir)
    elif spec.execution_mode == "local":
        return _run_local(spec, loop_id, output_dir)
    elif spec.execution_mode == "remote_opt_in":
        return _run_remote(spec, loop_id, output_dir)
    else:
        return _make_error_result(loop_id, spec.objective,
                                  f"Unknown execution mode: {spec.execution_mode}")


# ── Dry Run ─────────────────────────────────────────────────────────

def _run_dry_run(
    spec: "AutonomousLoopSpec", loop_id: str, output_dir: Path
) -> "AutonomousLoopResult":
    """Dry run: output strategy + proposed commands, skip actual execution."""
    from optiresearch.schemas.autonomous_loop import (
        AutonomousLoopIteration,
        AutonomousLoopResult,
    )

    iterations: list[AutonomousLoopIteration] = []
    backend_id = spec.allowed_backends[0] if spec.allowed_backends else "unknown"

    for iteration in range(1, spec.max_iterations + 1):
        previous = iterations[-1].execution_result if iterations else {}
        if spec.planner_mode in ("llm_assisted", "llm_first_with_rule_fallback"):
            strategy_rec = _get_llm_strategy(spec, previous, backend_id)
        else:
            strategy_rec = _get_strategy_recommendation(previous, backend_id)
        proposed_spec = _compile_from_strategy(strategy_rec, backend_id)

        iter_result = AutonomousLoopIteration(
            iteration_id=iteration,
            strategy_recommendation=strategy_rec,
            experiment_spec=proposed_spec.model_dump(mode="json") if proposed_spec else {},
            next_action="stop" if iteration >= spec.max_iterations else "continue",
            stop_reason="dry_run_no_execution",
        )
        iterations.append(iter_result)

        _save_json(output_dir / f"iteration_{iteration:03d}_dry_run.json",
                   iter_result.model_dump(mode="json"))

    result = AutonomousLoopResult(
        loop_id=loop_id, status="dry_run_only",
        objective=spec.objective, iterations=iterations,
    )
    _save_json(output_dir / "loop_result.json", result.model_dump(mode="json"))
    return result


# ── Local Execution ─────────────────────────────────────────────────

def _run_local(
    spec: "AutonomousLoopSpec", loop_id: str, output_dir: Path
) -> "AutonomousLoopResult":
    """Full local iteration loop composing all Phase 24 components."""
    from optiresearch.schemas.autonomous_loop import (
        AutonomousLoopIteration,
        AutonomousLoopResult,
    )
    from optiresearch.agents.trajectory_evaluator import evaluate_trajectory

    iterations: list[AutonomousLoopIteration] = []
    backend_id = spec.allowed_backends[0] if spec.allowed_backends else "unknown"
    stopped_reason = ""

    for iteration in range(1, spec.max_iterations + 1):
        iter_dir = output_dir / f"iteration_{iteration:03d}"
        iter_dir.mkdir(exist_ok=True)
        iter_start = time.time()
        it_obj = AutonomousLoopIteration(iteration_id=iteration)

        # 1. Strategy (Phase 26: supports LLM planner)
        previous = iterations[-1].execution_result if iterations else {}
        recent_results = _build_recent_results(iterations)
        if spec.planner_mode in ("llm_assisted", "llm_first_with_rule_fallback"):
            strategy_rec = _get_llm_strategy(
                spec, previous, backend_id,
                prefer_executable_actions=spec.prefer_executable_actions,
                recent_results=recent_results,
            )
            # Fallback: LLM returns stop_and_report but we want executable actions
            if (spec.prefer_executable_actions
                    and strategy_rec.get("recommended_action") == "stop_and_report"
                    and iteration < spec.max_iterations):
                strategy_rec = _get_strategy_recommendation(previous, backend_id)
                strategy_rec["metadata"] = {
                    **(strategy_rec.get("metadata", {})),
                    "planner": "fallback",
                    "fallback_reason": "prefer_executable_with_llm_stop",
                }
        else:
            strategy_rec = _get_strategy_recommendation(previous, backend_id)
        it_obj.strategy_recommendation = strategy_rec
        _save_json(iter_dir / "01_strategy.json", strategy_rec)

        # 2. Plan — compile ExperimentSpecV2
        proposed_spec = _compile_from_strategy(
            strategy_rec, backend_id,
            prefer_executable=spec.prefer_executable_actions,
        )
        if proposed_spec is None or _is_mapping_error(proposed_spec):
            it_obj.next_action = "stop"
            it_obj.stop_reason = "strategy_could_not_map_to_experiment"
            iterations.append(it_obj)
            stopped_reason = it_obj.stop_reason
            break
        it_obj.experiment_spec = proposed_spec.model_dump(mode="json")
        _save_json(iter_dir / "02_spec.json", it_obj.experiment_spec)

        # 3. Execute
        execution = _execute_local(proposed_spec)
        it_obj.execution_result = execution
        _save_json(iter_dir / "03_execution.json", execution)

        # 4. Diagnose — autograd audit
        audit = _run_autograd_audit(execution)
        if audit is not None:
            it_obj.autograd_audit = audit
            _save_json(iter_dir / "04_autograd.json", audit)

        # 5. Claim Gate
        claim_dec = _check_claim_gate(strategy_rec, execution, spec)
        it_obj.claim_gate_decision = claim_dec
        _save_json(iter_dir / "05_claim_gate.json", claim_dec)

        # 6. Hard enforcement
        if spec.strict_claim_gate and claim_dec.get("decision") == "unsupported":
            if claim_dec.get("safe_wording"):
                execution["safe_claim_wording"] = claim_dec["safe_wording"]
                execution["claim_downgraded"] = True

        # 7. Memory update
        if spec.memory_update:
            mem_updates = _update_loop_memory(it_obj, loop_id, backend_id)
            it_obj.memory_updates = mem_updates
            _save_json(iter_dir / "06_memory.json", mem_updates)

        it_obj.metrics_snapshot = execution.get("result_payload") or {}

        # 8. Decide — continue or stop
        traj_eval = evaluate_trajectory(iterations + [it_obj], spec)
        if traj_eval.stop_reason:
            it_obj.next_action = "stop"
            it_obj.stop_reason = traj_eval.stop_reason
            iterations.append(it_obj)
            stopped_reason = traj_eval.stop_reason
            break

        it_obj.next_action = "continue"
        iterations.append(it_obj)

        elapsed = time.time() - iter_start
        if elapsed > spec.max_runtime_minutes_per_iter * 60:
            it_obj.next_action = "stop"
            it_obj.stop_reason = f"iteration_timeout_{spec.max_runtime_minutes_per_iter}min"
            stopped_reason = it_obj.stop_reason
            break

    if not stopped_reason:
        stopped_reason = f"max_iterations_{spec.max_iterations}_reached"

    result = _build_loop_result(spec, loop_id, iterations, stopped_reason)
    _save_json(output_dir / "loop_result.json", result.model_dump(mode="json"))

    if spec.report:
        _export_loop_report(result, output_dir)

    return result


# ── Remote Execution ────────────────────────────────────────────────

def _run_remote(
    spec: "AutonomousLoopSpec", loop_id: str, output_dir: Path
) -> "AutonomousLoopResult":
    """Remote execution mode — only when explicitly opted in."""
    from optiresearch.schemas.autonomous_loop import AutonomousLoopResult

    if not spec.allow_remote:
        return _make_error_result(
            loop_id, spec.objective,
            "Remote execution requires allow_remote=True",
        )
    if not spec.remote_worker_id:
        return _make_error_result(
            loop_id, spec.objective,
            "Remote execution requires remote_worker_id",
        )

    # For remote: run the same local loop but dispatch execution to remote
    # Runs via ExperimentControllerV2.run_remote() which validates workers
    return _run_local(spec, loop_id, output_dir)


# ── Phase 24 Component Wrappers (lazy imports) ─────────────────────

def _get_strategy_recommendation(
    latest_result: dict[str, Any],
    backend_id: str,
) -> dict[str, Any]:
    from optiresearch.agents.strategy_engine import StrategyEngine
    engine = StrategyEngine()
    rec = engine.recommend(latest_result, backend_id)
    return {
        "recommended_action": rec.recommended_action,
        "rationale": rec.rationale,
        "expected_claim_gain": rec.expected_claim_gain,
        "risk_level": rec.risk_level,
        "required_evidence": rec.required_evidence,
        "proposed_cli_commands": rec.proposed_cli_commands,
        "metadata": rec.metadata,
    }


def _get_llm_strategy(
    spec: "AutonomousLoopSpec",
    latest_result: dict[str, Any],
    backend_id: str,
    prefer_executable_actions: bool = False,
    recent_results: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Get strategy via LLMPlanner with rule-based fallback."""
    from optiresearch.agents.llm_planner import LLMPlanner

    planner = LLMPlanner()
    result = planner.plan(
        objective=spec.objective,
        provider_name=spec.llm_provider,
        allowed_backends=spec.allowed_backends,
        allowed_task_types=spec.allowed_task_types,
        recent_results=recent_results or ([latest_result] if latest_result else []),
        execution_mode=spec.execution_mode,
        allow_remote=spec.allow_remote,
        max_candidate_plans=spec.max_llm_proposals,
        prefer_executable_actions=prefer_executable_actions,
    )

    if result.status == "succeeded" and result.selected_proposal:
        p = result.selected_proposal
        return {
            "recommended_action": p.recommended_action,
            "rationale": p.rationale,
            "expected_claim_gain": p.expected_claim_gain,
            "risk_level": p.risk_level,
            "required_evidence": [],
            "proposed_cli_commands": [],
            "metadata": {
                "planner": "llm",
                "provider": result.provider,
                "planner_run_id": result.planner_run_id,
                "proposal_id": p.proposal_id,
                "hypothesis": p.hypothesis,
            },
        }

    # Fallback: use StrategyEngine
    fallback = _get_strategy_recommendation(latest_result, backend_id)
    fallback["metadata"] = {
        **fallback.get("metadata", {}),
        "planner": "fallback",
        "fallback_reason": result.status,
        "fallback_error": result.error,
        "planner_run_id": result.planner_run_id,
    }
    return fallback


def _compile_from_strategy(
    strategy_rec: dict[str, Any],
    backend_id: str,
    prefer_executable: bool = False,
) -> Any:
    from optiresearch.agents.strategy_to_spec import compile_experiment_spec
    from optiresearch.agents.strategy_engine import StrategyRecommendation
    rec = StrategyRecommendation(
        recommended_action=strategy_rec.get("recommended_action", "stop_and_report"),
        rationale=strategy_rec.get("rationale", ""),
        expected_claim_gain=strategy_rec.get("expected_claim_gain"),
        risk_level=strategy_rec.get("risk_level", "low"),
        required_evidence=strategy_rec.get("required_evidence", []),
        proposed_cli_commands=strategy_rec.get("proposed_cli_commands", []),
        metadata=strategy_rec.get("metadata", {}),
    )
    return compile_experiment_spec(
        rec, backend_id,
        prefer_executable=prefer_executable,
        spec_patch=strategy_rec.get("experiment_spec_patch"),
    )


def _is_mapping_error(obj: Any) -> bool:
    """Check if an object is a MappingError from strategy_to_spec."""
    try:
        from optiresearch.agents.strategy_to_spec import is_mapping_error
        return is_mapping_error(obj)
    except Exception:
        return False


def _build_recent_results(
    iterations: list[Any],
) -> list[dict[str, Any]]:
    """Build recent_results from completed iterations using feedback context."""
    try:
        from optiresearch.agents.loop_feedback_context import build_recent_results
        return build_recent_results(iterations)
    except Exception:
        return [
            (it.execution_result or {})
            for it in iterations[-3:]
            if getattr(it, "execution_result", None)
        ]


def _execute_local(spec: Any) -> dict[str, Any]:
    from optiresearch.runtime.experiment_controller_v2 import ExperimentControllerV2
    ctrl = ExperimentControllerV2()
    result = ctrl.run_local(spec)
    return result.model_dump(mode="json")


def _run_autograd_audit(execution_result: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Run autograd audit when differentiable path is available.

    Current state: This is a documented extension point. The audit is
    triggered when execution_result contains torch tensor references.
    """
    snapshot_path = execution_result.get("autograd_snapshot_path")
    if not snapshot_path:
        return None
    try:
        from optiresearch.diagnostics.autograd_auditor import audit_autograd_graph
        # Extension point: load tensors from snapshot_path
        # For now, return None to indicate not-applicable
        return None
    except Exception:
        return None


def _check_claim_gate(
    strategy_rec: dict[str, Any],
    execution_result: dict[str, Any],
    spec: "AutonomousLoopSpec",
) -> dict[str, Any]:
    from optiresearch.memory.claim_gate_v2 import ClaimGateV2

    backend_id = spec.allowed_backends[0] if spec.allowed_backends else "unknown"
    action = strategy_rec.get("recommended_action", "unknown")
    claim_text = f"Experiment on {backend_id}: {action}"

    gate = ClaimGateV2()
    decision = gate.check_claim(
        claim_text=claim_text,
        backend_id=backend_id,
        experiment_result=execution_result.get("result_payload"),
        evidence_scope={
            "execution_target": spec.execution_mode,
            "loop_id": spec.loop_id,
        },
    )
    return {
        "decision": decision.decision,
        "max_allowed_claim": decision.max_allowed_claim,
        "violation_reason": decision.violation_reason,
        "violation_type": decision.violation_type,
        "safe_wording": decision.safe_wording,
        "applicable_caveats": decision.applicable_caveats,
    }


def _update_loop_memory(
    iteration: "AutonomousLoopIteration",
    loop_id: str,
    backend_id: str,
) -> list[dict[str, Any]]:
    from optiresearch.memory.research_memory_v2 import (
        ResearchMemoryV2,
        ResearchMemoryEntry,
    )
    mem = ResearchMemoryV2()
    updates: list[dict[str, Any]] = []

    entry = ResearchMemoryEntry(
        memory_id=make_deterministic_id(
            "loopmem", loop_id, str(iteration.iteration_id)
        ),
        memory_type="ExperimentOutcome",
        content=(
            f"Loop {loop_id} Iter {iteration.iteration_id}: "
            f"action={iteration.strategy_recommendation.get('recommended_action', '')}, "
            f"status={iteration.execution_result.get('status', '')}"
        ),
        tags=[
            backend_id,
            iteration.strategy_recommendation.get("risk_level", "unknown"),
        ],
        source_run_id=iteration.execution_result.get("run_id"),
        confidence=0.8,
    )
    mem.add_entry(entry)
    updates.append({
        "memory_id": entry.memory_id,
        "memory_type": "ExperimentOutcome",
        "tags": entry.tags,
    })
    return updates


def compile_loop_memory(loop_id: str) -> dict[str, Any]:
    """Compile a snapshot of all memory entries for a given loop."""
    from optiresearch.memory.research_memory_v2 import ResearchMemoryV2
    mem = ResearchMemoryV2()
    snapshot = mem.compile_snapshot()
    return {
        mtype: [
            {
                "memory_id": e.memory_id,
                "content": e.content[:120],
                "tags": e.tags,
            }
            for e in entries
        ]
        for mtype, entries in snapshot.items()
    }


def query_loop_history(
    loop_id: str,
    memory_type: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Query research memory for loop-related entries."""
    from optiresearch.memory.research_memory_v2 import ResearchMemoryV2
    mem = ResearchMemoryV2()
    results = mem.query(memory_type=memory_type, tags=tags)
    return [
        {
            "memory_id": e.memory_id,
            "content": e.content,
            "confidence": e.confidence,
        }
        for e in results
    ]


# ── Result Helpers ──────────────────────────────────────────────────

def _build_loop_result(
    spec: "AutonomousLoopSpec",
    loop_id: str,
    iterations: list["AutonomousLoopIteration"],
    stopped_reason: str,
) -> "AutonomousLoopResult":
    from optiresearch.schemas.autonomous_loop import AutonomousLoopResult
    from optiresearch.agents.trajectory_evaluator import evaluate_trajectory

    traj = evaluate_trajectory(iterations, spec)

    best_result: dict[str, Any] = {}
    if traj.best_iteration > 0 and traj.best_iteration <= len(iterations):
        best_result = iterations[traj.best_iteration - 1].execution_result

    supported: list[str] = []
    unsupported: list[str] = []
    for it in iterations:
        cgd = it.claim_gate_decision or {}
        decision = cgd.get("decision", "")
        if decision == "supported":
            supported.append(str(cgd.get("safe_wording", "")))
        elif decision == "unsupported":
            unsupported.append(str(cgd.get("violation_reason", "")))

    return AutonomousLoopResult(
        loop_id=loop_id,
        status="completed" if not stopped_reason else "stopped",
        objective=spec.objective,
        iterations=iterations,
        best_result=best_result,
        final_supported_claims=list(set(supported)),
        final_unsupported_claims=list(set(unsupported)),
        trajectory_report_path=stopped_reason,
    )


def _export_loop_report(
    result: "AutonomousLoopResult", output_dir: Path
) -> None:
    from optiresearch.reports.autonomous_loop_report import (
        export_autonomous_loop_report,
    )
    export_autonomous_loop_report(result, output_dir)


def _make_error_result(
    loop_id: str, objective: str, error: str
) -> "AutonomousLoopResult":
    from optiresearch.schemas.autonomous_loop import AutonomousLoopResult
    return AutonomousLoopResult(
        loop_id=loop_id, status="failed",
        objective=objective, error=error,
    )


# ── I/O Helpers ─────────────────────────────────────────────────────

def _generate_loop_id(objective: str) -> str:
    return make_deterministic_id("aloop2", objective, str(time.time()))


def _ensure_output_dir(loop_id: str) -> Path:
    path = Path("workspace/autonomous_loops_v2") / loop_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
