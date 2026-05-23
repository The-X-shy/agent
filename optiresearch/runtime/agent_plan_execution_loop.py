"""Agent Plan Execution Loop for Phase 37.

Wires Phase 36 subunits into autonomous plan execution:
Failure → Strategies → Designs → Scores → Execute → Claim → Memory → State → Report
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from optiresearch.agent_system.event_bus import get_event_bus
from optiresearch.agent_system.events import AgentEvent
from optiresearch.agent_system.failure_taxonomy import FailureClassifier
from optiresearch.agent_system.recovery_policy import RecoveryPolicy
from optiresearch.agent_system.state_store import StateStore
from optiresearch.agents.evidence_strategy_reasoner import EvidenceStrategyReasoner
from optiresearch.agents.experiment_design_generator import (
    ExperimentDesignCandidate,
    ExperimentDesignGenerator,
)
from optiresearch.agents.candidate_plan_evaluator import CandidatePlanEvaluator
from optiresearch.schemas.agent_plan_execution import (
    AgentPlanExecutionResult,
    AgentPlanExecutionSpec,
)


def run_agent_plan_execution(spec: AgentPlanExecutionSpec) -> AgentPlanExecutionResult:
    bus = get_event_bus()
    store = StateStore()
    errors: list[str] = []
    event_count_start = bus.count()

    bus.publish(AgentEvent.create("experiment_requested", "planner",
        payload={"execution_id": spec.execution_id, "mode": spec.mode, "objective": spec.objective}))

    # Step 1: Load seed result
    seed_result: dict[str, Any] = {}
    if spec.seed_result_path and Path(spec.seed_result_path).exists():
        try:
            seed_result = json.loads(Path(spec.seed_result_path).read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"Failed to load seed result: {e}")

    # Step 2: Classify failure
    classifier = FailureClassifier()
    fm = classifier.classify(seed_result) if seed_result else classifier.classify_by_id("unstable_native_geolens_update")
    classified_failure = fm.failure_id if fm else "unstable_native_geolens_update"
    failure_category = fm.category if fm else "gradient_instability"
    bus.publish(AgentEvent.create("negative_result_recorded", "planner",
        payload={"failure_mode": classified_failure, "category": failure_category}))

    # Step 3: Recommend recoveries
    policy = RecoveryPolicy(classifier)
    recovery_rec = policy.recommend_recovery(classified_failure)
    bus.publish(AgentEvent.create("recovery_recommended", "planner",
        payload={"failure_id": classified_failure, "recovery_count": len(recovery_rec.get("recoveries", []))}))

    # Step 4: Generate candidate strategies
    reasoner = EvidenceStrategyReasoner()
    strategies = reasoner.reason(
        objective=spec.objective, failure_mode=classified_failure,
    )
    bus.publish(AgentEvent.create("strategy_recommended", "strategy_engine",
        payload={"strategies_count": len(strategies), "top_strategy": strategies[0].strategy_id if strategies else "none"}))

    # Step 5: Generate experiment designs
    gen = ExperimentDesignGenerator()
    designs = gen.generate_designs(strategies)

    # Step 6: Score plans
    evaluator = CandidatePlanEvaluator()
    scores = evaluator.evaluate(designs)
    bus.publish(AgentEvent.create("experiment_requested", "planner",
        payload={"designs_count": len(designs), "top_score": scores[0].total_score if scores else 0}))

    # Step 7: Select top-k executable designs
    executable = [d for d in designs if d.estimated_runtime_sec > 0][:spec.execute_top_k]
    if not executable:
        executable = designs[:spec.execute_top_k]

    # Step 8: Execute or dry-run
    execution_results: list[dict[str, Any]] = []
    claim_decisions: list[dict[str, Any]] = []
    executed = False

    if spec.mode == "dry_run":
        for d in executable:
            execution_results.append({
                "design_id": d.design_id, "status": "dry_run",
                "proposed_spec": d.spec_payload,
                "backend_id": d.backend_id, "task_type": d.task_type,
                "risk_level": d.risk_level, "estimated_runtime_sec": d.estimated_runtime_sec,
            })
    elif spec.mode == "local":
        executed = True
        for d in executable:
            try:
                result = _execute_design(d)
                execution_results.append(result)
                bus.publish(AgentEvent.create(
                    "experiment_completed" if result.get("status") == "succeeded" else "experiment_failed",
                    "controller", payload=result))
                if spec.require_claim_gate:
                    decision = _run_claim_gate(d, result)
                    claim_decisions.append(decision)
            except Exception as e:
                errors.append(f"Execution failed for {d.design_id}: {e}")
                execution_results.append({"design_id": d.design_id, "status": "failed", "error": str(e)})
                bus.publish(AgentEvent.create("experiment_failed", "controller",
                    payload={"design_id": d.design_id, "error": str(e)}, severity="error"))

    # Step 9: Update StateStore
    if spec.snapshot_state:
        store.state.last_failure_mode = classified_failure
        store.state.pending_actions = [d.design_id for d in executable]
        store.state.last_strategy = {"top_strategy": strategies[0].strategy_id if strategies else "none"}
        store.snapshot()
        bus.publish(AgentEvent.create("state_snapshot_saved", "reporter",
            payload={"snapshot_count": store.state.snapshot_count}))

    # Step 10: Update ResearchMemory
    memory_updates: list[str] = []
    try:
        from optiresearch.memory.research_memory_v2 import ResearchMemoryV2, ResearchMemoryEntry
        mem = ResearchMemoryV2()
        entry_id = mem.add_entry(ResearchMemoryEntry(
            memory_id=f"phase37_plan_{spec.execution_id[:8]}",
            memory_type="NegativeResult" if not executed else "ExperimentOutcome",
            content=f"Plan execution {spec.execution_id}: {classified_failure} → {len(strategies)} strategies → {len(designs)} designs → mode={spec.mode}",
            tags=[classified_failure, spec.mode, f"strategies_{len(strategies)}"],
            confidence=0.9,
        ))
        memory_updates.append(entry_id)
        bus.publish(AgentEvent.create("memory_updated", "memory",
            payload={"entry_id": entry_id, "memory_type": "NegativeResult"}))
    except Exception as e:
        errors.append(f"Memory update failed: {e}")

    # Step 11: Export report
    report_path = _export_report(spec, strategies, designs, scores, executable,
                                  execution_results, claim_decisions,
                                  classified_failure, memory_updates, bus, store)

    # Build result
    result = AgentPlanExecutionResult(
        execution_id=spec.execution_id,
        status="dry_run_only" if spec.mode == "dry_run" else ("completed" if not errors else "failed"),
        objective=spec.objective,
        classified_failure=classified_failure,
        failure_category=failure_category,
        candidate_strategies_count=len(strategies),
        candidate_strategies=[{"strategy_id": s.strategy_id, "strategy_type": s.strategy_type,
                               "rationale": s.rationale[:200]} for s in strategies],
        candidate_designs_count=len(designs),
        candidate_designs=[{"design_id": d.design_id, "backend_id": d.backend_id,
                            "task_type": d.task_type, "risk_level": d.risk_level} for d in designs],
        plan_scores=[{"design_id": s.design_id, "total_score": s.total_score,
                      "recommendation": s.recommendation} for s in scores[:5]],
        selected_designs=[{"design_id": d.design_id, "spec_payload": d.spec_payload,
                           "backend_id": d.backend_id, "task_type": d.task_type} for d in executable],
        execution_results=execution_results,
        claim_gate_decisions=claim_decisions,
        memory_updates=memory_updates,
        state_snapshots_count=store.state.snapshot_count,
        event_count=bus.count() - event_count_start,
        event_log_path=f"workspace/agent_plan_executions/{spec.execution_id}/events.json",
        report_path=report_path,
        final_recommendation=_final_recommendation(scores, executable, spec.mode),
        mode=spec.mode,
        executed_or_dry_run="executed" if executed else "dry_run",
        errors=errors,
    )

    # Save result
    out_dir = Path("workspace/agent_plan_executions") / spec.execution_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "execution_result.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    bus.export_events(out_dir / "events.json")

    return result


def _execute_design(d: ExperimentDesignCandidate) -> dict[str, Any]:
    try:
        from optiresearch.runtime.experiment_controller_v2 import (
            ExperimentControllerV2, ExperimentSpecV2,
        )
        from optiresearch.memory.schemas import make_deterministic_id
        ctrl = ExperimentControllerV2()
        spec = ExperimentSpecV2(
            spec_id=make_deterministic_id("plan", d.design_id, str(time.time())),
            task_type=d.task_type,
            backend_id=d.backend_id,
            execution_target="local",
            spec_payload=d.spec_payload,
            execution_fidelity="deeplens_native_geometric",
        )
        result = ctrl.run_local(spec)
        return {
            "design_id": d.design_id, "status": result.status,
            "evidence_level": result.evidence_level,
            "run_id": result.run_id, "error_code": result.error_code,
        }
    except Exception as e:
        return {"design_id": d.design_id, "status": "failed", "error": str(e)}


def _run_claim_gate(d: ExperimentDesignCandidate, result: dict[str, Any]) -> dict[str, Any]:
    try:
        from optiresearch.memory.claim_gate_v2 import ClaimGateV2
        gate = ClaimGateV2()
        claim_text = f"Native GeoLens HSI co-design experiment {d.design_id} succeeded"
        decision = gate.check_claim(claim_text, d.backend_id or "deeplens_geolens_geometric", result)
        return {
            "design_id": d.design_id, "decision": decision.decision,
            "max_allowed_claim": decision.max_allowed_claim,
            "violation_type": decision.violation_type,
            "safe_wording": decision.safe_wording,
        }
    except Exception as e:
        return {"design_id": d.design_id, "error": str(e)}


def _export_report(
    spec: AgentPlanExecutionSpec,
    strategies: list, designs: list, scores: list,
    executable: list, execution_results: list, claim_decisions: list,
    classified_failure: str, memory_updates: list,
    bus, store,
) -> str:
    out_dir = Path("workspace/agent_plan_executions") / spec.execution_id
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Agent Plan Execution Report",
        "",
        f"**Execution ID:** `{spec.execution_id}`",
        f"**Objective:** {spec.objective}",
        f"**Mode:** {spec.mode}",
        f"**Status:** {'dry_run' if spec.mode == 'dry_run' else 'executed'}",
        "",
        "## Failure Classification",
        f"- **Failure:** {classified_failure}",
        "",
        "## Candidate Strategies",
        f"**Count:** {len(strategies)}",
    ]
    for s in strategies:
        lines.append(f"- [{s.strategy_type}] {s.strategy_id}: {s.rationale[:120]}...")
    lines.extend([
        "",
        "## Generated Designs",
        f"**Count:** {len(designs)}",
    ])
    for d in designs:
        lines.append(f"- {d.design_id}: {d.backend_id} {d.task_type} risk={d.risk_level}")
    lines.extend([
        "",
        "## Plan Scores",
    ])
    for s in scores[:3]:
        lines.append(f"- {s.design_id}: score={s.total_score:.3f} → {s.recommendation}")
    lines.extend([
        "",
        "## Selected Design",
    ])
    for d in executable:
        lines.append(f"- {d.design_id}: {d.spec_payload}")
    lines.extend([
        "",
        "## Execution Results" if spec.mode != "dry_run" else "## Dry Run (no execution)",
    ])
    for r in execution_results:
        lines.append(f"- {r}")
    lines.extend([
        "",
        "## Events",
        f"- Total events: {bus.count()}",
    ])
    lines.extend([
        "",
        "## State Store",
        f"- Snapshots: {store.state.snapshot_count}",
        f"- Failure mode: {store.state.last_failure_mode}",
    ])
    path = out_dir / "plan_execution_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _final_recommendation(scores: list, executable: list, mode: str) -> str:
    if mode == "dry_run":
        return f"Dry run complete. Top design: {executable[0].design_id if executable else 'none'}. Set --mode local to execute."
    if scores:
        top = scores[0]
        return f"Top-ranked design: {top.design_id} (score={top.total_score:.3f}). Recommendation: {top.recommendation}."
    return "No designs evaluated."
