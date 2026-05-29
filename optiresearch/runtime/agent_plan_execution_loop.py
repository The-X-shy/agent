"""Agent Plan Execution Loop for Phase 37.

Wires Phase 36 subunits into autonomous plan execution:
Failure → Strategies → Designs → Scores → Execute → Claim → Memory → State → Report
"""

from __future__ import annotations

import json
import time
import importlib.util
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
from optiresearch.agents.candidate_plan_evaluator import (
    CandidatePlanEvaluator,
    _is_lightweight_scientific_design,
)
from optiresearch.schemas.agent_plan_execution import (
    AgentPlanExecutionResult,
    AgentPlanExecutionSpec,
)


def run_agent_plan_execution(spec: AgentPlanExecutionSpec) -> AgentPlanExecutionResult:
    bus = get_event_bus()
    store = StateStore()
    errors: list[str] = []
    event_count_start = bus.count()

    bus.publish(AgentEvent.create(
        "experiment_requested",
        "planner",
        payload={"execution_id": spec.execution_id, "mode": spec.mode, "objective": spec.objective},
    ))

    seed_result: dict[str, Any] = {}
    if spec.seed_result_path and Path(spec.seed_result_path).exists():
        try:
            seed_result = json.loads(Path(spec.seed_result_path).read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"Failed to load seed result: {e}")

    classifier = FailureClassifier()
    fm = classifier.classify(seed_result) if seed_result else classifier.classify_by_id("unstable_native_geolens_update")
    classified_failure = fm.failure_id if fm else "unstable_native_geolens_update"
    failure_category = fm.category if fm else "gradient_instability"
    bus.publish(AgentEvent.create(
        "negative_result_recorded",
        "planner",
        payload={"failure_mode": classified_failure, "category": failure_category},
    ))

    policy = RecoveryPolicy(classifier)
    recovery_rec = policy.recommend_recovery(classified_failure)
    bus.publish(AgentEvent.create(
        "recovery_recommended",
        "planner",
        payload={"failure_id": classified_failure, "recovery_count": len(recovery_rec.get("recoveries", []))},
    ))

    # Phase 52-53: Diagnosis-driven planning
    diagnosis_context: dict[str, Any] = {}
    if seed_result.get("status") == "diagnosed" and isinstance(seed_result.get("failure_modes"), list):
        diagnosis_context = {
            "diagnosis_id": seed_result.get("diagnosis_id", "seed_diagnosis"),
            "status": seed_result.get("status", "diagnosed"),
            "severity": seed_result.get("severity", "medium"),
            "failure_modes": seed_result.get("failure_modes", []),
            "likely_causes": seed_result.get("likely_causes", []),
            "recommended_recoveries": seed_result.get("recommended_recoveries", []),
            "source_count": seed_result.get("source_count", 1),
        }
    if spec.use_gradient_diagnosis:
        try:
            from optiresearch.analysis.gradient_instability_analyzer import (
                analyze_gradient_instability,
            )
            source_paths = [spec.diagnosis_source_path] if spec.diagnosis_source_path else []
            if spec.seed_result_path:
                source_paths.append(spec.seed_result_path)
            diag = analyze_gradient_instability(source_paths=source_paths)
            diagnosis_context = {
                "diagnosis_id": diag.diagnosis_id,
                "status": diag.status,
                "severity": diag.severity,
                "failure_modes": diag.failure_modes,
                "likely_causes": diag.likely_causes,
                "recommended_recoveries": diag.recommended_recoveries,
                "source_count": diag.source_count,
            }
            bus.publish(AgentEvent.create("diagnosis_completed", "analyzer",
                payload={"diagnosis_id": diag.diagnosis_id, "status": diag.status}))
        except Exception as e:
            errors.append(f"Diagnosis failed: {e}")

    if diagnosis_context and diagnosis_context.get("status") == "diagnosed":
        strategist = EvidenceStrategyReasoner()
        strategies = strategist.reason_from_diagnosis(
            diagnosis=diagnosis_context, objective=spec.objective,
        )[:spec.max_candidate_strategies]
    else:
        strategies = EvidenceStrategyReasoner().reason(
            objective=spec.objective,
            failure_mode=classified_failure,
        )[:spec.max_candidate_strategies]
    bus.publish(AgentEvent.create(
        "strategy_recommended",
        "strategy_engine",
        payload={
            "strategies_count": len(strategies),
            "top_strategy": strategies[0].strategy_id if strategies else "none",
        },
    ))

    designs = ExperimentDesignGenerator().generate_designs(strategies)[:spec.max_candidate_designs]
    if spec.mode == "remote_opt_in" and spec.allow_remote:
        designs = _ensure_remote_validation_design(designs)
    evaluator = CandidatePlanEvaluator()
    if diagnosis_context:
        evaluator.set_diagnosis_context(diagnosis_context)
    scores = evaluator.evaluate(designs)
    selection = evaluator.select_executable_designs(
        scores,
        designs,
        mode=spec.mode,
        limit=spec.execute_top_k,
        allow_remote=spec.allow_remote,
    )
    selected_designs = selection.selected_designs
    bus.publish(AgentEvent.create(
        "experiment_requested",
        "planner",
        payload={
            "designs_count": len(designs),
            "top_score": scores[0].total_score if scores else 0,
            "selected_design": selection.selected_design or "none",
            "stop_reason": selection.stop_reason or "",
        },
    ))

    execution_results: list[dict[str, Any]] = []
    attempted_designs: list[dict[str, Any]] = []
    claim_gate_decisions: list[dict[str, Any]] = []
    claim_gate_decision: dict[str, Any] = {}
    execution_result: dict[str, Any] = {}
    executed = False
    fallback_to_report_only = False

    if spec.mode == "dry_run":
        for d in selected_designs:
            dry = {
                "design_id": d.design_id,
                "status": "dry_run",
                "outcome": "dry_run",
                "proposed_spec": d.spec_payload,
                "backend_id": d.backend_id,
                "task_type": d.task_type,
                "risk_level": d.risk_level,
                "estimated_runtime_sec": d.estimated_runtime_sec,
            }
            execution_results.append(dry)
        execution_result = execution_results[0] if execution_results else {}
    elif spec.mode == "local":
        if not selected_designs:
            execution_result = {}
        else:
            executed = True
            for d in _build_attempt_sequence(selected_designs, designs, max_attempts=max(3, spec.execute_top_k + 2)):
                bus.publish(AgentEvent.create(
                    "experiment_started",
                    "controller",
                    payload={"execution_id": spec.execution_id, "design_id": d.design_id},
                ))
                bus.publish(AgentEvent.create(
                    "skill_called",
                    "skill_runtime",
                    payload={"design_id": d.design_id, "skill_id": _skill_id_for_design(d), "status": "called"},
                ))
                try:
                    result = _execute_design(d)
                except Exception as e:
                    result = _failed_execution_result(d, "EXECUTION_EXCEPTION", str(e))
                    errors.append(f"Execution failed for {d.design_id}: {e}")

                execution_results.append(result)
                attempted_designs.append(_attempt_summary(result))
                completed = result.get("status") == "completed"
                bus.publish(AgentEvent.create(
                    "experiment_completed" if completed else "experiment_failed",
                    "controller",
                    payload=result,
                    severity="info" if completed else "warning",
                    related_run_id=result.get("run_id"),
                ))
                if completed:
                    execution_result = result
                    break
            if not execution_result and execution_results:
                execution_result = execution_results[-1]
            fallback_to_report_only = (
                bool(selection.selected_design)
                and execution_result.get("design_id") == "report_negative_result_doc"
                and selection.selected_design != "report_negative_result_doc"
            )

        if spec.require_claim_gate and execution_result:
            final_design = _find_design(designs, execution_result.get("design_id")) or (
                selected_designs[0] if selected_designs else None
            )
            claim_gate_decision = _run_claim_gate(final_design, execution_result)
            claim_gate_decisions.append(claim_gate_decision)

    elif spec.mode == "remote_opt_in":
        if not spec.allow_remote:
            execution_result = {
                "status": "stopped",
                "outcome": "remote_not_allowed",
                "design_id": selection.selected_design or "none",
                "evidence_level": "structured_unsupported",
                "errors": [{"type": "REMOTE_NOT_ALLOWED", "message": "allow_remote must be true for remote_opt_in mode"}],
            }
            errors.append("Remote execution requires --allow-remote flag")
        elif not spec.remote_worker_id:
            execution_result = {
                "status": "stopped",
                "outcome": "remote_no_worker",
                "design_id": selection.selected_design or "none",
                "evidence_level": "structured_unsupported",
                "errors": [{"type": "REMOTE_NO_WORKER", "message": "remote_worker_id is required for remote_opt_in mode"}],
            }
            errors.append("Remote execution requires --remote-worker-id")
        elif not selected_designs:
            execution_result = {}
        else:
            executed = True
            for d in selected_designs[:max(1, spec.execute_top_k)]:
                handler_id = getattr(d, "handler_id", "") or _handler_id_for_design(d)
                cap = _get_handler_cap(handler_id)
                if not cap or not cap.supports_remote:
                    result = {
                        "status": "unsupported",
                        "outcome": "handler_does_not_support_remote",
                        "design_id": d.design_id,
                        "evidence_level": "structured_unsupported",
                        "errors": [{"type": "HANDLER_NO_REMOTE", "message": f"Handler '{handler_id}' does not support remote execution"}],
                    }
                    execution_results.append(result)
                    attempted_designs.append(_attempt_summary(result))
                    continue

                from optiresearch.remote.worker_registry import validate_remote_worker_requirements
                requirements = validate_remote_worker_requirements(cap, spec.remote_worker_id)
                if not requirements.get("requirements_met"):
                    result = {
                        "status": "stopped",
                        "outcome": "remote_worker_requirements_not_met",
                        "design_id": d.design_id,
                        "handler_id": handler_id,
                        "task_type": d.task_type,
                        "backend_id": d.backend_id,
                        "evidence_level": "needs_followup",
                        "execution_target": "remote_wsl",
                        "remote_worker_id": spec.remote_worker_id,
                        "remote_validation_passed": False,
                        "metrics": {},
                        "artifacts": [],
                        "errors": requirements.get("errors") or [
                            {
                                "type": "REMOTE_WORKER_REQUIREMENTS_NOT_MET",
                                "message": ", ".join(requirements.get("missing_requirements", [])),
                            }
                        ],
                        "caveats": ["Remote worker requirements were not met"],
                        "remote_worker_requirements": requirements,
                        "handler_claim_ceiling": "needs_followup",
                    }
                    execution_results.append(result)
                    attempted_designs.append(_attempt_summary(result))
                    execution_result = result
                    bus.publish(AgentEvent.create(
                        "remote_validation_failed",
                        "controller",
                        payload=result,
                        severity="warning",
                    ))
                    break

                bus.publish(AgentEvent.create("remote_execution_requested", "controller",
                    payload={"execution_id": spec.execution_id, "design_id": d.design_id, "worker_id": spec.remote_worker_id}))
                bus.publish(AgentEvent.create("remote_execution_started", "controller",
                    payload={"design_id": d.design_id, "handler_id": handler_id}))

                try:
                    if _is_diagnostic_design(d):
                        bus.publish(AgentEvent.create(
                            "diagnostic_remote_execution_started", "diagnostic_runtime",
                            payload={"design_id": d.design_id, "worker_id": spec.remote_worker_id}))
                        remote_result = _execute_remote_diagnostic_design(d, spec.remote_worker_id)
                        bus.publish(AgentEvent.create(
                            "diagnostic_remote_execution_completed" if remote_result.get("status") == "completed" else "diagnosis_failed",
                            "diagnostic_runtime",
                            payload=remote_result,
                            severity="info" if remote_result.get("status") == "completed" else "warning",
                        ))
                    else:
                        remote_result = _execute_remote_design(d, spec.remote_worker_id)
                    bus.publish(AgentEvent.create(
                        "remote_execution_completed" if remote_result.get("status") == "completed" else "remote_execution_failed",
                        "controller",
                        payload=remote_result,
                        severity="info" if remote_result.get("status") == "completed" else "warning",
                    ))
                    if not _is_diagnostic_design(d):
                        bus.publish(AgentEvent.create(
                            "remote_validation_passed" if remote_result.get("remote_validation_passed") else "remote_validation_failed",
                            "controller",
                            payload=remote_result,
                            severity="info" if remote_result.get("remote_validation_passed") else "warning",
                            related_run_id=remote_result.get("run_id"),
                            related_job_id=remote_result.get("remote_job_id"),
                        ))
                    if remote_result.get("artifacts"):
                        bus.publish(AgentEvent.create(
                            "artifact_ingested",
                            "controller",
                            payload={
                                "remote_job_id": remote_result.get("remote_job_id", ""),
                                "artifact_return_path": remote_result.get("artifact_return_path", ""),
                                "artifacts": remote_result.get("artifacts", []),
                            },
                            related_run_id=remote_result.get("run_id"),
                            related_job_id=remote_result.get("remote_job_id"),
                        ))
                except Exception as e:
                    remote_result = {
                        "status": "failed",
                        "design_id": d.design_id,
                        "evidence_level": "structured_unsupported",
                        "errors": [{"type": "REMOTE_EXECUTION_EXCEPTION", "message": str(e)}],
                    }
                    bus.publish(AgentEvent.create("remote_execution_failed", "controller",
                        payload={"design_id": d.design_id, "error": str(e)}, severity="error"))
                    errors.append(f"Remote execution failed for {d.design_id}: {e}")

                execution_results.append(remote_result)
                attempted_designs.append(_attempt_summary(remote_result))
                if remote_result.get("status") == "completed":
                    execution_result = remote_result
                    break
            if not execution_result and execution_results:
                execution_result = execution_results[-1]

        if spec.require_claim_gate and execution_result:
            final_design = _find_design(designs, execution_result.get("design_id")) or (
                selected_designs[0] if selected_designs else None
            )
            claim_gate_decision = _run_claim_gate(final_design, execution_result)
            claim_gate_decisions.append(claim_gate_decision)

    # Phase 48: Auto-bind artifact IDs into ClaimEvidence
    evidence_binding: dict[str, Any] = {}
    if execution_result and claim_gate_decision and spec.require_claim_gate:
        try:
            from optiresearch.memory.claim_evidence import (
                bind_artifacts_from_claim_gate_decision,
                ClaimEvidenceManager,
            )
            evidence_ids = claim_gate_decision.get("evidence_artifact_ids", [])
            if evidence_ids:
                ev_manager = ClaimEvidenceManager()
                claim_text = claim_gate_decision.get("safe_wording", f"Plan execution {spec.execution_id}")
                claim = ev_manager.create_claim(claim_text, {
                    "execution_id": spec.execution_id,
                    "design_id": execution_result.get("design_id", ""),
                    "evidence_level": execution_result.get("evidence_level", ""),
                })
                edges = bind_artifacts_from_claim_gate_decision(
                    claim.claim_id, claim_gate_decision,
                )
                for edge in edges:
                    ev_manager.attach_support(
                        claim.claim_id, edge["artifact_id"],
                        edge["score"], edge.get("relation", "supports"),
                    )
                evidence_binding = {
                    "claim_id": claim.claim_id,
                    "evidence_edges_count": len(edges),
                    "evidence_artifact_ids": evidence_ids,
                    "evidence_completeness": claim_gate_decision.get("evidence_completeness", ""),
                }
                execution_result["claim_id"] = claim.claim_id
                execution_result["evidence_edges_count"] = len(edges)
        except Exception as e:
            errors.append(f"Evidence binding failed: {e}")

    memory_updates: list[str] = []
    memory_updated = False
    if execution_result or spec.mode == "dry_run":
        try:
            from optiresearch.memory.research_memory_v2 import ResearchMemoryV2
            mem = ResearchMemoryV2()
            entry_id = mem.record_plan_execution_outcome(
                execution_id=spec.execution_id,
                selected_design=selection.selected_design,
                execution_result=execution_result or {"status": "dry_run", "evidence_level": "dry_run"},
                attempted_designs=attempted_designs,
                skipped_higher_ranked_designs=selection.skipped_higher_ranked_designs,
                claim_decision=claim_gate_decision,
            )
            memory_updates.append(entry_id)
            memory_updated = True
            bus.publish(AgentEvent.create(
                "memory_updated",
                "memory",
                payload={"entry_id": entry_id, "memory_type": "ExperimentOutcome"},
            ))
        except Exception as e:
            errors.append(f"Memory update failed: {e}")

    state_snapshot_refs: list[str] = []
    if spec.snapshot_state:
        store.state.active_objective = spec.objective
        store.state.last_failure_mode = classified_failure
        if execution_result:
            store.record_plan_execution_outcome(
                execution_id=spec.execution_id,
                selected_design=selection.selected_design,
                execution_result=execution_result,
                claim_decision=claim_gate_decision,
                attempted_designs=attempted_designs,
                skipped_higher_ranked_designs=selection.skipped_higher_ranked_designs,
            )
        else:
            store.state.pending_actions = [s.get("design_id", "") for s in selection.skipped_higher_ranked_designs]
            store.state.last_strategy = {"execution_id": spec.execution_id, "selected_design": "none"}
            store.save()
        store.snapshot()
        state_snapshot_refs.append(f"workspace/agent_state/snapshots/snapshot_{store.state.snapshot_count:04d}.json")
        bus.publish(AgentEvent.create(
            "state_snapshot_saved",
            "reporter",
            payload={"snapshot_count": store.state.snapshot_count},
        ))

    status = _result_status(spec.mode, selection.stop_reason, execution_result, errors)
    outcome = execution_result.get("outcome") or execution_result.get("evidence_level") or (
        "dry_run" if spec.mode == "dry_run" else selection.stop_reason or ""
    )
    result = AgentPlanExecutionResult(
        execution_id=spec.execution_id,
        status=status,
        outcome=outcome,
        objective=spec.objective,
        classified_failure=classified_failure,
        failure_category=failure_category,
        candidate_strategies_count=len(strategies),
        candidate_strategies=[{
            "strategy_id": s.strategy_id,
            "strategy_type": s.strategy_type,
            "rationale": s.rationale[:200],
        } for s in strategies],
        candidate_designs_count=len(designs),
        candidate_designs=[{
            "design_id": d.design_id,
            "backend_id": d.backend_id,
            "task_type": d.task_type,
            "risk_level": d.risk_level,
            "required_skills": d.required_skills,
        } for d in designs],
        plan_scores=[{
            "design_id": s.design_id,
            "total_score": s.total_score,
            "recommendation": s.recommendation,
            "reason": s.reason,
            "diagnosis_score_bonus": getattr(s, "diagnosis_score_bonus", 0.0),
            "diagnosis_factors_used": getattr(s, "diagnosis_factors_used", []),
            "scoring_explanation": getattr(s, "scoring_explanation", ""),
        } for s in scores],
        selected_design=selection.selected_design,
        selected_design_rank=selection.selected_design_rank,
        skipped_higher_ranked_designs=selection.skipped_higher_ranked_designs,
        executable_selection_reason=selection.executable_selection_reason,
        stop_reason=selection.stop_reason,
        selected_designs=[{
            "design_id": d.design_id,
            "spec_payload": d.spec_payload,
            "backend_id": d.backend_id,
            "task_type": d.task_type,
        } for d in selected_designs],
        attempted_designs=attempted_designs,
        execution_result=execution_result,
        execution_results=execution_results,
        claim_gate_decision=claim_gate_decision,
        claim_gate_decisions=claim_gate_decisions,
        memory_updates=memory_updates,
        memory_updated=memory_updated,
        state_snapshots_count=store.state.snapshot_count,
        state_snapshot_refs=state_snapshot_refs,
        event_count=bus.count() - event_count_start,
        event_log_path=f"workspace/agent_plan_executions/{spec.execution_id}/events.json",
        report_path=f"workspace/agent_plan_executions/{spec.execution_id}/plan_execution_report.md",
        final_recommendation=_final_recommendation(scores, selected_designs, spec.mode, execution_result, fallback_to_report_only),
        mode=spec.mode,
        executed_or_dry_run="executed" if executed else "dry_run",
        selected_design_executed=bool(
            selection.selected_design
            and any(a.get("design_id") == selection.selected_design for a in attempted_designs)
        ),
        fallback_to_report_only=fallback_to_report_only,
        diagnosis_id=diagnosis_context.get("diagnosis_id", ""),
        diagnosis_status=diagnosis_context.get("status", ""),
        diagnosis_failure_modes=diagnosis_context.get("failure_modes", []),
        diagnosis_used_for_planning=bool(diagnosis_context),
        diagnosis_strategy_count=len(strategies) if diagnosis_context else 0,
        errors=errors,
    )

    out_dir = Path("workspace/agent_plan_executions") / spec.execution_id
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "execution_result.json"
    result_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    bus.export_events(out_dir / "events.json")
    try:
        from optiresearch.reports.agent_plan_execution_report import export_agent_plan_execution_report
        report_path = export_agent_plan_execution_report(spec.execution_id)
        result.report_path = str(report_path)
        result_path.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        errors.append(f"Report export failed: {e}")

    return result


def _extract_component_from_design(d: ExperimentDesignCandidate) -> str:
    """Extract component name from design spec_payload or design_id."""
    component = d.spec_payload.get("component", "") if d.spec_payload else ""
    if component:
        return component
    did = d.design_id.lower()
    if "fresnel" in did:
        return "fresnel"
    if "binary2phase" in did:
        return "binary2phase"
    if "diffractive" in did:
        return "diffractive"
    return "fresnel"


def _is_diagnostic_design(d: ExperimentDesignCandidate) -> bool:
    did = d.design_id
    return any(kw in did for kw in (
        "autograd", "trainable_parameter", "curriculum_probe",
        "regularized_probe", "component_first", "surface_freeze_unfreeze",
        "verify_trainable", "component_level_geolens",
    ))


def _execute_diagnostic_design(d: ExperimentDesignCandidate) -> dict[str, Any]:
    did = d.design_id
    if "autograd_audit" in did or "autograd" in did:
        from optiresearch.runtime.deeplens_autograd_audit import run_deeplens_autograd_audit
        result = run_deeplens_autograd_audit(device="cpu")
        return {
            "status": "completed" if result["status"] == "succeeded" else result["status"],
            "outcome": "diagnostic_execution",
            "design_id": d.design_id, "task_type": d.task_type,
            "backend_id": d.backend_id, "evidence_level": result["evidence_level"],
            "metrics": {k: result[k] for k in (
                "trainable_param_count", "params_with_grad", "grad_norm_max",
                "graph_connected", "detach_suspected", "psf_requires_grad",
                "loss_requires_grad",
            ) if k in result},
            "artifacts": [], "errors": [],
            "caveats": ["Diagnostic evidence only — does not confirm optical design improvement"],
            "diagnosis": result.get("diagnosis", []),
            "recommended_next_strategy": result.get("recommended_next_strategy", ""),
        }
    if "trainable_parameter" in did or "verify_trainable" in did or "surface_freeze" in did:
        from optiresearch.runtime.deeplens_trainable_parameter_inspection import (
            inspect_deeplens_trainable_parameters,
        )
        result = inspect_deeplens_trainable_parameters(device="cpu")
        return {
            "status": "completed" if result["status"] == "succeeded" else result["status"],
            "outcome": "diagnostic_execution",
            "design_id": d.design_id, "task_type": d.task_type,
            "backend_id": d.backend_id, "evidence_level": result["evidence_level"],
            "metrics": {k: result[k] for k in (
                "parameter_count", "trainable_count",
            ) if k in result},
            "artifacts": [], "errors": [],
            "caveats": ["Parameter inspection — does not confirm optical design improvement"],
            "recommended_strategy": result.get("recommended_strategy", ""),
        }
    if "curriculum_probe" in did:
        from optiresearch.runtime.deeplens_curriculum_probe import run_deeplens_curriculum_probe
        result = run_deeplens_curriculum_probe(max_steps=3, device="cpu")
        return {
            "status": "completed" if result["status"] == "succeeded" else result["status"],
            "outcome": "diagnostic_execution",
            "design_id": d.design_id, "task_type": d.task_type,
            "backend_id": d.backend_id, "evidence_level": result["evidence_level"],
            "metrics": {k: result.get(k) for k in (
                "stages_completed", "curriculum_progress",
            )},
            "artifacts": [], "errors": [],
            "caveats": ["Curriculum probe only — not a validated optical design improvement"],
        }
    if "regularized_probe" in did:
        from optiresearch.runtime.deeplens_regularized_probe import run_deeplens_regularized_probe
        result = run_deeplens_regularized_probe(max_steps=3, device="cpu")
        return {
            "status": "completed" if result["status"] == "succeeded" else result["status"],
            "outcome": "diagnostic_execution",
            "design_id": d.design_id, "task_type": d.task_type,
            "backend_id": d.backend_id, "evidence_level": result["evidence_level"],
            "metrics": {k: result.get(k) for k in (
                "base_loss", "regularized_loss", "update_accepted",
            )},
            "artifacts": [], "errors": [],
            "caveats": ["Regularization probe — not a validated optical design improvement"],
        }
    if "component_first" in did or "component_level" in did:
        component = _extract_component_from_design(d)
        try:
            from optiresearch.schemas.component_probe import ComponentProbeSpec, make_component_probe_id
            from optiresearch.runtime.deeplens_component_first_probe import run_deeplens_component_probe
            spec = ComponentProbeSpec(
                probe_id=make_component_probe_id(component, "parameter_sanity_check"),
                component=component,
                objective="parameter_sanity_check",
                max_steps=5,
                device="cpu",
            )
            result = run_deeplens_component_probe(spec)
        except Exception as exc:
            return _unsupported_execution_result(
                d, "COMPONENT_PROBE_EXECUTION_FAILED",
                f"Component probe execution failed: {exc}",
            )
        return {
            "status": "completed" if result.status == "succeeded" else result.status,
            "outcome": "diagnostic_execution",
            "design_id": d.design_id, "task_type": d.task_type,
            "backend_id": result.backend_id,
            "evidence_level": result.evidence_level,
            "metrics": {
                "component": result.component,
                "surface_class": result.surface_class,
                "differentiable": result.differentiable,
                "trainable_param_count": result.trainable_param_count,
                "params_with_grad": result.params_with_grad,
                "parameters_changed": result.parameters_changed,
                "gradient_norm": result.gradient_norm,
                "loss_before": result.loss_before,
                "loss_after": result.loss_after,
                "claim_ceiling": result.claim_ceiling,
                "error_code": result.error_code,
            },
            "artifacts": [], "errors": [],
            "caveats": [
                "Component probe — not a validated optical design improvement",
                "Component-level evidence only — does not confirm lens-level optimization",
            ],
        }
    return _unsupported_execution_result(d, "UNKNOWN_DIAGNOSTIC_DESIGN", f"No handler for {d.design_id}")


def _execute_remote_diagnostic_design(d: ExperimentDesignCandidate, worker_id: str) -> dict[str, Any]:
    """Execute a diagnostic design on a remote WSL worker.

    Maps diagnostic design_id keywords to remote job functions that dispatch
    via SSH to the WSL worker, then downloads and ingests results.
    """
    did = d.design_id
    from optiresearch.runtime import remote_jobs

    if "autograd" in did:
        payload = remote_jobs.run_remote_deeplens_autograd_audit(
            worker_id, lens_file="auto:cooke", device="cpu",
        )
    elif "trainable_parameter" in did or "verify_trainable" in did or "surface_freeze" in did:
        payload = remote_jobs.run_remote_deeplens_trainable_parameter_inspection(
            worker_id, lens_file="auto:cooke", device="cpu",
        )
    elif "curriculum_probe" in did:
        payload = remote_jobs.run_remote_deeplens_curriculum_probe(
            worker_id, max_steps=3, device="cpu",
        )
    elif "regularized_probe" in did:
        payload = remote_jobs.run_remote_deeplens_regularized_probe(
            worker_id, max_steps=3, device="cpu",
        )
    elif "component_first" in did or "component_level" in did:
        component = _extract_component_from_design(d)
        payload = remote_jobs.run_remote_deeplens_component_probe(
            worker_id, component=component, objective="parameter_sanity_check",
            max_steps=5, device="cpu",
        )
    else:
        return _unsupported_execution_result(
            d, "NO_REMOTE_DIAGNOSTIC_HANDLER",
            f"No remote diagnostic handler for design: {did}",
        )

    result = payload["result"]
    metrics = result.metrics_summary if hasattr(result, "metrics_summary") else {}
    status = "completed" if result.status == "succeeded" else "failed"

    return {
        "status": status,
        "outcome": "remote_diagnostic_execution" if status == "completed" else "remote_diagnostic_failed",
        "design_id": d.design_id,
        "task_type": d.task_type if hasattr(d, "task_type") else "diagnostic_probe",
        "backend_id": d.backend_id if hasattr(d, "backend_id") else "deeplens_geolens_geometric",
        "evidence_level": "diagnostic_evidence",
        "handler_id": getattr(d, "handler_id", "") or _handler_id_for_design(d),
        "execution_target": "remote_wsl",
        "remote_worker_id": worker_id,
        "remote_job_id": result.job_id,
        "remote_validation_passed": False,
        "execution_fidelity": "deeplens_native_geometric",
        "proxy_fallback_used": False,
        "deeplens_native_psf_path": "geolens.psf_geometric",
        "full_wave_optics": False,
        "phase_to_fft_proxy_used": False,
        "diagnostic_type": metrics.get("diagnostic_type", did),
        "metrics": metrics,
        "artifacts": [
            str(Path(result.local_output_dir) / result.job_id / fn)
            for fn in ("result.json", "diagnostic_metrics.json")
        ] if result.local_output_dir else [],
        "artifact_return_path": result.local_output_dir or "",
        "artifact_ids": [],
        "evidence_artifact_ids": [],
        "artifact_manifest_path": "",
        "artifact_manifest_complete": False,
        "artifact_ingestion_status": "",
        "sha256_verified": False,
        "errors": [],
        "caveats": [
            "Diagnostic evidence only — does not confirm optical design improvement",
            f"Executed on remote WSL worker: {worker_id}",
        ],
        "run_id": result.remote_run_id or result.job_id,
        "error_code": result.error_code,
    }


def _execute_design(d: ExperimentDesignCandidate) -> dict[str, Any]:
    if "report_generation" in d.required_skills or d.design_id == "report_negative_result_doc":
        return _execute_report_design(d)

    if "component_surrogate" in d.design_id or d.task_type == "component_surrogate_hsi_codesign":
        return _execute_component_surrogate_hsi_design(d)

    if _is_diagnostic_design(d):
        return _execute_diagnostic_design(d)

    if d.design_id == "param_reduction_sweep" or d.spec_payload.get("param_subset"):
        return _execute_param_reduction_sweep(d)

    if _is_lightweight_scientific_design(d):
        return _execute_lightweight_scientific_design(d)

    if d.design_id == "backend_switch_waveoptics_coherent" or d.backend_id == "deeplens_coherent_asm":
        return _needs_followup_execution_result(
            d,
            "COHERENT_ASM_REQUIRES_GRAD_FALSE",
            "DeepLens coherent ASM path is probe-only because PSF tensors do not expose usable gradients.",
            metrics={"requires_grad": False, "probe_only": True},
        )

    if (
        d.backend_id == "deeplens_geolens_geometric"
        and d.task_type in ("stable_lens_hsi_codesign", "native_lens_simulation_codesign")
        and not _deeplens_available()
    ):
        code = "DIFFRACTIVE_LENS_LOCAL_UNAVAILABLE" if d.spec_payload.get("candidate") == "DiffractiveLens" else "DEEPLENS_UNAVAILABLE"
        return _unsupported_execution_result(
            d,
            code,
            "DeepLens native GeoLens execution is unavailable in this local environment.",
        )

    try:
        from optiresearch.runtime.experiment_controller_v2 import (
            ExperimentControllerV2,
            ExperimentSpecV2,
        )
        from optiresearch.memory.schemas import make_deterministic_id

        ctrl = ExperimentControllerV2()
        controller_spec = ExperimentSpecV2(
            spec_id=make_deterministic_id("plan", d.design_id, str(time.time())),
            task_type=d.task_type,
            backend_id=d.backend_id,
            execution_target="local",
            spec_payload=d.spec_payload,
            execution_fidelity="deeplens_native_geometric",
            allow_proxy_fallback=False,
            require_deeplens_native=d.backend_id == "deeplens_geolens_geometric",
        )
        result = ctrl.run_local(controller_spec)
        return _controller_result_to_execution_result(d, result)
    except Exception as e:
        return _failed_execution_result(d, "EXECUTION_EXCEPTION", str(e))


def _execute_report_design(d: ExperimentDesignCandidate) -> dict[str, Any]:
    from optiresearch.skills.runtime_v2 import SkillRuntimeV2

    skill_result = SkillRuntimeV2().execute_skill(
        "report_generation",
        {"report_type": "agent_plan_negative_result"},
    )
    output = skill_result.output or {}
    if skill_result.status == "succeeded":
        path = output.get("path", "")
        return {
            "status": "completed",
            "outcome": "report_only",
            "design_id": d.design_id,
            "task_type": "report_generation",
            "backend_id": d.backend_id,
            "evidence_level": "report_only",
            "metrics": {"report_generated": True},
            "artifacts": [path] if path else [],
            "errors": [],
            "caveats": ["Report-only evidence does not support optical improvement"],
            "skill_result": {"skill_id": skill_result.skill_id, "status": skill_result.status},
        }
    return {
        "status": "unsupported",
        "outcome": "structured_unsupported",
        "design_id": d.design_id,
        "task_type": "report_generation",
        "backend_id": d.backend_id,
        "evidence_level": "structured_unsupported",
        "metrics": {},
        "artifacts": [],
        "errors": [{"type": "REPORT_GENERATION_UNSUPPORTED", "message": "; ".join(skill_result.errors)}],
        "caveats": ["Report generation did not complete"],
    }


def _execute_lightweight_scientific_design(d: ExperimentDesignCandidate) -> dict[str, Any]:
    from optiresearch.runtime.lightweight_experiments import run_lightweight_mse_only_hsi

    result = run_lightweight_mse_only_hsi(
        backend_id=d.backend_id or "phase_to_fft_proxy",
        max_steps=d.spec_payload.get("max_steps", 10),
        optical_lr=d.spec_payload.get("optical_lr", 1e-6),
        bands=d.spec_payload.get("bands", 4),
    )
    return _lightweight_result_to_execution_result(d, result)


def _execute_component_surrogate_hsi_design(d: ExperimentDesignCandidate) -> dict[str, Any]:
    from optiresearch.runtime.component_surrogate_hsi_codesign import (
        run_component_surrogate_hsi_codesign,
    )
    from optiresearch.schemas.component_surrogate_psf import (
        ComponentSurrogateHSICoDesignSpec,
    )

    payload = d.spec_payload or {}
    component = payload.get("component", _extract_component_from_design(d))
    spec = ComponentSurrogateHSICoDesignSpec(
        component_type=component,
        dataset=payload.get("dataset", "synthetic"),
        steps=int(payload.get("steps", 3)),
        band_count=int(payload.get("bands", 4)),
        image_size=int(payload.get("image_size", 16)),
        psf_size=int(payload.get("psf_size", 9)),
        batch_size=int(payload.get("batch_size", 1)),
        device=payload.get("device", "cpu"),
    )
    result = run_component_surrogate_hsi_codesign(spec)
    completed = result.status == "succeeded"
    metrics = {
        "component_type": result.component_type,
        "reconstruction_loss_before": result.reconstruction_loss_before,
        "reconstruction_loss_after": result.reconstruction_loss_after,
        "mse_before": result.mse_before,
        "mse_after": result.mse_after,
        "psnr_before": result.psnr_before,
        "psnr_after": result.psnr_after,
        "sam_before": result.sam_before,
        "sam_after": result.sam_after,
        "component_grad_norm_max": result.component_grad_norm_max,
        "component_parameter_changed": result.component_parameter_changed,
        "psf_requires_grad": result.psf_requires_grad,
        "loss_requires_grad": result.loss_requires_grad,
        "synthetic_data": True,
        "physical_backend": False,
    }
    return {
        "status": "completed" if completed else result.status,
        "outcome": result.evidence_level if completed else "structured_unsupported",
        "design_id": d.design_id,
        "task_type": d.task_type,
        "backend_id": "component_surrogate_psf",
        "evidence_level": result.evidence_level,
        "backend_evidence_level": result.evidence_level,
        "metrics": metrics,
        "artifacts": list(result.artifacts),
        "errors": [{"type": e, "message": e} for e in result.errors],
        "caveats": [
            "Component surrogate PSF — not full GeoLens PSF",
            "Synthetic HSI data — no real HSI performance claim",
            "Component-level update — not lens-level physical validation",
        ],
        "run_id": result.run_id,
        "handler_id": "component_surrogate_hsi_codesign",
        "metadata": result.metadata,
        "handler_claim_ceiling": "component_surrogate_hsi_codesign",
        "design_backend_claim_ceiling": _get_backend_claim_ceiling("component_surrogate_psf"),
        "dataset_claim_ceiling": "lightweight_scientific_execution",
        "execution_fidelity_claim_ceiling": "component_surrogate_hsi_codesign",
        "synthetic_data": True,
        "physical_backend": False,
        "native_backend": False,
        "full_wave_optics": False,
        "phase_to_fft_proxy_used": True,
        "claim_ceiling": result.claim_ceiling,
    }


def _lightweight_result_to_execution_result(
    d: ExperimentDesignCandidate, result: Any
) -> dict[str, Any]:
    completed = result.status == "succeeded"
    payload = result.result_payload or {}
    errors = list(result.errors or [])
    metrics: dict[str, Any] = {}
    for key in (
        "reconstruction_loss_before",
        "reconstruction_loss_after",
        "best_reconstruction_loss",
        "mse_before",
        "mse_after",
        "psnr_before",
        "psnr_after",
        "improvement_detected",
        "metrics_valid",
        "accepted_update_count",
        "execution_time_sec",
        "synthetic_data",
        "physical_backend",
        "mse_only_objective",
    ):
        if key in payload:
            metrics[key] = payload[key]
    return {
        "status": "completed" if completed else "failed",
        "outcome": "lightweight_scientific_execution" if completed else "structured_unsupported",
        "design_id": d.design_id,
        "task_type": d.task_type,
        "backend_id": payload.get("backend_id", d.backend_id),
        "evidence_level": "lightweight_scientific_execution" if completed else "structured_unsupported",
        "backend_evidence_level": result.evidence_level,
        "metrics": metrics,
        "artifacts": list(result.artifact_paths or []),
        "errors": errors,
        "caveats": [
            "MSE-only synthetic HSI experiment — not native DeepLens simulation",
            "Synthetic HSI data — real HSI performance may differ",
        ],
        "run_id": result.run_id,
        "handler_id": "objective_redesign_simpler_metric",
        "metadata": {
            "synthetic_data": True,
            "physical_backend": False,
            "mse_only_objective": True,
            "deepens_used": False,
            "psf_method": "fft_fraunhofer",
        },
        "handler_claim_ceiling": "lightweight_scientific_execution",
        "design_backend_claim_ceiling": _get_backend_claim_ceiling(d.backend_id),
        "dataset_claim_ceiling": "lightweight_scientific_execution",
        "execution_fidelity_claim_ceiling": "lightweight_scientific_execution",
        "synthetic_data": True,
        "physical_backend": False,
        "native_backend": False,
        "full_wave_optics": False,
        "phase_to_fft_proxy_used": True,
    }


def _execute_param_reduction_sweep(d: ExperimentDesignCandidate) -> dict[str, Any]:
    from optiresearch.runtime.local_scientific_handlers import (
        run_param_reduction_sweep_lightweight,
    )

    result = run_param_reduction_sweep_lightweight(
        design=d,
        max_steps=d.spec_payload.get("max_steps", 3),
        optical_lr=d.spec_payload.get("optical_lr", 1e-6),
        bands=d.spec_payload.get("bands", 4),
    )
    return _param_reduction_result_to_execution_result(d, result)


def _param_reduction_result_to_execution_result(
    d: ExperimentDesignCandidate, result: Any
) -> dict[str, Any]:
    completed = result.status == "succeeded"
    payload = result.result_payload or {}
    errors = list(result.errors or [])
    metrics: dict[str, Any] = {}
    for key in (
        "reconstruction_loss_before",
        "reconstruction_loss_after",
        "best_reconstruction_loss",
        "mse_before",
        "mse_after",
        "psnr_before",
        "psnr_after",
        "improvement_detected",
        "metrics_valid",
        "accepted_update_count",
        "execution_time_sec",
        "configs_tested",
        "best_k",
        "synthetic_data",
        "physical_backend",
        "parameter_changed",
    ):
        if key in payload:
            metrics[key] = payload[key]
    return {
        "status": "completed" if completed else "failed",
        "outcome": "lightweight_scientific_execution" if completed else "structured_unsupported",
        "design_id": d.design_id,
        "task_type": d.task_type,
        "backend_id": payload.get("backend_id", d.backend_id),
        "evidence_level": "lightweight_scientific_execution" if completed else "structured_unsupported",
        "backend_evidence_level": result.evidence_level,
        "metrics": metrics,
        "artifacts": list(result.artifact_paths or []),
        "errors": errors,
        "caveats": [
            "Param reduction sweep on synthetic HSI — not native DeepLens simulation",
            "Low-dimensional pseudo-optical parameter sweep",
            "Synthetic HSI data — real HSI performance may differ",
        ],
        "run_id": result.run_id,
        "handler_id": "param_reduction_sweep",
        "metadata": {
            "synthetic_data": True,
            "physical_backend": False,
            "native_backend": False,
            "handler_id": "param_reduction_sweep",
            "deepens_used": False,
        },
        "handler_claim_ceiling": "lightweight_scientific_execution",
        "design_backend_claim_ceiling": _get_backend_claim_ceiling(d.backend_id),
        "dataset_claim_ceiling": "lightweight_scientific_execution",
        "execution_fidelity_claim_ceiling": "lightweight_scientific_execution",
        "synthetic_data": True,
        "physical_backend": False,
        "native_backend": False,
        "full_wave_optics": False,
        "phase_to_fft_proxy_used": True,
    }


def _execute_remote_design(d: Any, worker_id: str) -> dict[str, Any]:
    from optiresearch.skills.runtime_v2 import SkillRuntimeV2
    skill_result = SkillRuntimeV2().execute_skill(
        "remote_execution",
        {
            "allow_remote": True,
            "worker_id": worker_id,
            "design_id": d.design_id,
            "handler_id": getattr(d, "handler_id", ""),
            "spec_payload": d.spec_payload if hasattr(d, "spec_payload") else {},
        },
    )
    output = skill_result.output or {}
    remote_handler_result = output.get("remote_handler_result", {}) if isinstance(output, dict) else {}
    if output.get("status") == "succeeded" and output.get("remote_validation_passed") is True:
        return {
            "status": "completed",
            "outcome": "remote_execution_completed",
            "design_id": d.design_id,
            "task_type": d.task_type if hasattr(d, "task_type") else "",
            "backend_id": d.backend_id if hasattr(d, "backend_id") else "",
            "evidence_level": output.get("evidence_level", "native_lens_simulation"),
            "handler_id": getattr(d, "handler_id", "") or _handler_id_for_design(d),
            "execution_target": "remote_wsl",
            "remote_worker_id": worker_id,
            "remote_job_id": output.get("remote_job_id", ""),
            "remote_validation_passed": output.get("remote_validation_passed", False),
            "execution_fidelity": output.get("execution_fidelity", ""),
            "proxy_fallback_used": output.get("proxy_fallback_used", False),
            "deeplens_native_psf_path": output.get("deeplens_native_psf_path", ""),
            "full_wave_optics": output.get("full_wave_optics", False),
            "phase_to_fft_proxy_used": output.get("phase_to_fft_proxy_used", False),
            "metrics": output.get("metrics", {}),
            "artifacts": output.get("artifacts", []),
            "artifact_return_path": output.get("artifact_return_path", ""),
            "artifact_ids": output.get("artifact_ids", []),
            "evidence_artifact_ids": output.get("evidence_artifact_ids", []),
            "artifact_manifest_path": output.get("artifact_manifest_path", ""),
            "artifact_manifest_complete": output.get("artifact_manifest_complete", False),
            "artifact_ingestion_status": output.get("artifact_ingestion_status", ""),
            "sha256_verified": output.get("sha256_verified", False),
            "errors": [],
            "caveats": output.get("caveats", ["Remote WSL execution — validation performed on remote worker"]),
            "run_id": output.get("run_id", ""),
            "remote_handler_result": remote_handler_result,
        }
    return {
        "status": "failed",
        "outcome": "remote_execution_failed",
        "design_id": d.design_id,
        "handler_id": getattr(d, "handler_id", "") or _handler_id_for_design(d),
        "task_type": d.task_type if hasattr(d, "task_type") else "",
        "backend_id": d.backend_id if hasattr(d, "backend_id") else "",
        "evidence_level": output.get("evidence_level", "needs_followup"),
        "execution_target": "remote_wsl",
        "remote_worker_id": worker_id,
        "remote_job_id": output.get("remote_job_id", ""),
        "remote_validation_passed": False,
        "execution_fidelity": output.get("execution_fidelity", ""),
        "proxy_fallback_used": output.get("proxy_fallback_used", False),
        "deeplens_native_psf_path": output.get("deeplens_native_psf_path", ""),
        "full_wave_optics": output.get("full_wave_optics", False),
        "phase_to_fft_proxy_used": output.get("phase_to_fft_proxy_used", False),
        "metrics": {},
        "artifacts": output.get("artifacts", []),
        "artifact_return_path": output.get("artifact_return_path", ""),
        "artifact_ids": output.get("artifact_ids", []),
        "evidence_artifact_ids": output.get("evidence_artifact_ids", []),
        "artifact_manifest_path": output.get("artifact_manifest_path", ""),
        "artifact_manifest_complete": output.get("artifact_manifest_complete", False),
        "artifact_ingestion_status": output.get("artifact_ingestion_status", ""),
        "sha256_verified": output.get("sha256_verified", False),
        "errors": output.get("errors") or [{"type": "REMOTE_SKILL_FAILED", "message": "; ".join(skill_result.errors)}],
        "caveats": output.get("caveats", ["Remote execution did not complete"]),
        "run_id": output.get("run_id", ""),
        "remote_handler_result": remote_handler_result,
    }


def _handler_id_for_design(d: Any) -> str:
    hid = getattr(d, "handler_id", "")
    if hid:
        return hid
    try:
        from optiresearch.skills.handler_capability_registry import (
            get_handler_capability_registry,
        )
        cap = get_handler_capability_registry().find_by_design_id(d.design_id)
        if cap:
            return cap.handler_id
    except Exception:
        pass
    return ""


def _get_handler_cap(handler_id: str) -> Any:
    try:
        from optiresearch.skills.handler_capability_registry import (
            get_handler_capability_registry,
        )
        return get_handler_capability_registry().get(handler_id)
    except Exception:
        return None


def _get_backend_claim_ceiling(backend_id: str) -> str:
    try:
        from optiresearch.backends.registry import get_backend
        backend = get_backend(backend_id)
        if backend:
            return backend.claim_ceiling
    except Exception:
        pass
    return "unsupported"


def _controller_result_to_execution_result(d: ExperimentDesignCandidate, result: Any) -> dict[str, Any]:
    completed = result.status == "succeeded"
    unsupported = result.status in ("unsupported", "skipped")
    payload = result.result_payload or {}
    errors = list(result.errors or [])
    if result.error_code and not errors:
        errors.append({"type": result.error_code, "message": result.error_message or result.error_code})
    metrics = _extract_metrics(payload)
    if not metrics and hasattr(result, "status"):
        metrics = {"controller_status": result.status}
    evidence_level = "local_execution_completed" if completed else "structured_unsupported"
    return {
        "status": "completed" if completed else ("unsupported" if unsupported else "failed"),
        "outcome": "local_execution_completed" if completed else "structured_unsupported",
        "design_id": d.design_id,
        "task_type": d.task_type,
        "backend_id": d.backend_id,
        "evidence_level": evidence_level,
        "backend_evidence_level": result.evidence_level,
        "metrics": metrics,
        "artifacts": list(result.artifact_paths or []),
        "errors": errors,
        "caveats": list(payload.get("caveats", [])) if isinstance(payload.get("caveats", []), list) else [],
        "run_id": result.run_id,
        "error_code": result.error_code,
        "error_message": result.error_message,
    }


def _unsupported_execution_result(
    d: ExperimentDesignCandidate,
    error_code: str,
    message: str,
) -> dict[str, Any]:
    return {
        "status": "unsupported",
        "outcome": "structured_unsupported",
        "design_id": d.design_id,
        "task_type": d.task_type,
        "backend_id": d.backend_id,
        "evidence_level": "structured_unsupported",
        "metrics": {},
        "artifacts": [],
        "errors": [{"type": error_code, "message": message}],
        "caveats": [message],
        "error_code": error_code,
        "error_message": message,
    }


def _needs_followup_execution_result(
    d: ExperimentDesignCandidate,
    error_code: str,
    message: str,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "needs_followup",
        "outcome": "structured_unsupported",
        "design_id": d.design_id,
        "task_type": "backend_probe",
        "backend_id": d.backend_id,
        "evidence_level": "structured_unsupported",
        "metrics": metrics or {},
        "artifacts": [],
        "errors": [{"type": error_code, "message": message}],
        "caveats": [message],
        "error_code": error_code,
        "error_message": message,
    }


def _failed_execution_result(
    d: ExperimentDesignCandidate,
    error_code: str,
    message: str,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "outcome": "structured_unsupported",
        "design_id": d.design_id,
        "task_type": d.task_type,
        "backend_id": d.backend_id,
        "evidence_level": "structured_unsupported",
        "metrics": {},
        "artifacts": [],
        "errors": [{"type": error_code, "message": message}],
        "caveats": [message],
        "error_code": error_code,
        "error_message": message,
    }


def _run_claim_gate(d: ExperimentDesignCandidate | None, result: dict[str, Any]) -> dict[str, Any]:
    try:
        from optiresearch.memory.claim_gate_v2 import ClaimGateV2
        gate = ClaimGateV2()
        if result.get("evidence_level") == "component_surrogate_hsi_codesign":
            claim_text = f"Component surrogate HSI co-design completed for {result.get('design_id', 'design')}"
        elif result.get("evidence_level") == "lightweight_scientific_execution":
            claim_text = f"Lightweight scientific HSI co-design completed with MSE-only objective for {result.get('design_id', 'design')}"
        elif result.get("evidence_level") == "report_only":
            claim_text = "The negative result is documented"
        elif result.get("status") in ("unsupported", "needs_followup"):
            claim_text = f"Boundary detected for local execution of {result.get('design_id', 'design')}"
        elif result.get("execution_target") == "remote_wsl":
            claim_text = f"Remote native lens simulation completed for {result.get('design_id', 'design')}"
        else:
            claim_text = f"Local native lens simulation completed for {result.get('design_id', 'design')}"
        backend_id = result.get("backend_id") or (d.backend_id if d else "")
        handler_id = result.get("handler_id") or (d.handler_id if d else "")
        decision = gate.check_claim(
            claim_text,
            backend_id,
            experiment_result=result,
            evidence_scope={"execution_target": result.get("execution_target", "local")},
            handler_id=handler_id,
        )
        return {
            "design_id": result.get("design_id") or (d.design_id if d else ""),
            "decision": decision.decision,
            "max_allowed_claim": decision.max_allowed_claim,
            "violation_type": decision.violation_type,
            "violation_reason": decision.violation_reason,
            "safe_wording": decision.safe_wording,
            "required_additional_evidence": decision.required_additional_evidence,
            "applicable_caveats": decision.applicable_caveats,
            "metadata": decision.metadata,
            "final_claim_ceiling": decision.final_claim_ceiling,
            "ceiling_source": decision.ceiling_source,
            "limiting_factor": decision.limiting_factor,
            "downgrade_reasons": decision.downgrade_reasons,
            "evidence_artifact_ids": result.get("evidence_artifact_ids", []),
            "evidence_completeness": "complete" if result.get("artifact_manifest_complete") else "partial",
            "missing_evidence_artifacts": result.get("missing_required_artifacts", []),
        }
    except Exception as e:
        return {"design_id": result.get("design_id", ""), "error": str(e)}


def _build_attempt_sequence(
    selected_designs: list[ExperimentDesignCandidate],
    all_designs: list[ExperimentDesignCandidate],
    max_attempts: int,
) -> list[ExperimentDesignCandidate]:
    sequence: list[ExperimentDesignCandidate] = []
    seen: set[str] = set()
    for design in selected_designs:
        if design.design_id not in seen:
            sequence.append(design)
            seen.add(design.design_id)
    # Insert lightweight scientific designs before report fallback
    for design in all_designs:
        if design.design_id not in seen and _is_lightweight_scientific_design(design):
            sequence.append(design)
            seen.add(design.design_id)
    report = _find_design(all_designs, "report_negative_result_doc")
    if report is not None and report.design_id not in seen:
        sequence.append(report)
        seen.add(report.design_id)
    for design in all_designs:
        if len(sequence) >= max_attempts:
            break
        if design.design_id not in seen and design.estimated_runtime_sec > 0:
            sequence.append(design)
            seen.add(design.design_id)
    return sequence[:max_attempts]


def _find_design(
    designs: list[ExperimentDesignCandidate],
    design_id: str | None,
) -> ExperimentDesignCandidate | None:
    if not design_id:
        return None
    for design in designs:
        if design.design_id == design_id:
            return design
    return None


def _attempt_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "design_id": result.get("design_id", ""),
        "status": result.get("status", ""),
        "evidence_level": result.get("evidence_level", ""),
        "errors": result.get("errors", []),
    }
    for key in ("handler_id", "execution_target", "remote_worker_id", "remote_job_id", "remote_validation_passed", "metrics"):
        if key in result:
            summary[key] = result.get(key)
    return summary


def _skill_id_for_design(d: ExperimentDesignCandidate) -> str:
    if "report_generation" in d.required_skills or d.design_id == "report_negative_result_doc":
        return "report_generation"
    if d.design_id == "param_reduction_sweep" or d.spec_payload.get("param_subset"):
        return "param_reduction_sweep"
    if "component_surrogate" in d.design_id or d.task_type == "component_surrogate_hsi_codesign":
        return "component_surrogate_hsi_codesign"
    if _is_lightweight_scientific_design(d):
        return "lightweight_scientific_hsi_mse_only"
    if d.design_id == "backend_switch_waveoptics_coherent":
        return "backend_probe"
    if d.required_skills:
        return d.required_skills[0]
    return "experiment_controller_v2"


def _extract_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in (
        "reconstruction_loss_before",
        "reconstruction_loss_after",
        "best_reconstruction_loss",
        "mse_before",
        "mse_after",
        "psnr_before",
        "psnr_after",
        "accepted_update_count",
        "rejected_update_count",
        "rollback_count",
        "optical_gradient_norm_max",
        "optical_gradient_norm_mean",
        "optical_parameters_changed",
        "stable_training_succeeded",
        "requires_grad",
        "report_generated",
        "improvement_detected",
        "metrics_valid",
        "synthetic_data",
        "physical_backend",
        "mse_only_objective",
    ):
        if key in payload:
            metrics[key] = payload[key]
    return metrics


def _deeplens_available() -> bool:
    try:
        return importlib.util.find_spec("deeplens") is not None
    except Exception:
        return False


def _result_status(
    mode: str,
    stop_reason: str | None,
    execution_result: dict[str, Any],
    errors: list[str],
) -> str:
    if mode == "dry_run":
        return "dry_run_only"
    if stop_reason and not execution_result:
        return "stopped"
    if execution_result.get("status") == "stopped":
        return "stopped"
    if errors and execution_result.get("status") == "failed":
        return "failed"
    if execution_result.get("status") == "failed":
        return "failed"
    return "completed"


def _ensure_remote_validation_design(
    designs: list[ExperimentDesignCandidate],
) -> list[ExperimentDesignCandidate]:
    if any(d.design_id == "remote_native_geolens_validation" for d in designs):
        return designs
    cap = _get_handler_cap("remote_native_geolens_validation")
    if not cap or not cap.enabled:
        return designs
    remote_design = ExperimentDesignCandidate(
        design_id="remote_native_geolens_validation",
        objective="Validate native GeoLens geometric HSI path on WSL",
        backend_id="deeplens_geolens_geometric",
        task_type="native_lens_simulation_codesign",
        spec_payload={
            "candidate": "auto:cooke",
            "dataset": "synthetic",
            "reconstructor": "differentiable_linear",
            "max_steps": 5,
            "optical_lr": 1e-6,
            "device": "cpu",
        },
        expected_evidence_level=cap.actual_evidence_level,
        expected_failure_modes=["remote_execution_failure"],
        required_skills=["remote_execution"],
        claim_ceiling=cap.remote_evidence_ceiling or cap.max_claim_ceiling,
        estimated_runtime_sec=cap.default_timeout_sec,
        risk_level=cap.risk_level,
        handler_id=cap.handler_id,
        actual_handler_evidence_level=cap.actual_evidence_level,
        evidence_alignment_status="aligned",
    )
    return [remote_design, *designs]


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


def _final_recommendation(
    scores: list,
    executable: list,
    mode: str,
    execution_result: dict[str, Any] | None = None,
    fallback_to_report_only: bool = False,
) -> str:
    if mode == "dry_run":
        return f"Dry run complete. Top design: {executable[0].design_id if executable else 'none'}. Set --mode local to execute."
    execution_result = execution_result or {}
    if fallback_to_report_only:
        return "Report-only fallback completed. Do not upgrade optical improvement claims."
    if execution_result.get("evidence_level") == "lightweight_scientific_execution":
        metrics = execution_result.get("metrics", {})
        improvement = "yes" if metrics.get("improvement_detected") else "no"
        mse_after = metrics.get("mse_after", "?")
        return (
            f"Lightweight scientific execution completed for "
            f"{execution_result.get('design_id', 'selected design')}. "
            f"MSE after: {mse_after:.6f}, improvement detected: {improvement}."
            if isinstance(mse_after, (int, float)) else
            f"Lightweight scientific execution completed for "
            f"{execution_result.get('design_id', 'selected design')}. "
            f"Improvement detected: {improvement}."
        )
    if execution_result.get("status") == "completed":
        return f"Local execution completed for {execution_result.get('design_id', 'selected design')}."
    if execution_result.get("status") in ("unsupported", "needs_followup"):
        return "Local execution found a structured boundary; choose another executable design or collect the missing dependency."
    if scores:
        top = scores[0]
        return f"Top-ranked design: {top.design_id} (score={top.total_score:.3f}). Recommendation: {top.recommendation}."
    return "No designs evaluated."
