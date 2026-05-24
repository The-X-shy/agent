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
    evaluator = CandidatePlanEvaluator()
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


def _execute_design(d: ExperimentDesignCandidate) -> dict[str, Any]:
    if "report_generation" in d.required_skills or d.design_id == "report_negative_result_doc":
        return _execute_report_design(d)

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
        if result.get("evidence_level") == "lightweight_scientific_execution":
            claim_text = f"Lightweight scientific HSI co-design completed with MSE-only objective for {result.get('design_id', 'design')}"
        elif result.get("evidence_level") == "report_only":
            claim_text = "The negative result is documented"
        elif result.get("status") in ("unsupported", "needs_followup"):
            claim_text = f"Boundary detected for local execution of {result.get('design_id', 'design')}"
        else:
            claim_text = f"Local native lens simulation completed for {result.get('design_id', 'design')}"
        backend_id = result.get("backend_id") or (d.backend_id if d else "")
        handler_id = result.get("handler_id") or (d.handler_id if d else "")
        decision = gate.check_claim(
            claim_text,
            backend_id,
            experiment_result=result,
            evidence_scope={"execution_target": "local"},
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
    return {
        "design_id": result.get("design_id", ""),
        "status": result.get("status", ""),
        "evidence_level": result.get("evidence_level", ""),
        "errors": result.get("errors", []),
    }


def _skill_id_for_design(d: ExperimentDesignCandidate) -> str:
    if "report_generation" in d.required_skills or d.design_id == "report_negative_result_doc":
        return "report_generation"
    if d.design_id == "param_reduction_sweep" or d.spec_payload.get("param_subset"):
        return "param_reduction_sweep"
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
    if errors and execution_result.get("status") == "failed":
        return "failed"
    if execution_result.get("status") == "failed":
        return "failed"
    return "completed"


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
