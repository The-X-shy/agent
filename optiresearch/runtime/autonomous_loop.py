"""Autonomous research loop runtime.

Implements the plan -> execute -> evaluate -> revise -> repeat cycle
driven by LLM (or rule fallback when LLM is unavailable).

Key constraints:
- LLM must NOT execute shell commands.
- LLM must NOT bypass ClaimEvidence.
- Mock/proxy results must NOT be presented as real optical validation.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from optiresearch.memory.claim_evidence import ClaimEvidenceManager
from optiresearch.memory.meta_trace import MetaTrace, MetaTraceWriter
from optiresearch.memory.schemas import StrictModel, make_deterministic_id
from optiresearch.runtime.hsi_pipeline import run_hsi_reconstruction_flow
from optiresearch.schemas.autonomous import (
    AutonomousLoopConfig,
    AutonomousLoopSummary,
    ResearchIterationPlan,
    ResearchIterationResult,
    ReviewerOutput,
)
from optiresearch.llm.registry import get_llm_provider
from optiresearch.llm.base import LLMProviderError


def _load_prompt(name: str) -> str:
    prompt_path = Path(__file__).parent.parent / "llm" / "prompts" / name
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


def _fill_template(template: str, variables: dict[str, str]) -> str:
    result = template
    for key, value in variables.items():
        result = result.replace("{{ " + key + " }}", value)
        result = result.replace("{{" + key + "}}", value)
    return result


def run_autonomous_research_loop(config: AutonomousLoopConfig, remote_executor: Any | None = None) -> AutonomousLoopSummary:
    loop_id = make_deterministic_id("aloop", config.objective, str(time.time()))
    output_dir = Path("workspace/autonomous_loops") / loop_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    (output_dir / "loop_config.json").write_text(
        config.model_dump_json(indent=2), encoding="utf-8"
    )

    # Initialize services
    claim_manager = ClaimEvidenceManager(workspace_id=loop_id)
    trace_writer = MetaTraceWriter()
    provider = get_llm_provider(config.llm_provider)
    llm_available = provider.available()

    execution_mode = str(config.metadata.get("execution_mode", "local"))
    worker_id = str(config.metadata.get("worker_id", ""))

    # Run baseline (conventional encoder) for comparison. Remote mode keeps
    # baseline conservative unless a remote executor is explicitly supplied.
    try:
        if execution_mode == "remote" and remote_executor is not None and worker_id:
            baseline_result = _run_remote_plan(
                config=config,
                worker_id=worker_id,
                encoder="conventional",
                reconstructor=config.allowed_reconstructors[0] if config.allowed_reconstructors else "optical_conditioned_linear",
                remote_executor=remote_executor,
            )
        elif execution_mode == "remote":
            baseline_result = {"metrics": {}}
        else:
            baseline_result = _run_baseline(config, loop_id)
    except Exception:
        baseline_result = {"metrics": {}}
    baseline_metrics = baseline_result.get("metrics", {})

    iterations: list[ResearchIterationResult] = []
    current_encoder = config.allowed_encoders[-1] if config.allowed_encoders else "controlled_chromatic_edof"
    current_reconstructor = config.allowed_reconstructors[0] if config.allowed_reconstructors else "optical_conditioned_linear"
    current_forward_mode = config.allowed_forward_modes[0] if config.allowed_forward_modes else "depth_spectral_coded"

    stopped_reason = ""
    best_metrics: dict[str, Any] = {}
    best_iteration = -1
    best_score = -1.0

    for iteration in range(1, config.max_iterations + 1):
        # Phase 1: Plan
        plan = _propose_plan(
            config, provider, llm_available, iteration,
            current_encoder, current_reconstructor, current_forward_mode,
            iterations, baseline_metrics,
        )

        # Save plan
        (output_dir / f"iteration_{iteration:03d}_plan.json").write_text(
            plan.model_dump_json(indent=2), encoding="utf-8"
        )

        # Write plan trace
        trace_writer.write_trace(MetaTrace(
            trace_id=make_deterministic_id("trace", loop_id, f"plan_{iteration}"),
            workspace_id=loop_id,
            run_id=loop_id,
            branch_id=None,
            step_id=None,
            phase="Explore",
            actor="System",
            task=f"Propose iteration {iteration} plan",
            skill_id=None,
            skill_version=None,
            tool=None,
            next_action=None,
            status="succeeded",
            timestamp_start=None,
            timestamp_end=None,
            content_hash=None,
            metadata={
                "llm_used": llm_available,
                "llm_provider": config.llm_provider,
                "selected_encoder": plan.selected_encoder,
                "selected_reconstructor": plan.selected_reconstructor,
                "hypothesis": plan.hypothesis,
                "evidence_level": "mock" if config.backend == "mock_deeplens" else "deeplens_adapter_proxy",
            },
        ))

        # Phase 2: Validate plan
        validation_error = _validate_plan(plan, config)
        if validation_error:
            result = ResearchIterationResult(
                iteration_id=iteration,
                status="validation_rejected",
                error_message=validation_error,
                next_recommendation=f"Plan rejected: {validation_error}",
            )
            iterations.append(result)
            (output_dir / f"iteration_{iteration:03d}_result.json").write_text(
                result.model_dump_json(indent=2), encoding="utf-8"
            )
            continue

        # Phase 3: Execute plan
        try:
            if execution_mode == "remote":
                if not worker_id:
                    raise ValueError("remote execution requires config.metadata['worker_id']")
                execution_result = _run_remote_plan(
                    config=config,
                    worker_id=worker_id,
                    encoder=plan.selected_encoder,
                    reconstructor=plan.selected_reconstructor,
                    remote_executor=remote_executor,
                )
            else:
                execution_result = run_hsi_reconstruction_flow(
                    objective=config.objective,
                    backend=plan.selected_backend,
                    encoder_type=plan.selected_encoder,
                    workspace_id=loop_id,
                    use_llm=False,
                    forward_mode=plan.selected_forward_mode,
                    reconstructor_type=plan.selected_reconstructor,
                    dataset=config.dataset,
                    dataset_pattern="mixed_materials",
                )
        except Exception as exc:
            result = ResearchIterationResult(
                iteration_id=iteration,
                status="failed",
                error_message=str(exc),
                next_recommendation=f"Execution failed: {exc}",
            )
            iterations.append(result)
            (output_dir / f"iteration_{iteration:03d}_result.json").write_text(
                result.model_dump_json(indent=2), encoding="utf-8"
            )
            trace_writer.write_trace(MetaTrace(
                trace_id=make_deterministic_id("trace", loop_id, f"exec_{iteration}"),
                workspace_id=loop_id,
                run_id=loop_id,
                branch_id=None,
                step_id=None,
                phase="Execute",
                actor="System",
                task=f"Execute iteration {iteration}",
                skill_id=None,
                skill_version=None,
                tool=None,
                next_action=None,
                status="failed",
                timestamp_start=None,
                timestamp_end=None,
                content_hash=None,
                metadata={"error": str(exc)},
            ))
            continue

        # Write execution trace
        exec_status = "succeeded" if execution_result.get("status") == "succeeded" else "failed"
        trace_writer.write_trace(MetaTrace(
            trace_id=make_deterministic_id("trace", loop_id, f"exec_{iteration}"),
            workspace_id=loop_id,
            run_id=loop_id,
            branch_id=None,
            step_id=None,
            phase="Execute",
            actor="System",
            task=f"Execute iteration {iteration}",
            skill_id=None,
            skill_version=None,
            tool=None,
            next_action=None,
            status=exec_status,
            timestamp_start=None,
            timestamp_end=None,
            content_hash=None,
            metadata={
                "run_id": execution_result.get("run_id", ""),
                "encoder": plan.selected_encoder,
                "reconstructor": plan.selected_reconstructor,
                "forward_mode": plan.selected_forward_mode,
                "evidence_level": execution_result.get("evidence_level", "mock"),
                "metrics": execution_result.get("metrics", {}),
            },
        ))

        # Phase 4: Evaluate
        metrics = execution_result.get("metrics", {})
        score = float(metrics.get("reconstruction_score", 0))
        improvement = score - best_score if best_score >= 0 else None

        if score > best_score:
            best_score = score
            best_iteration = iteration
            best_metrics = dict(metrics)

        # Create claim
        claim_text = f"{plan.selected_encoder} achieves reconstruction_score={score:.3f} under {plan.selected_backend} with {plan.selected_reconstructor}"
        scope = {
            "run_id": execution_result.get("run_id", ""),
            "backend": plan.selected_backend,
            "encoder_type": plan.selected_encoder,
            "evidence_domain": "hsi_reconstruction",
            "reconstructor": plan.selected_reconstructor,
        }
        claim = claim_manager.create_claim(claim_text, scope)
        if execution_result.get("artifact_ids"):
            for aid in execution_result["artifact_ids"]:
                claim_manager.attach_support(claim.claim_id, aid, min(0.85, score / 100.0))
        claim_manager.review_claim(claim.claim_id)

        result = ResearchIterationResult(
            iteration_id=iteration,
            run_id=execution_result.get("run_id", ""),
            status="succeeded" if execution_result.get("status") == "succeeded" else "failed",
            metrics=metrics,
            claims=[claim_manager.explain_claim(claim.claim_id)],
            design_rules=[],
            artifacts=execution_result.get("artifact_uris", []),
            improvement_over_baseline=improvement,
        )

        # Phase 5: Review (ask LLM to evaluate)
        review = _review_iteration(
            config, provider, llm_available, iteration, plan, result,
            iterations, baseline_metrics,
        )

        result.next_recommendation = review.recommendation_for_human

        # Write review trace
        trace_writer.write_trace(MetaTrace(
            trace_id=make_deterministic_id("trace", loop_id, f"review_{iteration}"),
            workspace_id=loop_id,
            run_id=loop_id,
            branch_id=None,
            step_id=None,
            phase="Review",
            actor="System",
            task=f"Review iteration {iteration}",
            skill_id=None,
            skill_version=None,
            tool=None,
            next_action=None,
            status="succeeded",
            timestamp_start=None,
            timestamp_end=None,
            content_hash=None,
            metadata={
                "llm_used": llm_available,
                "improvement_detected": review.improvement_detected,
                "next_action": review.next_action,
                "evidence_level": review.evidence_level,
                "caveats": review.caveats,
            },
        ))

        iterations.append(result)

        # Save result
        (output_dir / f"iteration_{iteration:03d}_result.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )

        # Decide next step
        if review.next_action == "stop":
            stopped_reason = review.stopping_reason or "Reviewer requested stop"
            break
        elif review.next_action == "change_encoder" and review.next_encoder in config.allowed_encoders:
            current_encoder = review.next_encoder
        elif review.next_action == "change_reconstructor" and review.next_reconstructor in config.allowed_reconstructors:
            current_reconstructor = review.next_reconstructor
        elif review.next_action == "change_forward_mode" and review.next_forward_mode in config.allowed_forward_modes:
            current_forward_mode = review.next_forward_mode
        # else: continue with same settings

    if not stopped_reason:
        stopped_reason = f"Max iterations ({config.max_iterations}) reached"

    # Build summary
    supported: list[str] = []
    unsupported: list[str] = []
    caveats: list[str] = [
        "Mock backend results are NOT real optical validation.",
        "Synthetic HSI is NOT real camera HSI.",
        "DeepLens adapter_proxy is NOT native validation." if config.backend in ("deeplens",) else "",
    ]
    caveats = [c for c in caveats if c]

    for it in iterations:
        for claim_data in it.claims:
            status = claim_data.get("status", "")
            claim_text = claim_data.get("claim_text", "")
            if status == "supported":
                supported.append(claim_text)
            elif status in ("unsupported", "contradicted"):
                unsupported.append(claim_text)

    improvement = best_iteration > 0 and baseline_metrics and (
        best_metrics.get("reconstruction_score", 0) > baseline_metrics.get("reconstruction_score", 0)
    )

    summary = AutonomousLoopSummary(
        objective=config.objective,
        loop_id=loop_id,
        iterations=iterations,
        total_iterations=len(iterations),
        best_iteration=best_iteration,
        best_metrics=best_metrics,
        stopped_reason=stopped_reason,
        supported_claims=supported,
        unsupported_claims=unsupported,
        caveats=caveats,
        baseline_metrics=baseline_metrics,
        improvement_achieved=improvement,
    )

    # Export summary
    (output_dir / "autonomous_loop_summary.json").write_text(
        summary.model_dump_json(indent=2), encoding="utf-8"
    )

    # Export report
    from optiresearch.reports.autonomous_loop import export_autonomous_loop_report
    export_autonomous_loop_report(summary, output_dir)

    return summary


def _run_baseline(config: AutonomousLoopConfig, loop_id: str) -> dict[str, Any]:
    """Run conventional encoder as baseline for comparison."""
    try:
        return run_hsi_reconstruction_flow(
            objective=config.objective,
            backend=config.backend,
            encoder_type="conventional",
            workspace_id=loop_id,
            use_llm=False,
            forward_mode=config.allowed_forward_modes[0] if config.allowed_forward_modes else "depth_spectral_coded",
            reconstructor_type=config.allowed_reconstructors[0] if config.allowed_reconstructors else "optical_conditioned_linear",
            dataset=config.dataset,
            dataset_pattern="mixed_materials",
        )
    except Exception:
        return {"metrics": {}}


def _run_remote_plan(
    config: AutonomousLoopConfig,
    worker_id: str,
    encoder: str,
    reconstructor: str,
    remote_executor: Any | None,
) -> dict[str, Any]:
    """Execute one autonomous-loop plan on a registered remote worker."""
    if remote_executor is None:
        from optiresearch.runtime.remote_jobs import run_remote_hsi_reconstruction

        payload = run_remote_hsi_reconstruction(
            worker_id=worker_id,
            objective=config.objective,
            backend=config.backend,
            encoder=encoder,
            reconstructor=reconstructor,
            dataset=config.dataset,
            ingest=True,
        )
        result = payload["result"]
        ingestion = payload.get("ingestion") or {}
        return {
            "status": result.status,
            "run_id": result.remote_run_id or result.job_id,
            "metrics": result.metrics_summary,
            "artifact_ids": ingestion.get("artifact_ids", []),
            "artifact_uris": ingestion.get("artifact_uris", []),
            "evidence_level": result.metrics_summary.get("evidence_level", "remote_deeplens_worker"),
        }

    return remote_executor(
        worker_id=worker_id,
        objective=config.objective,
        backend=config.backend,
        encoder=encoder,
        reconstructor=reconstructor,
        dataset=config.dataset,
    )


def _propose_plan(
    config: AutonomousLoopConfig,
    provider: Any,
    llm_available: bool,
    iteration: int,
    current_encoder: str,
    current_reconstructor: str,
    current_forward_mode: str,
    previous_iterations: list[ResearchIterationResult],
    baseline_metrics: dict[str, Any],
) -> ResearchIterationPlan:
    """Propose next iteration plan via LLM or rule fallback."""
    if llm_available:
        try:
            template = _load_prompt("autonomous_planner.md")
            prev_text = _format_previous_results(previous_iterations)
            prompt = _fill_template(template, {
                "objective": config.objective,
                "allowed_encoders": ", ".join(config.allowed_encoders),
                "allowed_reconstructors": ", ".join(config.allowed_reconstructors),
                "allowed_forward_modes": ", ".join(config.allowed_forward_modes),
                "backend": config.backend,
                "dataset": config.dataset,
                "remaining_iterations": str(config.max_iterations - iteration + 1),
                "previous_results": prev_text,
            })
            response = provider.structured_complete(
                [{"role": "user", "content": prompt}],
                ResearchIterationPlan,
            )
            if isinstance(response, ResearchIterationPlan):
                plan = response
                plan.iteration_id = iteration
                return plan
        except (LLMProviderError, Exception):
            pass

    # Rule fallback
    return _rule_based_plan(config, iteration, current_encoder, current_reconstructor, current_forward_mode)


def _rule_based_plan(
    config: AutonomousLoopConfig,
    iteration: int,
    current_encoder: str,
    current_reconstructor: str,
    current_forward_mode: str,
) -> ResearchIterationPlan:
    """Deterministic fallback when LLM is unavailable."""
    encoders = config.allowed_encoders
    reconstructors = config.allowed_reconstructors

    # Rotate through encoders, focusing on controlled_chromatic_edof last
    if iteration == 1:
        encoder = encoders[0] if encoders else "conventional"
    else:
        idx = (iteration - 1) % len(encoders)
        encoder = encoders[idx]

    # Try controlled_chromatic_edof in later iterations
    if iteration >= len(encoders) and "controlled_chromatic_edof" in encoders:
        encoder = "controlled_chromatic_edof"

    recon = reconstructors[0] if reconstructors else "optical_conditioned_linear"

    return ResearchIterationPlan(
        iteration_id=iteration,
        hypothesis=f"Rule-based exploration: {encoder} with {recon} at iteration {iteration}",
        selected_encoder=encoder,
        selected_reconstructor=recon,
        selected_forward_mode=config.allowed_forward_modes[0] if config.allowed_forward_modes else "depth_spectral_coded",
        selected_backend=config.backend,
        expected_improvement="Rule fallback: exploring encoder space systematically",
        required_skills=["hsi_reconstruction"],
        risk_notes="Rule-based plan — no LLM reasoning applied.",
        evidence_requirements=["synthetic_hsi_metrics"],
    )


def _validate_plan(plan: ResearchIterationPlan, config: AutonomousLoopConfig) -> str:
    """Validate plan against config constraints. Returns empty string if valid."""
    if plan.selected_encoder not in config.allowed_encoders:
        return f"Encoder '{plan.selected_encoder}' not in allowed list: {config.allowed_encoders}"
    if plan.selected_reconstructor not in config.allowed_reconstructors:
        return f"Reconstructor '{plan.selected_reconstructor}' not in allowed list: {config.allowed_reconstructors}"
    if plan.selected_forward_mode not in config.allowed_forward_modes:
        return f"Forward mode '{plan.selected_forward_mode}' not in allowed list: {config.allowed_forward_modes}"
    if config.backend in ("mock_deeplens",) and "native" in plan.hypothesis.lower():
        return "Plan claims native validation but backend is mock_deeplens."
    if "real camera" in plan.hypothesis.lower() and config.backend == "mock_deeplens":
        return "Plan references real camera but backend is mock_deeplens."
    return ""


def _review_iteration(
    config: AutonomousLoopConfig,
    provider: Any,
    llm_available: bool,
    iteration: int,
    plan: ResearchIterationPlan,
    result: ResearchIterationResult,
    all_iterations: list[ResearchIterationResult],
    baseline_metrics: dict[str, Any],
) -> ReviewerOutput:
    """Review iteration result via LLM or rule fallback."""
    if llm_available:
        try:
            template = _load_prompt("autonomous_reviewer.md")
            prompt = _fill_template(template, {
                "objective": config.objective,
                "iteration_id": str(iteration),
                "hypothesis": plan.hypothesis,
                "selected_encoder": plan.selected_encoder,
                "selected_reconstructor": plan.selected_reconstructor,
                "status": result.status,
                "metrics": json.dumps(result.metrics, indent=2),
                "claims": json.dumps(result.claims, indent=2),
                "all_iterations": _format_previous_results(all_iterations),
                "baseline_metrics": json.dumps(baseline_metrics, indent=2),
            })
            response = provider.structured_complete(
                [{"role": "user", "content": prompt}],
                ReviewerOutput,
            )
            if isinstance(response, ReviewerOutput):
                return response
        except (LLMProviderError, Exception):
            pass

    return _rule_based_review(config, iteration, plan, result, all_iterations, baseline_metrics)


def _rule_based_review(
    config: AutonomousLoopConfig,
    iteration: int,
    plan: ResearchIterationPlan,
    result: ResearchIterationResult,
    all_iterations: list[ResearchIterationResult],
    baseline_metrics: dict[str, Any],
) -> ReviewerOutput:
    """Deterministic review fallback."""
    metrics = result.metrics
    score = float(metrics.get("reconstruction_score", 0))
    baseline_score = float(baseline_metrics.get("reconstruction_score", 0))

    improved = score > baseline_score
    max_iter = config.max_iterations

    return ReviewerOutput(
        iteration_assessment=f"Iteration {iteration}: {plan.selected_encoder} + {plan.selected_reconstructor} = {score:.3f}",
        improvement_detected=improved,
        improvement_detail=f"Score {score:.3f} vs baseline {baseline_score:.3f}" if baseline_score > 0 else "No baseline for comparison",
        evidence_level="mock" if config.backend == "mock_deeplens" else "synthetic_hsi",
        caveats=["Rule-based review — no LLM reasoning applied.", "Mock backend — not real optical validation."],
        supported_claim="" if improved else "",
        unsupported_claim="" if improved else f"{plan.selected_encoder} did not improve over baseline.",
        next_action="stop" if iteration >= max_iter else "continue",
        next_encoder="controlled_chromatic_edof" if "controlled_chromatic_edof" in config.allowed_encoders else "",
        next_reconstructor=config.allowed_reconstructors[0] if config.allowed_reconstructors else "",
        next_forward_mode="",
        stopping_reason=f"Max iterations ({max_iter}) reached" if iteration >= max_iter else "",
        recommendation_for_human=f"Rule-based loop completed {iteration}/{max_iter} iterations. Best score: {score:.3f}. Review manually for next steps.",
    )


def _format_previous_results(iterations: list[ResearchIterationResult]) -> str:
    if not iterations:
        return "No previous iterations."
    lines = []
    for it in iterations:
        lines.append(
            f"- Iter {it.iteration_id}: status={it.status}, "
            f"metrics={json.dumps(it.metrics, sort_keys=True)}, "
            f"improvement={it.improvement_over_baseline}"
        )
    return "\n".join(lines)
