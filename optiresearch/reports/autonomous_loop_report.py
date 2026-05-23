"""Phase 25 autonomous research loop report generator.

Generates a markdown trajectory report from AutonomousLoopResult.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_autonomous_loop_report(
    result: "AutonomousLoopResult",
    output_dir: Path,
) -> Path:
    """Generate the autonomous research loop report.

    Args:
        result: Completed AutonomousLoopResult.
        output_dir: Directory to write the report into.

    Returns:
        Path to the generated markdown file.
    """
    path = output_dir / "autonomous_research_loop_report.md"
    sections = _build_sections(result)
    path.write_text("\n\n".join(sections), encoding="utf-8")
    return path


def _build_sections(result: Any) -> list[str]:
    sections: list[str] = []

    # Header
    sections.append(
        f"""# Autonomous Research Loop Report v2

**Loop ID:** `{result.loop_id}`
**Objective:** {result.objective}
**Status:** {result.status}
**Total Iterations:** {len(result.iterations)}
**Error:** {result.error or "None"}"""
    )

    # Iteration summary table
    table = [
        "## Iteration Summary",
        "",
        "| # | Action | Status | Risk | Claim Decision | Next |",
        "|---|---|---|---|---|---|",
    ]
    for it in result.iterations:
        action = (it.strategy_recommendation or {}).get("recommended_action", "-")
        status = (it.execution_result or {}).get("status", "-")
        risk = (it.strategy_recommendation or {}).get("risk_level", "-")
        claim = (it.claim_gate_decision or {}).get("decision", "-")
        table.append(
            f"| {it.iteration_id} | {action} | {status} | {risk} | {claim} | {it.next_action} |"
        )
    sections.append("\n".join(table))

    # Strategy decisions
    strat_lines = ["## Strategy Decisions", ""]
    for it in result.iterations:
        sr = it.strategy_recommendation or {}
        strat_lines.append(
            f"### Iteration {it.iteration_id}: {sr.get('recommended_action', '-')}"
        )
        strat_lines.append(f"- **Rationale:** {sr.get('rationale', '-')}")
        strat_lines.append(f"- **Risk Level:** {sr.get('risk_level', '-')}")
        strat_lines.append(
            f"- **Expected Claim Gain:** {sr.get('expected_claim_gain', '-')}"
        )
        cmds = sr.get("proposed_cli_commands", [])
        if cmds:
            strat_lines.append("- **Proposed Commands:**")
            for cmd in cmds:
                strat_lines.append(f"  ```\n  {cmd}\n  ```")
        strat_lines.append("")
    sections.append("\n".join(strat_lines))

    # Claim gate decisions
    claim_lines = ["## Claim Gate Decisions", ""]
    for it in result.iterations:
        cgd = it.claim_gate_decision or {}
        claim_lines.append(
            f"### Iteration {it.iteration_id}: {cgd.get('decision', '-')}"
        )
        if cgd.get("violation_type"):
            claim_lines.append(f"- **Violation:** {cgd['violation_type']}")
            claim_lines.append(f"- **Reason:** {cgd.get('violation_reason', '-')}")
            claim_lines.append(f"- **Safe Wording:** {cgd.get('safe_wording', '-')}")
        claim_lines.append("")
    sections.append("\n".join(claim_lines))

    # Metric trajectory
    trajectory = _extract_trajectory(result)
    if trajectory:
        traj_table = [
            "## Metric Trajectory",
            "",
            "| Iteration | Primary Metric |",
            "|---|---|",
        ]
        for i, val in enumerate(trajectory):
            traj_table.append(f"| {i + 1} | {val:.6f} |")
        sections.append("\n".join(traj_table))

    # LLM proposal table
    sections.append(_build_llm_proposal_table(result))

    # Fallback events table
    sections.append(_build_fallback_table(result))

    # Best iteration summary
    sections.append(_build_best_iteration(result))

    # Backend progression
    sections.append(_build_backend_progression(result))

    # Evidence level progression
    sections.append(_build_evidence_level_progression(result))

    # Backend probe results
    sections.append(_build_backend_probe_results(result))

    # Backend switch validation
    sections.append(_build_backend_switch_validation(result))

    # Post-probe continuation
    sections.append(_build_post_probe_continuation(result))

    # Alternative backend attempts
    sections.append(_build_alternative_backend_attempts(result))

    # Execution fidelity
    sections.append(_build_execution_fidelity(result))

    # Claim evolution table
    sections.append(_build_claim_evolution(result))

    # Claim downgrade events
    sections.append(_build_claim_downgrade_events(result))

    # Enhanced metric trajectory data
    sections.append(_build_metric_trajectory_data(result))

    # Stop condition diagnostics
    sections.append(_build_stop_diagnostics(result))

    # Experiment spec patch table
    sections.append(_build_spec_patch_table(result))

    # Final recommended next step
    sections.append(_build_final_recommendation(result))

    # Final claim status
    sections.append(
        f"""## Final Claim Status

### Supported
{_format_list(result.final_supported_claims, "None")}

### Unsupported
{_format_list(result.final_unsupported_claims, "None")}"""
    )

    # Stop reason
    sections.append(
        f"""## Stop Reason

{result.trajectory_report_path or "Not specified"}"""
    )

    return sections


def _extract_trajectory(result: Any) -> list[float]:
    trajectory: list[float] = []
    for it in result.iterations:
        payload = {}
        if it.execution_result:
            payload = it.execution_result.get("result_payload") or {}
        val = payload.get(
            "reconstruction_loss_after",
            payload.get("loss_after", None),
        )
        if val is not None:
            try:
                trajectory.append(float(val))
            except (ValueError, TypeError):
                trajectory.append(0.0)
    return trajectory


def _build_llm_proposal_table(result: Any) -> str:
    lines = ["## LLM Proposals", ""]
    rows = []
    for it in result.iterations:
        meta = it.strategy_recommendation.get("metadata", {})
        planner = meta.get("planner", "rule_based")
        if planner == "llm":
            rows.append(
                f"| {it.iteration_id} | {meta.get('proposal_id', '-')} | "
                f"{it.strategy_recommendation.get('recommended_action', '-')} | "
                f"{meta.get('hypothesis', '-')[:80]} | accepted |"
            )
        elif planner == "fallback":
            rows.append(
                f"| {it.iteration_id} | fallback | "
                f"{it.strategy_recommendation.get('recommended_action', '-')} | "
                f"{meta.get('fallback_reason', '-')} | fallback_used |"
            )
    if rows:
        lines.append("| Iteration | Proposal ID | Action | Hypothesis | Status |")
        lines.append("|---|---|---|---|---|")
        lines.extend(rows)
    else:
        lines.append("No LLM proposals in this loop.")
    lines.append("")
    return "\n".join(lines)


def _build_fallback_table(result: Any) -> str:
    lines = ["## Fallback Events", ""]
    rows = []
    for it in result.iterations:
        meta = it.strategy_recommendation.get("metadata", {})
        if meta.get("planner") == "fallback":
            rows.append(
                f"| {it.iteration_id} | {meta.get('fallback_reason', '-')} | "
                f"{it.strategy_recommendation.get('recommended_action', '-')} |"
            )
    if rows:
        lines.append("| Iteration | Fallback Reason | Fallback Action |")
        lines.append("|---|---|---|")
        lines.extend(rows)
    else:
        lines.append("No fallback events in this loop.")
    lines.append("")
    return "\n".join(lines)


def _build_best_iteration(result: Any) -> str:
    lines = ["## Best Iteration", ""]
    best = result.best_result or {}
    if best:
        payload = best.get("result_payload") or {}
        lines.append(f"- **Best Iteration Backend:** {best.get('backend_id', '-')}")
        lines.append(f"- **Best Status:** {best.get('status', '-')}")
        for key in ("reconstruction_loss_after", "loss_after", "mse_after"):
            val = payload.get(key)
            if val is not None:
                lines.append(f"- **Best Metric ({key}):** {val:.6f}")
                break
    else:
        lines.append("No best iteration data.")
    lines.append("")
    return "\n".join(lines)


def _build_claim_evolution(result: Any) -> str:
    lines = ["## Claim Evolution", ""]
    rows = []
    for it in result.iterations:
        cgd = it.claim_gate_decision or {}
        rows.append(
            f"| {it.iteration_id} | {cgd.get('decision', '-')} | "
            f"{cgd.get('max_allowed_claim', '-')} | "
            f"{str(cgd.get('safe_wording', '-'))[:60]} |"
        )
    if rows:
        lines.append("| Iteration | Decision | Max Allowed | Safe Wording |")
        lines.append("|---|---|---|---|")
        lines.extend(rows)
    else:
        lines.append("No claim gate decisions recorded.")
    lines.append("")
    return "\n".join(lines)


def _build_final_recommendation(result: Any) -> str:
    lines = ["## Final Recommended Next Step", ""]
    if result.iterations:
        last = result.iterations[-1]
        lines.append(f"- **Stop Reason:** {last.stop_reason or result.trajectory_report_path or 'N/A'}")
        lines.append(f"- **Last Action:** {last.strategy_recommendation.get('recommended_action', '-')}")
        lines.append(f"- **Last Status:** {last.execution_result.get('status', '-')}")
    lines.append("")
    return "\n".join(lines)


def _build_claim_downgrade_events(result: Any) -> str:
    lines = ["## Claim Downgrade Events", ""]
    rows = []
    for it in result.iterations:
        exec_result = it.execution_result or {}
        is_downgraded = exec_result.get("claim_downgraded", False)
        cgd = it.claim_gate_decision or {}
        decision = cgd.get("decision", "")
        if is_downgraded or decision in ("qualified", "unsupported"):
            rows.append(
                f"| {it.iteration_id} | {decision} | "
                f"{exec_result.get('downgraded_from', '-')} | "
                f"{exec_result.get('downgraded_to', '-')} | "
                f"{str(cgd.get('safe_wording', '-'))[:50]} |"
            )
    if rows:
        lines.append("| Iter | Decision | From | To | Safe Wording |")
        lines.append("|---|---|---|---|---|")
        lines.extend(rows)
    else:
        lines.append("No claim downgrade events in this loop.")
    lines.append("")
    return "\n".join(lines)


def _build_metric_trajectory_data(result: Any) -> str:
    lines = ["## Metric Trajectory Data", ""]
    rows = []
    for it in result.iterations:
        payload = (
            it.execution_result.get("result_payload")
            if it.execution_result else {}
        ) or {}
        loss_before = payload.get("reconstruction_loss_before", "-")
        loss_after = payload.get("reconstruction_loss_after", "-")
        mse_after = payload.get("mse_after", "-")
        psnr_after = payload.get("psnr_after", "-")
        improved = payload.get("improvement_detected", "-")
        status = it.execution_result.get("status", "-") if it.execution_result else "-"
        fmt = lambda v: f"{v:.6f}" if isinstance(v, float) else str(v)
        rows.append(
            f"| {it.iteration_id} | {fmt(loss_before)} | {fmt(loss_after)} | "
            f"{fmt(mse_after)} | {fmt(psnr_after)} | {improved} | {status} |"
        )
    if rows:
        lines.append(
            "| Iter | Loss Before | Loss After | MSE After | PSNR After "
            "| Improvement | Status |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        lines.extend(rows)
    else:
        lines.append("No metric data in this loop.")
    lines.append("")
    return "\n".join(lines)


def _build_stop_diagnostics(result: Any) -> str:
    lines = ["## Stop Condition Diagnostics", ""]
    lines.append(f"- **Total Iterations:** {len(result.iterations)}")
    last = result.iterations[-1] if result.iterations else None
    if last:
        lines.append(f"- **Stop Reason:** {last.stop_reason or result.trajectory_report_path or 'N/A'}")
        lines.append(f"- **Next Action:** {last.next_action}")
    lines.append(f"- **Trajectory Report Path:** {result.trajectory_report_path or 'N/A'}")
    lines.append("")
    return "\n".join(lines)


def _build_spec_patch_table(result: Any) -> str:
    lines = ["## Experiment Spec Patches", ""]
    rows = []
    for it in result.iterations:
        spec = it.experiment_spec or {}
        payload = spec.get("spec_payload", {})
        patch_keys = [k for k in payload if k not in ("candidate", "reconstructor")]
        if patch_keys:
            patch_summary = ", ".join(
                f"{k}={payload[k]}" for k in sorted(patch_keys)[:5]
            )
            rows.append(f"| {it.iteration_id} | {patch_summary} |")
    if rows:
        lines.append("| Iter | Applied Patches |")
        lines.append("|---|---|")
        lines.extend(rows)
    else:
        lines.append("No spec patches applied in this loop.")
    lines.append("")
    return "\n".join(lines)


def _build_backend_progression(result: Any) -> str:
    lines = ["## Backend Progression", ""]
    rows = []
    for it in result.iterations:
        exec_result = it.execution_result or {}
        bid = exec_result.get("backend_id", "-")
        evidence = exec_result.get("evidence_level", "-")
        validated = exec_result.get("backend_switch_validated", None)
        if validated is True:
            flag = "Yes"
        elif validated is False:
            flag = "No"
        else:
            flag = ""
        rows.append(f"| {it.iteration_id} | {bid} | {evidence} | {flag} |")
    if rows:
        lines.append("| Iter | Backend | Evidence Level | Validated |")
        lines.append("|---|---|---|---|")
        lines.extend(rows)
    else:
        lines.append("No backend progression data.")
    lines.append("")
    return "\n".join(lines)


def _build_evidence_level_progression(result: Any) -> str:
    lines = ["## Evidence Level Progression", ""]
    rows = []
    for it in result.iterations:
        exec_result = it.execution_result or {}
        evidence = exec_result.get("evidence_level", "-")
        cgd = it.claim_gate_decision or {}
        ceiling = cgd.get("max_allowed_claim", "-")
        rows.append(f"| {it.iteration_id} | {evidence} | {ceiling} |")
    if rows:
        lines.append("| Iter | Evidence Level | Claim Ceiling |")
        lines.append("|---|---|---|")
        lines.extend(rows)
    else:
        lines.append("No evidence level data.")
    lines.append("")
    return "\n".join(lines)


def _build_backend_probe_results(result: Any) -> str:
    lines = ["## Backend Probe Results", ""]
    rows = []
    for it in result.iterations:
        exec_result = it.execution_result or {}
        payload = exec_result.get("result_payload") or {}
        probe_status = payload.get("probe_status", "")
        if probe_status:
            rows.append(
                f"| {it.iteration_id} | "
                f"{exec_result.get('backend_id', '-')} | "
                f"{probe_status} | "
                f"{payload.get('probe_time_seconds', '-')} |"
            )
    if rows:
        lines.append("| Iter | Backend | Probe Status | Probe Time (s) |")
        lines.append("|---|---|---|---|")
        lines.extend(rows)
    else:
        lines.append("No backend probe results in this loop.")
    lines.append("")
    return "\n".join(lines)


def _build_backend_switch_validation(result: Any) -> str:
    lines = ["## Backend Switch Validation", ""]
    triggered = False
    validated = False
    probe_success = False
    for it in result.iterations:
        exec_result = it.execution_result or {}
        if exec_result.get("switched_from_backend"):
            triggered = True
        if exec_result.get("backend_switch_validated"):
            validated = True
        payload = exec_result.get("result_payload") or {}
        if payload.get("probe_status") == "succeeded":
            probe_success = True
    lines.append(f"- **Switch Triggered:** {triggered}")
    lines.append(f"- **Switch Validated:** {validated}")
    lines.append(f"- **Probe Success:** {probe_success}")
    lines.append("")
    return "\n".join(lines)


def _build_post_probe_continuation(result: Any) -> str:
    lines = ["## Post-Probe Continuation", ""]
    rows = []
    for it in result.iterations:
        exec_result = it.execution_result or {}
        payload = exec_result.get("result_payload") or {}
        continuation = exec_result.get("post_probe_continuation_required")
        if continuation is not None:
            validated = exec_result.get("validated_backend_id", "-")
            evidence = exec_result.get("validated_backend_evidence_level", "-")
            exp_status = payload.get("status", "-")
            rows.append(
                f"| {it.iteration_id} | {validated} | {evidence} | {exp_status} |"
            )
    if rows:
        lines.append("| Iter | Validated Backend | Evidence Level | Experiment Status |")
        lines.append("|---|---|---|---|")
        lines.extend(rows)
    else:
        lines.append("No post-probe continuation events in this loop.")
    lines.append("")
    return "\n".join(lines)


def _build_alternative_backend_attempts(result: Any) -> str:
    lines = ["## Alternative Backend Attempts", ""]
    rows = []
    for it in result.iterations:
        exec_result = it.execution_result or {}
        metrics = it.metrics_snapshot or {}
        alt_list = exec_result.get("alternative_backends_attempted") or metrics.get("alternative_backends_attempted") or []
        if alt_list:
            for alt in alt_list:
                rows.append(f"| {it.iteration_id} | {alt} | attempted |")
        failed = exec_result.get("failed_alternatives") or metrics.get("failed_alternatives") or []
        if failed:
            for alt in failed:
                rows.append(f"| {it.iteration_id} | {alt} | failed |")
    if rows:
        lines.append("| Iter | Backend | Status |")
        lines.append("|---|---|---|")
        lines.extend(rows)
    else:
        lines.append("No alternative backend attempts in this loop.")
    lines.append("")
    return "\n".join(lines)


def _build_execution_fidelity(result: Any) -> str:
    lines = ["## Execution Fidelity", ""]
    rows = []
    for it in result.iterations:
        exec_result = it.execution_result or {}
        payload = exec_result.get("result_payload") or {}
        bid = exec_result.get("backend_id", "-")
        fidelity = payload.get("execution_fidelity") or payload.get("actual_execution_fidelity", "")
        proxy_fallback = payload.get("proxy_fallback_used", "")
        native_psf_path = payload.get("deeplens_native_psf_path", "")
        if fidelity:
            rows.append(
                f"| {it.iteration_id} | {bid} | {fidelity} | "
                f"{proxy_fallback} | {native_psf_path} |"
            )
    if rows:
        lines.append("| Iter | Backend | Execution Fidelity | Proxy Fallback | Native PSF Path |")
        lines.append("|---|---|---|---|---|")
        lines.extend(rows)
    else:
        lines.append("No execution fidelity data in this loop.")
    lines.append("")
    return "\n".join(lines)


def _format_list(items: list[str], fallback: str) -> str:
    if not items:
        return fallback
    return "\n".join(f"- {item}" for item in items)
