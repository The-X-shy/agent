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

    # Claim evolution table
    sections.append(_build_claim_evolution(result))

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


def _format_list(items: list[str], fallback: str) -> str:
    if not items:
        return fallback
    return "\n".join(f"- {item}" for item in items)
