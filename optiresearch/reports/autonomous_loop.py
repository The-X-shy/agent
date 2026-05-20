"""Autonomous loop evaluation report."""

from __future__ import annotations

from pathlib import Path

from optiresearch.schemas.autonomous import AutonomousLoopSummary


def export_autonomous_loop_report(
    summary: AutonomousLoopSummary,
    output_dir: Path,
) -> Path:
    path = output_dir / "autonomous_iteration_report.md"
    path.write_text(_markdown(summary), encoding="utf-8")
    return path


def _markdown(summary: AutonomousLoopSummary) -> str:
    lines = [
        "# Autonomous Research Loop Report",
        "",
        f"**Loop ID:** `{summary.loop_id}`",
        f"**Objective:** {summary.objective}",
        f"**Total iterations:** {summary.total_iterations}",
        f"**Stopped reason:** {summary.stopped_reason}",
        f"**Improvement achieved:** {summary.improvement_achieved}",
        "",
        "## 1. Objective",
        "",
        summary.objective,
        "",
        "## 2. Baseline Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key, value in summary.baseline_metrics.items():
        if isinstance(value, (int, float)):
            lines.append(f"| {key} | {value:.4f} |")
    if not summary.baseline_metrics:
        lines.append("| (no baseline) | — |")

    lines.extend([
        "",
        "## 3. Iteration Plans and Results",
        "",
        "| Iter | Encoder | Reconstructor | Status | Score | vs Baseline |",
        "|---|---:|---:|---:|---:|",
    ])
    for it in summary.iterations:
        encoder = ""
        recon = ""
        for claim_data in it.claims:
            encoder = claim_data.get("claim_text", "")
        score = it.metrics.get("reconstruction_score", "—")
        improvement = it.improvement_over_baseline
        imp_str = f"{improvement:+.4f}" if improvement is not None else "—"
        lines.append(
            f"| {it.iteration_id} | {encoder[:40]} | {recon[:20]} | {it.status} | {score} | {imp_str} |"
        )

    lines.extend([
        "",
        "## 4. Metric Trajectory",
        "",
        "| Iteration | PSNR | SSIM | SAM | ERGAS | Rec Score | Coding Str | Depth Stab | Spectral Sep |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for it in summary.iterations:
        m = it.metrics
        lines.append(
            f"| {it.iteration_id} | "
            f"{_fmt(m.get('PSNR'))} | {_fmt(m.get('SSIM'))} | {_fmt(m.get('SAM'))} | "
            f"{_fmt(m.get('ERGAS'))} | {_fmt(m.get('reconstruction_score'))} | "
            f"{_fmt(m.get('coding_strength'))} | {_fmt(m.get('depth_stability_score'))} | "
            f"{_fmt(m.get('spectral_separability_score'))} |"
        )

    lines.extend([
        "",
        "## 5. Best Result",
        "",
        f"**Best iteration:** {summary.best_iteration}",
        "",
        "| Metric | Value |",
        "|---|---|",
    ])
    for key, value in summary.best_metrics.items():
        if isinstance(value, (int, float)):
            lines.append(f"| {key} | {value:.4f} |")

    lines.extend([
        "",
        "## 6. Claims Supported",
        "",
    ])
    if summary.supported_claims:
        for c in summary.supported_claims:
            lines.append(f"- {c}")
    else:
        lines.append("(no claims supported)")

    lines.extend([
        "",
        "## 7. Claims Rejected / Unsupported",
        "",
    ])
    if summary.unsupported_claims:
        for c in summary.unsupported_claims:
            lines.append(f"- {c}")
    else:
        lines.append("(no claims rejected)")

    lines.extend([
        "",
        "## 8. Evidence Caveats",
        "",
    ])
    for c in summary.caveats:
        lines.append(f"- **{c}**")

    lines.extend([
        "",
        "## 9. Improvement Assessment",
        "",
        f"The loop {'achieved' if summary.improvement_achieved else 'did not achieve'} improvement over baseline.",
        f"Best iteration: {summary.best_iteration}.",
        "",
        "## 10. What Human Still Needs to Decide",
        "",
        "- Whether the best encoder/reconstructor combination is worth further investigation.",
        "- Whether to run additional iterations with different datasets or backends.",
        "- Whether to proceed with native DeepLens optimization (Phase 14+).",
        "- Whether the results are sufficient for paper claims or need real lab validation.",
        "",
        "## 11. Limitations",
        "",
        "- All results are synthetic/mock unless DeepLens backend was used.",
        "- LLM-driven decisions are recommendations, not validated conclusions.",
        "- The autonomous loop does NOT perform native DeepLens optimization.",
        "- ClaimEvidence remains the final gate for all claims.",
    ])
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "—"
    return str(value)
