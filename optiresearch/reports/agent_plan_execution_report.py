"""Agent Plan Execution Report for Phase 37."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_agent_plan_execution_report(execution_id: str) -> Path:
    run_dir = Path("workspace/agent_plan_executions") / execution_id
    result = _read_json(run_dir / "execution_result.json", {})
    path = run_dir / "plan_execution_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_markdown(execution_id, result), encoding="utf-8")
    return path


def _markdown(execution_id: str, r: dict[str, Any]) -> str:
    lines = [
        "# Agent Plan Execution Report",
        "",
        f"**Execution ID:** `{execution_id}`",
        f"**Objective:** {r.get('objective', '-')}",
        f"**Status:** {r.get('status', '-')}",
        f"**Mode:** {r.get('mode', '-')} ({r.get('executed_or_dry_run', '-')})",
        "",
        "## 1. Seed Evidence / Failure",
        f"- **Classified Failure:** {r.get('classified_failure', '-')}",
        f"- **Category:** {r.get('failure_category', '-')}",
        "",
        "## 2. Candidate Strategies",
        f"- **Count:** {r.get('candidate_strategies_count', 0)}",
    ]
    for s in r.get("candidate_strategies", []):
        lines.append(f"- [{s.get('strategy_type', '?')}] {s.get('strategy_id', '?')}")

    lines.extend([
        "",
        "## 3. Generated Experiment Designs",
        f"- **Count:** {r.get('candidate_designs_count', 0)}",
    ])
    for d in r.get("candidate_designs", []):
        lines.append(f"- {d.get('design_id', '?')}: {d.get('backend_id', '?')} {d.get('task_type', '?')}")

    lines.extend(["", "## 4. Plan Scores"])
    for s in r.get("plan_scores", []):
        lines.append(f"- {s.get('design_id', '?')}: score={s.get('total_score', '-')} → {s.get('recommendation', '-')}")

    lines.extend([
        "",
        "## 5. Executable Selection",
        f"- **Selected Design:** {r.get('selected_design') or '-'}",
        f"- **Selected Rank:** {r.get('selected_design_rank') or '-'}",
        f"- **Reason:** {r.get('executable_selection_reason') or '-'}",
        "",
        "### Selected Design Details",
    ])
    for d in r.get("selected_designs", []):
        lines.append(f"- {d.get('design_id', '?')}: {d.get('spec_payload', {})}")
    if not r.get("selected_designs"):
        lines.append("(none)")

    lines.extend([
        "",
        "### Skipped Higher-Ranked Designs",
        "| Rank | Design | Reason | Recommendation |",
        "|---|---|---|---|",
    ])
    skipped = r.get("skipped_higher_ranked_designs", [])
    if skipped:
        for item in skipped:
            lines.append(
                f"| {item.get('rank', '-')} | {item.get('design_id', '-')} | "
                f"{item.get('skipped_reason', '-')} | {item.get('recommendation', '-')} |"
            )
    else:
        lines.append("| - | none | - | - |")

    lines.extend([
        "",
        "## 6. Attempted Designs",
        "| Design | Status | Evidence | Errors |",
        "|---|---|---|---|",
    ])
    attempts = r.get("attempted_designs", [])
    if attempts:
        for attempt in attempts:
            lines.append(
                f"| {attempt.get('design_id', '-')} | {attempt.get('status', '-')} | "
                f"{attempt.get('evidence_level', '-')} | {len(attempt.get('errors', []))} |"
            )
    else:
        lines.append("| - | none | - | - |")

    lines.extend([
        "",
        "## 7. Local Execution Result",
    ])
    ex = r.get("execution_result") or {}
    if ex:
        lines.extend([
            f"- **Status:** {ex.get('status', '-')}",
            f"- **Design:** {ex.get('design_id', '-')}",
            f"- **Task:** {ex.get('task_type', '-')}",
            f"- **Backend:** {ex.get('backend_id', '-')}",
            f"- **Evidence:** {ex.get('evidence_level', '-')}",
            f"- **Metrics:** `{json.dumps(ex.get('metrics', {}), ensure_ascii=False, default=str)}`",
            f"- **Artifacts:** {', '.join(ex.get('artifacts', [])) if ex.get('artifacts') else '(none)'}",
            f"- **Errors:** `{json.dumps(ex.get('errors', []), ensure_ascii=False, default=str)}`",
        ])
    else:
        lines.append("(no local execution result)")

    if ex.get("evidence_level") == "lightweight_scientific_execution":
        m = ex.get("metrics", {})
        lines.extend([
            "",
            "## 7a. Scientific Execution Metrics",
            f"- **Evidence Level:** {ex.get('evidence_level', '-')}",
            f"- **Reconstruction Loss Before:** {m.get('reconstruction_loss_before', '-')}",
            f"- **Reconstruction Loss After:** {m.get('reconstruction_loss_after', '-')}",
            f"- **Best Reconstruction Loss:** {m.get('best_reconstruction_loss', '-')}",
            f"- **MSE Before:** {m.get('mse_before', '-')}",
            f"- **MSE After:** {m.get('mse_after', '-')}",
            f"- **PSNR Before:** {m.get('psnr_before', '-')}",
            f"- **PSNR After:** {m.get('psnr_after', '-')}",
            f"- **Improvement Detected:** {m.get('improvement_detected', '-')}",
            f"- **Metrics Valid:** {m.get('metrics_valid', '-')}",
            f"- **Execution Time (s):** {m.get('execution_time_sec', '-')}",
            f"- **Synthetic Data:** {m.get('synthetic_data', '-')}",
            f"- **Physical Backend:** {m.get('physical_backend', '-')}",
            f"- **MSE-only Objective:** {m.get('mse_only_objective', '-')}",
            "",
            "### Caveats",
            "- MSE-only synthetic HSI experiment — not native DeepLens simulation",
            "- Synthetic HSI data — real HSI performance may differ",
            "- Lightweight scientific execution — claim ceiling: synthetic_lightweight_metric_experiment",
        ])

    lines.extend(["", "## 8. ClaimGate Outcome"])
    claim = r.get("claim_gate_decision") or {}
    if claim:
        lines.extend([
            f"- **Decision:** {claim.get('decision', '-')}",
            f"- **Max Allowed Claim:** {claim.get('max_allowed_claim', '-')}",
            f"- **Violation:** {claim.get('violation_type') or 'none'}",
            f"- **Safe Wording:** {claim.get('safe_wording', '-')}",
        ])
    else:
        lines.append("(none)")

    lines.extend([
        "",
        "## 9. Events",
        f"- **Event count:** {r.get('event_count', 0)}",
        f"- **Event log:** {r.get('event_log_path', '-')}",
        "",
        "## 10. Memory / State Updates",
        f"- **Memory Updated:** {r.get('memory_updated', False)}",
        *([f"- **Memory Entry:** {m}" for m in r.get("memory_updates", [])] or ["- **Memory Entry:** (none)"]),
        f"- **Snapshots:** {r.get('state_snapshots_count', 0)}",
        *([f"- **Snapshot:** {s}" for s in r.get("state_snapshot_refs", [])] or []),
        "",
        "## 11. Final Recommendation",
        r.get("final_recommendation", "-"),
        "",
        "## 12. Errors",
        *([f"- {e}" for e in r.get("errors", [])] or ["(none)"]),
    ])
    return "\n".join(lines) + "\n"


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
