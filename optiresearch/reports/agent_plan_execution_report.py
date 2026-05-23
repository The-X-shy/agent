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

    lines.extend([
        "",
        "## 4. Plan Scores",
    ])
    for s in r.get("plan_scores", []):
        lines.append(f"- {s.get('design_id', '?')}: score={s.get('total_score', '-')} → {s.get('recommendation', '-')}")

    lines.extend([
        "",
        "## 5. Selected Design",
    ])
    for d in r.get("selected_designs", []):
        lines.append(f"- {d.get('design_id', '?')}: {d.get('spec_payload', {})}")

    lines.extend([
        "",
        "## 6. Execution Results",
    ])
    for ex in r.get("execution_results", []):
        lines.append(f"- {ex}")
    if not r.get("execution_results"):
        lines.append("(no execution — dry run)")

    lines.extend([
        "",
        "## 7. ClaimGate Decisions",
    ])
    for c in r.get("claim_gate_decisions", []):
        lines.append(f"- {c.get('decision', '?')}: {c.get('violation_type', 'none')}")

    lines.extend([
        "",
        "## 8. Events",
        f"- **Event count:** {r.get('event_count', 0)}",
        f"- **Event log:** {r.get('event_log_path', '-')}",
        "",
        "## 9. StateStore",
        f"- **Snapshots:** {r.get('state_snapshots_count', 0)}",
        "",
        "## 10. Memory Updates",
        *([f"- {m}" for m in r.get("memory_updates", [])] or ["(none)"]),
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
