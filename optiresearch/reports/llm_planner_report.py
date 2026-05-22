"""Phase 26 LLM planner report generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_llm_planner_report(
    planner_run_id: str,
    output_dir: Path,
) -> Path:
    """Generate an LLM planner run report.

    Args:
        planner_run_id: The planner run ID to report on.
        output_dir: Directory to write the report into.

    Returns:
        Path to the generated markdown file.
    """
    path = output_dir / f"llm_planner_report_{planner_run_id}.md"
    trace_dir = Path("workspace/planner_traces") / planner_run_id

    sections: list[str] = []

    # Load trace data
    ctx = _load_json(trace_dir / "context_summary.json")
    raw = _load_json(trace_dir / "raw_response.json")
    val = _load_json(trace_dir / "validation_report.json")
    sel = _load_json(trace_dir / "selected_proposal.json")

    sections.append(f"""# LLM Planner Report

**Planner Run ID:** `{planner_run_id}`
**Trace Path:** `{trace_dir}`""")

    if ctx:
        sections.append(f"""## Objective
{ctx.get('objective', 'Unknown')}

## Execution Mode
{ctx.get('execution_mode', 'Unknown')}

## Allowed Backends
{', '.join(ctx.get('allowed_backends', []))}""")

    if val:
        lines = ["## Validation Results", ""]
        for v in val:
            status = "PASS" if v.get("valid") else "FAIL"
            lines.append(f"- **{v.get('proposal_id', '?')}:** {status}")
            for err in v.get("errors", []):
                lines.append(f"  - {err}")
        sections.append("\n".join(lines))

    if sel:
        sections.append(f"""## Selected Proposal

- **ID:** {sel.get('proposal_id', '-')}
- **Action:** {sel.get('recommended_action', '-')}
- **Backend:** {sel.get('backend_id', '-')}
- **Risk Level:** {sel.get('risk_level', '-')}
- **Hypothesis:** {sel.get('hypothesis', '-')}
- **Rationale:** {sel.get('rationale', '-')}
- **Claim:** {sel.get('proposed_claim', '-')}
- **Safe Wording:** {sel.get('safe_wording', '-')}""")

    path.write_text("\n\n".join(sections), encoding="utf-8")
    return path


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
