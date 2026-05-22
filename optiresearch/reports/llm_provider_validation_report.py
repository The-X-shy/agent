"""LLM provider validation report.

Consolidates provider check, planner trace, and autonomous loop
results into a single validation report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_llm_provider_validation_report(
    planner_run_id: str,
    loop_id: str,
    output_dir: Path,
) -> Path:
    """Export a consolidated LLM provider validation report.

    Args:
        planner_run_id: Planner run ID for trace loading.
        loop_id: Autonomous loop ID for result loading.
        output_dir: Directory to write the report.

    Returns:
        Path to the generated markdown report.
    """
    provider_data = _load_provider_check()
    planner_data = _load_planner_trace(planner_run_id)
    loop_data = _load_loop_result(loop_id)

    lines = _build_report(provider_data, planner_data, loop_data, planner_run_id, loop_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "llm_provider_validation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _load_provider_check() -> dict[str, Any] | None:
    check_path = Path("workspace/reports/llm_provider_check.json")
    if not check_path.exists():
        return None
    try:
        return json.loads(check_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_planner_trace(planner_run_id: str) -> dict[str, Any] | None:
    trace_dir = Path("workspace/planner_traces") / planner_run_id
    if not trace_dir.exists():
        return None
    result: dict[str, Any] = {"run_id": planner_run_id, "files": {}}
    for f in sorted(trace_dir.iterdir()):
        if f.suffix == ".json":
            try:
                result["files"][f.name] = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                result["files"][f.name] = None
    return result


def _load_loop_result(loop_id: str) -> dict[str, Any] | None:
    loop_path = Path("workspace/autonomous_loops_v2") / loop_id / "loop_result.json"
    if not loop_path.exists():
        return None
    try:
        return json.loads(loop_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _build_report(
    provider_data: dict[str, Any] | None,
    planner_data: dict[str, Any] | None,
    loop_data: dict[str, Any] | None,
    planner_run_id: str,
    loop_id: str,
) -> list[str]:
    lines: list[str] = [
        "# LLM Provider Validation Report",
        "",
        f"**Planner Run ID:** `{planner_run_id}`",
        f"**Loop ID:** `{loop_id}`",
        "",
    ]

    # 1. Provider environment status
    lines.append("## 1. Provider Environment Status")
    lines.append("")
    if provider_data:
        lines.append(f"- **Status:** `{provider_data.get('status', '-')}`")
        lines.append(f"- **Provider:** `{provider_data.get('provider', '-')}`")
        lines.append(f"- **Model:** `{provider_data.get('model', '-')}`")
        lines.append(f"- **Base URL:** `{provider_data.get('base_url', '-')}`")
        lines.append(f"- **Error Code:** `{provider_data.get('error_code', '-')}`")
        lines.append(f"- **Latency:** {provider_data.get('latency_ms', '-')} ms")
    else:
        lines.append("No provider check data available.")
    lines.append("")

    # 2. Planner smoke result
    lines.append("## 2. Planner Smoke Result")
    lines.append("")
    if planner_data:
        validation = planner_data.get("files", {}).get("validation_report.json")
        selected = planner_data.get("files", {}).get("selected_proposal.json")
        if validation:
            lines.append(f"- **Validation entries:** {len(validation) if isinstance(validation, list) else 'N/A'}")
        if selected:
            if isinstance(selected, dict) and "fallback" in selected:
                lines.append(f"- **Fallback used:** {json.dumps(selected['fallback'])}")
            else:
                lines.append(f"- **Selected proposal:** {json.dumps(selected) if selected else 'None'}")
    else:
        lines.append("No planner trace data available.")
    lines.append("")

    # 3. Proposal validation summary
    lines.append("## 3. Proposal Validation Summary")
    lines.append("")
    if planner_data:
        validation = planner_data.get("files", {}).get("validation_report.json")
        if validation and isinstance(validation, list):
            for entry in validation:
                status = "PASS" if entry.get("valid") else "FAIL"
                lines.append(f"- **{entry.get('proposal_id', '-')}:** {status}")
                if entry.get("errors"):
                    for err in entry["errors"]:
                        lines.append(f"  - {err}")
        else:
            lines.append("No validation data available.")
    else:
        lines.append("No planner trace data available.")
    lines.append("")

    # 4. Claim gate summary
    lines.append("## 4. Claim Gate Summary")
    lines.append("")
    if planner_data:
        selected = planner_data.get("files", {}).get("selected_proposal.json")
        if selected and isinstance(selected, dict):
            safe = selected.get("safe_wording", "")
            claim = selected.get("proposed_claim", "")
            lines.append(f"- **Proposed claim:** {claim}")
            lines.append(f"- **Safe wording:** {safe if safe else '(not modified)'}")
        else:
            lines.append("No claim gate data available.")
    else:
        lines.append("No planner trace data available.")
    lines.append("")

    # 5. Autonomous loop dry run result
    lines.append("## 5. Autonomous Loop Dry Run Result")
    lines.append("")
    if loop_data:
        lines.append(f"- **Status:** `{loop_data.get('status', '-')}`")
        lines.append(f"- **Total iterations:** {loop_data.get('total_iterations', '-')}")
        iterations = loop_data.get("iterations", [])
        if iterations:
            for it in iterations:
                lines.append(f"  - {it.get('id', '-')}: action={it.get('action', '-')}, "
                           f"status={it.get('exec_status', '-')}, next={it.get('next_action', '-')}")
        lines.append(f"- **Final supported claims:** {loop_data.get('final_supported_claims', [])}")
        lines.append(f"- **Final unsupported claims:** {loop_data.get('final_unsupported_claims', [])}")
    else:
        lines.append("No loop result data available.")
    lines.append("")

    # 6. Local execution result
    lines.append("## 6. Autonomous Loop Local Result")
    lines.append("")
    if loop_data:
        lines.append(f"- **Status:** `{loop_data.get('status', '-')}`")
        lines.append(f"- **Trajectory report:** `{loop_data.get('trajectory_report_path', '-')}`")
    else:
        lines.append("No local loop result data available.")
    lines.append("")

    # 7. Fallback behavior
    lines.append("## 7. Fallback Behavior")
    lines.append("")
    if loop_data:
        iterations = loop_data.get("iterations", [])
        fallbacks = [it for it in iterations if it.get("stop_reason") == "fallback"]
        if fallbacks:
            for fb in fallbacks:
                lines.append(f"- **{fb.get('id', '-')}:** Fell back to rule-based strategy")
        else:
            lines.append("No fallback events recorded.")
    else:
        lines.append("No fallback data available.")
    lines.append("")

    # 8. Safety checks
    lines.append("## 8. Safety Checks")
    lines.append("")
    lines.append("- PlannerValidator: 10 hard checks (schema, backend, task_type, claim_ceiling, execution_mode, shell_commands, forbidden_actions, dataset_claim, waveoptics_claim)")
    lines.append("- ClaimGateV2: 8 violation types (proxy_as_waveoptics, geometric_as_coherent, synthetic_as_real, differentiable_as_improves, local_only_as_robust, rollback_protection_as_improvement, unsupported_path_as_supported, black_box_as_native)")
    lines.append("- Trace sanitization: API keys, Authorization headers, environment values redacted")
    lines.append("- Dry run execution: no real experiments run without explicit opt-in")
    lines.append("")

    # 9. Current claim boundary
    lines.append("## 9. Current Claim Boundary")
    lines.append("")
    lines.append("| Backend | Claim Ceiling |")
    lines.append("|---------|---------------|")
    lines.append("| phase_to_fft_proxy | native_full_reconstruction_proxy |")
    lines.append("| deeplens_geolens_geometric | native_lens_simulation |")
    lines.append("| deeplens_coherent_asm | native_waveoptics |")
    lines.append("| local_synthetic_hsi | synthetic_validation |")
    lines.append("")

    # 10. Remaining limitations
    lines.append("## 10. Remaining Limitations")
    lines.append("")
    lines.append("- DeepSeek provider requires DEEPSEEK_API_KEY environment variable")
    lines.append("- Real LLM tests require explicit opt-in via OPTIRESEARCH_ENABLE_REAL_LLM_TESTS=1")
    lines.append("- LLM may produce proposals that fail schema validation (caught by PlannerValidator)")
    lines.append("- LLM may propose claims exceeding backend capability (caught by ClaimGateV2)")
    lines.append("- Network errors or API rate limits may cause fallback to rule-based planning")
    lines.append("- Remote execution (remote_opt_in mode) requires explicit --allow-remote flag")
    lines.append("")

    return lines
