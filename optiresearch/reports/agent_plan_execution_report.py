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

    ex = r.get("execution_result") or {}
    execution_title = "## 7. Remote Execution Result" if ex.get("execution_target") == "remote_wsl" else "## 7. Local Execution Result"
    lines.extend([
        "",
        execution_title,
    ])
    if ex:
        lines.extend([
            f"- **Status:** {ex.get('status', '-')}",
            f"- **Design:** {ex.get('design_id', '-')}",
            f"- **Task:** {ex.get('task_type', '-')}",
            f"- **Backend:** {ex.get('backend_id', '-')}",
            f"- **Evidence:** {ex.get('evidence_level', '-')}",
            f"- **Execution Target:** {ex.get('execution_target', 'local')}",
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

    if ex.get("execution_target") == "remote_wsl":
        remote_result = ex.get("remote_handler_result") or ex
        lines.extend([
            "",
            "## 7a. Remote Job",
            "| Field | Value |",
            "|---|---|",
            f"| Worker | {ex.get('remote_worker_id', '-')} |",
            f"| Job ID | {ex.get('remote_job_id', '-')} |",
            f"| Run ID | {ex.get('run_id', '-')} |",
            f"| Target | {ex.get('execution_target', '-')} |",
            f"| Status | {ex.get('status', '-')} |",
            "",
            "## 7b. Remote Result Ingestion",
            "| Field | Value |",
            "|---|---|",
            f"| Validation Passed | {remote_result.get('remote_validation_passed', ex.get('remote_validation_passed', '-'))} |",
            f"| Evidence Level | {remote_result.get('evidence_level', ex.get('evidence_level', '-'))} |",
            f"| Execution Fidelity | {remote_result.get('execution_fidelity', ex.get('execution_fidelity', '-'))} |",
            f"| Proxy Fallback Used | {remote_result.get('proxy_fallback_used', ex.get('proxy_fallback_used', '-'))} |",
            f"| Native PSF Path | {remote_result.get('deeplens_native_psf_path', ex.get('deeplens_native_psf_path', '-'))} |",
            f"| Full Wave Optics | {remote_result.get('full_wave_optics', ex.get('full_wave_optics', '-'))} |",
            f"| Phase-to-FFT Proxy Used | {remote_result.get('phase_to_fft_proxy_used', ex.get('phase_to_fft_proxy_used', '-'))} |",
            "",
            "## 7c. Artifact Return Path",
            "| Field | Value |",
            "|---|---|",
            f"| Artifact Return Path | {ex.get('artifact_return_path', remote_result.get('artifact_return_path', '-'))} |",
            f"| Artifact Count | {len(ex.get('artifacts', []))} |",
        ])
        artifacts = ex.get("artifacts", [])
        if artifacts:
            lines.extend(["", "| Artifact |", "|---|"])
            for artifact in artifacts:
                lines.append(f"| {artifact} |")
        claim = r.get("claim_gate_decision") or {}
        lines.extend([
            "",
            "## 7d. Remote Claim Ceiling",
            "| Field | Value |",
            "|---|---|",
            f"| Final Claim Ceiling | {claim.get('final_claim_ceiling') or claim.get('max_allowed_claim', '-')} |",
            f"| Ceiling Source | {claim.get('ceiling_source', '-')} |",
            f"| Validation | {'passed' if ex.get('remote_validation_passed') else 'failed'} |",
            "",
            "## 7e. Remote Event Sequence",
            "| # | Event |",
            "|---|---|",
        ])
        remote_events = _remote_event_sequence(execution_id, r)
        if remote_events:
            for idx, event_type in enumerate(remote_events, 1):
                lines.append(f"| {idx} | {event_type} |")
        else:
            lines.append("| - | none |")
        lines.extend([
            "",
            "## 7f. Remote Validation Explanation",
        ])
        if ex.get("remote_validation_passed"):
            lines.append("Remote validation passed without proxy fallback.")
        else:
            lines.append("Remote validation failed or did not run; claim ceiling remains needs_followup.")

    # Evidence alignment section
    lines.extend(["", "## 7g. Evidence Alignment" if ex.get("execution_target") == "remote_wsl" else "## 7b. Evidence Alignment"])
    designs = r.get("candidate_designs", [])
    aligned = sum(1 for d in designs if d.get("evidence_alignment_status") == "aligned")
    downgraded = sum(1 for d in designs if d.get("evidence_alignment_status") == "downgraded_to_handler_capability")
    unsupported = sum(1 for d in designs if d.get("evidence_alignment_status") == "unsupported")
    lines.extend([
        f"- **Aligned:** {aligned}",
        f"- **Downgraded to Handler Capability:** {downgraded}",
        f"- **Unsupported:** {unsupported}",
    ])
    if downgraded > 0:
        lines.extend(["", "### Downgraded Designs", "| Design | Expected | Actual | Reason |", "|---|---|---|---|"])
        for d in designs:
            if d.get("evidence_alignment_status") == "downgraded_to_handler_capability":
                lines.append(
                    f"| {d.get('design_id', '-')} | "
                    f"{d.get('expected_evidence_level', '-')} | "
                    f"{d.get('actual_handler_evidence_level', '-')} | "
                    f"{d.get('evidence_downgrade_reason', '-')[:100]} |"
                )

    # Handler comparison when multiple scientific handlers executed
    attempts = r.get("attempted_designs", [])
    scientific_attempts = [a for a in attempts if a.get("evidence_level") == "lightweight_scientific_execution"]
    if len(scientific_attempts) >= 2:
        lines.extend([
            "",
            "## 7h. Scientific Handler Comparison" if ex.get("execution_target") == "remote_wsl" else "## 7c. Scientific Handler Comparison",
            "| Design | Status | MSE After | PSNR After | Improvement | Handler |",
            "|---|---|---|---|---|---|",
        ])
        for a in scientific_attempts:
            metrics = a.get("metrics", {})
            lines.append(
                f"| {a.get('design_id', '-')} | {a.get('status', '-')} | "
                f"{metrics.get('mse_after', '-')} | {metrics.get('psnr_after', '-')} | "
                f"{metrics.get('improvement_detected', '-')} | {a.get('handler_id', '-')} |"
            )

    # Claim Ceiling Resolution (Phase 41)
    ex = r.get("execution_result") or {}
    lines.extend([
        "",
        "## 7i. Claim Ceiling Resolution" if ex.get("execution_target") == "remote_wsl" else "## 7d. Claim Ceiling Resolution",
        f"- **Design Backend Ceiling:** {ex.get('design_backend_claim_ceiling', '-')}",
        f"- **Handler Ceiling:** {ex.get('handler_claim_ceiling', '-')}",
        f"- **Dataset Ceiling:** {ex.get('dataset_claim_ceiling', '-')}",
        f"- **Execution Fidelity Ceiling:** {ex.get('execution_fidelity_claim_ceiling', '-')}",
    ])
    claim = r.get("claim_gate_decision") or {}
    if claim:
        lines.extend([
            f"- **Final Claim Ceiling:** {claim.get('final_claim_ceiling') or claim.get('max_allowed_claim', '-')}",
            f"- **Ceiling Source:** {claim.get('ceiling_source', '-')}",
            f"- **Limiting Factor:** {claim.get('limiting_factor', '-')}",
        ])
        reasons = claim.get("downgrade_reasons", [])
        if reasons:
            lines.append("- **Downgrade Reasons:**")
            for reason in reasons:
                lines.append(f"  - {reason}")

    # Phase 46: ArtifactStore Evidence Index
    artifact_ids = ex.get("artifact_ids", [])
    if artifact_ids:
        lines.extend([
            "",
            "## 7e. ArtifactStore Evidence Index",
            f"- **Ingested Count:** {len(artifact_ids)}",
            f"- **SHA256 Verified:** {ex.get('sha256_verified', '-')}",
            f"- **Ingestion Status:** {ex.get('artifact_ingestion_status', '-')}",
        ])
        evidence_ids = claim.get("evidence_artifact_ids", [])
        if evidence_ids:
            lines.append(f"- **Evidence Artifact IDs:** {', '.join(str(a) for a in evidence_ids[:10])}")

    lines.extend(["", "## 8. ClaimGate Outcome"])
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


def _remote_event_sequence(execution_id: str, r: dict[str, Any]) -> list[str]:
    event_path = r.get("event_log_path")
    candidates = []
    if event_path:
        candidates.append(Path(event_path))
    candidates.append(Path("workspace/agent_plan_executions") / execution_id / "events.json")
    for path in candidates:
        data = _read_json_list(path)
        if not data:
            continue
        return [
            item.get("event_type", "")
            for item in data
            if str(item.get("event_type", "")).startswith("remote_")
            or item.get("event_type") == "artifact_ingested"
        ]
    return []


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []
