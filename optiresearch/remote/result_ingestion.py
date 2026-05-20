"""Ingest remote worker outputs into local memory, artifacts, and claims."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from optiresearch.memory.claim_evidence import ClaimEvidenceManager
from optiresearch.memory.compiler import MemoryCompiler
from optiresearch.memory.meta_trace import MetaTrace, MetaTraceWriter
from optiresearch.memory.schemas import make_deterministic_id
from optiresearch.schemas.remote import RemoteJobResult
from optiresearch.storage.file_artifact_store import FileArtifactStore


def ingest_remote_job_result(
    result: RemoteJobResult,
    workspace_id: str = "default",
) -> dict[str, Any]:
    local_dir = Path(result.local_output_dir)
    artifact_manifest = result.artifact_manifest or _read_json(local_dir / "artifact_manifest.json", {})
    metrics_summary = result.metrics_summary or _read_json(local_dir / "metrics_summary.json", {})
    run_id = result.remote_run_id or result.job_id

    trace_id = make_deterministic_id("trace", workspace_id, result.job_id, "remote_ingest")
    trace = MetaTrace(
        trace_id=trace_id,
        workspace_id=workspace_id,
        run_id=run_id,
        branch_id=None,
        step_id=None,
        phase="Execute",
        actor="System",
        task=f"Ingest remote job {result.job_id}",
        skill_id=None,
        skill_version=None,
        tool="SSHRemoteRunner",
        next_action=None,
        status="succeeded" if result.status == "succeeded" else "failed",
        timestamp_start=None,
        timestamp_end=None,
        content_hash=None,
        findings=[f"remote_status={result.status}"],
        limitations=result.caveats,
        metadata={
            "objective": metrics_summary.get("objective") or result.job_id,
            "backend": metrics_summary.get("backend", "deeplens"),
            "remote_job_id": result.job_id,
            "remote_output_dir": result.remote_output_dir,
            "local_output_dir": result.local_output_dir,
            "error_code": result.error_code,
            "fallback_used": metrics_summary.get("fallback_used"),
        },
    )
    MetaTraceWriter().write_trace(trace)

    artifact_store = FileArtifactStore()
    artifact_ids: list[str] = []
    artifact_uris: list[str] = []
    for item in artifact_manifest.get("artifacts", []):
        rel_path = item.get("path")
        if not rel_path:
            continue
        path = local_dir / rel_path
        if not path.exists():
            nested = local_dir / local_dir.name / rel_path
            path = nested if nested.exists() else path
        if not path.exists() or not path.is_file():
            continue
        ref = artifact_store.register_file(
            path,
            workspace_id=workspace_id,
            run_id=run_id,
            trace_id=trace_id,
            producer="remote-worker",
            metadata={
                "filename": path.name,
                "artifact_type": item.get("artifact_type", "unknown"),
                "backend": metrics_summary.get("backend", "deeplens"),
                "remote_job_id": result.job_id,
                "remote_path": item.get("remote_path"),
                "backend_capability_level": metrics_summary.get("backend_capability_level"),
                "selected_realization_level": metrics_summary.get("selected_realization_level"),
                "proxy_fallback_used": metrics_summary.get("fallback_used"),
            },
            metrics=item.get("metrics", {}),
        )
        artifact_ids.append(ref.artifact_id)
        artifact_uris.append(ref.uri)

    claim_summaries: list[dict[str, Any]] = []
    claim_manager = ClaimEvidenceManager(workspace_id=workspace_id)
    if result.status == "succeeded" and artifact_ids:
        claim_text, scope = _claim_for_result(result, metrics_summary)
        claim = claim_manager.create_claim(claim_text, scope)
        score = 0.82 if not metrics_summary.get("fallback_used") else 0.35
        for artifact_id in artifact_ids:
            claim_manager.attach_support(claim.claim_id, artifact_id, score)
        reviewed = claim_manager.review_claim(claim.claim_id)
        claim_summaries.append(claim_manager.explain_claim(reviewed.claim_id))

    try:
        run_memory = MemoryCompiler().compile_run_memory(run_id)
        memory_payload: dict[str, Any] | None = run_memory.model_dump(mode="json")
    except Exception as exc:
        memory_payload = {"error": str(exc)}

    summary = {
        "job_id": result.job_id,
        "run_id": run_id,
        "artifact_ids": artifact_ids,
        "artifact_uris": artifact_uris,
        "claims": claim_summaries,
        "run_memory": memory_payload,
    }
    (local_dir / "ingestion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return summary


def _claim_for_result(result: RemoteJobResult, metrics_summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    job_type = metrics_summary.get("job_type")
    fallback_used = bool(metrics_summary.get("fallback_used"))
    scope = {
        "run_id": result.remote_run_id or result.job_id,
        "backend": "deeplens",
        "backend_capability_level": metrics_summary.get("backend_capability_level", "source"),
        "selected_realization_level": metrics_summary.get("selected_realization_level"),
        "evidence_domain": metrics_summary.get("evidence_domain", "remote_worker"),
        "proxy_fallback_used": fallback_used,
        "claim_scope": metrics_summary.get("claim_scope"),
    }
    if fallback_used:
        return "Remote job completed but used fallback; it is not DeepLens-backed evidence.", scope
    if job_type == "codesign_loop":
        scope["selected_realization_level"] = metrics_summary.get("selected_realization_level", "adapter_proxy")
        return "Remote DeepLens-backed black-box co-design completed without fallback.", scope
    if job_type == "deeplens_source_smoke":
        scope["backend_capability_level"] = "smoke"
        return "Remote WSL worker produced DeepLens source smoke evidence.", scope
    return "Remote WSL worker produced DeepLens-backed execution evidence.", scope


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
