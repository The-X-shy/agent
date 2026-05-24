"""Ingest remote worker outputs into local memory, artifacts, and claims."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from optiresearch.memory.claim_evidence import ClaimEvidenceManager
from optiresearch.memory.compiler import MemoryCompiler
from optiresearch.memory.meta_trace import MetaTrace, MetaTraceWriter
from optiresearch.memory.schemas import StrictModel, make_deterministic_id
from optiresearch.schemas.remote import RemoteJobResult
from optiresearch.storage.file_artifact_store import FileArtifactStore


NATIVE_REMOTE_EVIDENCE_LEVELS = {
    "native_lens_simulation",
    "native_lens_hsi_codesign",
    "rollback_protected_native_lens_hsi",
}


class RemoteHandlerResult(StrictModel):
    remote_job_id: str = ""
    worker_id: str = ""
    execution_target: Literal["remote_wsl"] = "remote_wsl"
    status: str = "failed"
    run_id: str = ""
    remote_validation_passed: bool = False
    evidence_level: str = "needs_followup"
    execution_fidelity: str = ""
    proxy_fallback_used: bool = False
    deeplens_native_psf_path: str = ""
    full_wave_optics: bool = False
    phase_to_fft_proxy_used: bool = False
    metrics: dict[str, Any] = {}
    artifacts: list[str] = []
    artifact_return_path: str = ""
    errors: list[dict[str, str]] = []
    caveats: list[str] = []
    warnings: list[dict[str, str]] = []
    ingestion: dict[str, Any] = {}


def parse_remote_handler_result(
    source: RemoteJobResult | dict[str, Any] | str | Path,
    worker_id: str = "",
    ingestion: dict[str, Any] | None = None,
) -> RemoteHandlerResult:
    """Parse remote job output into the native GeoLens handler contract."""

    remote_job = _coerce_remote_job_result(source)
    metrics_summary = dict(remote_job.metrics_summary or {})
    local_dir = Path(remote_job.local_output_dir)
    result_payload = _read_json(_find_result_json(local_dir), {})
    metrics = _extract_remote_metrics(metrics_summary, result_payload)
    artifacts = _artifact_paths(local_dir, remote_job.artifact_manifest)
    warnings: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    evidence_level = str(
        metrics_summary.get("evidence_level")
        or result_payload.get("evidence_level")
        or "needs_followup"
    )
    execution_fidelity = str(
        metrics_summary.get("execution_fidelity")
        or result_payload.get("execution_fidelity")
        or ""
    )
    proxy_fallback_used = _bool_field(metrics_summary, result_payload, "proxy_fallback_used", "fallback_used")
    full_wave_optics = _bool_field(metrics_summary, result_payload, "full_wave_optics")
    phase_to_fft_proxy_used = _bool_field(metrics_summary, result_payload, "phase_to_fft_proxy_used")
    deeplens_native_psf_path = str(
        metrics_summary.get("deeplens_native_psf_path")
        or result_payload.get("deeplens_native_psf_path")
        or ""
    )
    run_id = str(
        metrics_summary.get("remote_run_id")
        or metrics_summary.get("run_id")
        or result_payload.get("run_id")
        or remote_job.remote_run_id
        or remote_job.job_id
    )
    handler_status = str(
        metrics_summary.get("status")
        or result_payload.get("status")
        or remote_job.status
    )

    for field_name, value in (
        ("evidence_level", evidence_level),
        ("execution_fidelity", execution_fidelity),
        ("deeplens_native_psf_path", deeplens_native_psf_path),
    ):
        if value in ("", "None", "needs_followup"):
            errors.append({"type": "REMOTE_RESULT_MISSING_FIELD", "message": f"Missing required field: {field_name}"})

    if _missing_bool(metrics_summary, result_payload, "proxy_fallback_used", "fallback_used"):
        errors.append({"type": "REMOTE_RESULT_MISSING_FIELD", "message": "Missing required field: proxy_fallback_used"})
    if _missing_bool(metrics_summary, result_payload, "full_wave_optics"):
        errors.append({"type": "REMOTE_RESULT_MISSING_FIELD", "message": "Missing required field: full_wave_optics"})
    if _missing_bool(metrics_summary, result_payload, "phase_to_fft_proxy_used"):
        errors.append({"type": "REMOTE_RESULT_MISSING_FIELD", "message": "Missing required field: phase_to_fft_proxy_used"})

    if remote_job.status != "succeeded":
        errors.append({
            "type": remote_job.error_code or "REMOTE_JOB_FAILED",
            "message": f"Remote job status is {remote_job.status}",
        })
    if handler_status not in {"succeeded", "completed"}:
        errors.append({
            "type": "REMOTE_HANDLER_STATUS_NOT_SUCCEEDED",
            "message": f"Remote handler status is {handler_status}",
        })
    if proxy_fallback_used:
        errors.append({"type": "REMOTE_PROXY_FALLBACK_USED", "message": "Fallback result cannot be treated as native"})
    if deeplens_native_psf_path and deeplens_native_psf_path != "geolens.psf_geometric":
        errors.append({
            "type": "REMOTE_NATIVE_PSF_PATH_MISMATCH",
            "message": f"Expected geolens.psf_geometric, got {deeplens_native_psf_path}",
        })
    if phase_to_fft_proxy_used:
        errors.append({"type": "REMOTE_FFT_PROXY_USED", "message": "phase_to_fft_proxy_used must be false"})
    if full_wave_optics:
        errors.append({"type": "REMOTE_FULL_WAVE_OPTICS_UNEXPECTED", "message": "GeoLens geometric validation must not claim full wave optics"})
    if evidence_level not in NATIVE_REMOTE_EVIDENCE_LEVELS:
        errors.append({
            "type": "REMOTE_EVIDENCE_LEVEL_UNSUPPORTED",
            "message": f"Unsupported remote evidence level: {evidence_level}",
        })
    if not artifacts:
        warnings.append({"type": "REMOTE_ARTIFACTS_EMPTY", "message": "No returned artifacts were found"})

    validation_passed = not errors
    status = "succeeded" if validation_passed else "failed"
    if not validation_passed:
        evidence_level = "needs_followup"

    return RemoteHandlerResult(
        remote_job_id=remote_job.job_id,
        worker_id=worker_id,
        execution_target="remote_wsl",
        status=status,
        run_id=run_id,
        remote_validation_passed=validation_passed,
        evidence_level=evidence_level,
        execution_fidelity=execution_fidelity,
        proxy_fallback_used=proxy_fallback_used,
        deeplens_native_psf_path=deeplens_native_psf_path,
        full_wave_optics=full_wave_optics,
        phase_to_fft_proxy_used=phase_to_fft_proxy_used,
        metrics=metrics,
        artifacts=artifacts,
        artifact_return_path=str(local_dir),
        errors=errors,
        caveats=list(remote_job.caveats or []),
        warnings=warnings,
        ingestion=ingestion or {},
    )


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


def _coerce_remote_job_result(source: RemoteJobResult | dict[str, Any] | str | Path) -> RemoteJobResult:
    if isinstance(source, RemoteJobResult):
        return source
    if isinstance(source, (str, Path)):
        payload = _read_json(Path(source), {})
        return _coerce_remote_job_result(payload)
    payload = dict(source)
    if "result" in payload and isinstance(payload["result"], RemoteJobResult):
        return payload["result"]
    if "result" in payload and isinstance(payload["result"], dict):
        payload = payload["result"]
    return RemoteJobResult(**payload)


def _find_result_json(local_dir: Path) -> Path:
    candidates = [
        local_dir / "result.json",
        local_dir / local_dir.name / "result.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _extract_remote_metrics(
    metrics_summary: dict[str, Any],
    result_payload: dict[str, Any],
) -> dict[str, Any]:
    keys = (
        "reconstruction_loss_before",
        "reconstruction_loss_after",
        "best_reconstruction_loss",
        "mse_before",
        "mse_after",
        "psnr_before",
        "psnr_after",
        "improvement_detected",
        "metrics_valid",
        "accepted_update_count",
        "rollback_count",
        "optical_gradient_norm",
        "optical_gradient_norm_max",
        "stable_training_succeeded",
    )
    metrics: dict[str, Any] = {}
    for source in (result_payload, metrics_summary):
        for key in keys:
            if key in source:
                metrics[key] = source[key]
    return metrics


def _artifact_paths(local_dir: Path, manifest: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for item in manifest.get("artifacts", []) if isinstance(manifest, dict) else []:
        rel = item.get("path")
        if not rel:
            continue
        path = local_dir / rel
        if not path.exists():
            nested = local_dir / local_dir.name / rel
            path = nested if nested.exists() else path
        paths.append(str(path))
    for fallback in (local_dir / "result.json", local_dir / "spec.json"):
        if fallback.exists() and str(fallback) not in paths:
            paths.append(str(fallback))
    nested_dir = local_dir / local_dir.name
    for name in ("result.json", "spec.json"):
        fallback = nested_dir / name
        if fallback.exists() and str(fallback) not in paths:
            paths.append(str(fallback))
    return paths


def _bool_field(
    metrics_summary: dict[str, Any],
    result_payload: dict[str, Any],
    key: str,
    alias: str | None = None,
) -> bool:
    for source in (metrics_summary, result_payload):
        if key in source:
            return bool(source[key])
        if alias and alias in source:
            return bool(source[alias])
    return False


def _missing_bool(
    metrics_summary: dict[str, Any],
    result_payload: dict[str, Any],
    key: str,
    alias: str | None = None,
) -> bool:
    return not any(
        key in source or (alias is not None and alias in source)
        for source in (metrics_summary, result_payload)
    )
