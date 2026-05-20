"""Remote job orchestration and remote job export helpers."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from optiresearch.memory.schemas import make_deterministic_id
from optiresearch.remote.result_ingestion import ingest_remote_job_result
from optiresearch.remote.ssh_runner import SSHRemoteRunner
from optiresearch.remote.worker_registry import RemoteWorkerRegistry
from optiresearch.schemas.remote import RemoteJobResult, RemoteJobSpec, RemoteWorkerSpec


def run_remote_deeplens_source_smoke(
    worker_id: str,
    runner: Any | None = None,
    ingest: bool = True,
) -> dict[str, Any]:
    job = _job(
        "deeplens_source_smoke",
        objective="Run DeepLens source smoke on remote WSL worker",
        cli_args={},
        timeout_seconds=900,
        expected_outputs=["source_smoke_manifest.json"],
    )
    return execute_remote_job(worker_id, job, runner=runner, ingest=ingest)


def run_remote_codesign(
    worker_id: str,
    objective: str,
    psf_source: str,
    backend: str,
    fallback_policy: str,
    max_iterations: int,
    runner: Any | None = None,
    ingest: bool = True,
) -> dict[str, Any]:
    job = _job(
        "codesign_loop",
        objective=objective,
        cli_args={
            "psf_source": psf_source,
            "backend": backend,
            "fallback_policy": fallback_policy,
            "max_iterations": max_iterations,
            "strict_deeplens": fallback_policy == "fail" and backend == "deeplens",
        },
        timeout_seconds=3600,
        expected_outputs=["codesign_loop_summary.json", "iteration_001_state.json"],
    )
    return execute_remote_job(worker_id, job, runner=runner, ingest=ingest)


def run_remote_hsi_reconstruction(
    worker_id: str,
    objective: str,
    encoder: str,
    reconstructor: str,
    dataset: str,
    backend: str = "deeplens",
    runner: Any | None = None,
    ingest: bool = True,
) -> dict[str, Any]:
    job = _job(
        "hsi_reconstruction",
        objective=objective,
        cli_args={
            "backend": backend,
            "encoder": encoder,
            "reconstructor": reconstructor,
            "dataset": dataset,
        },
        timeout_seconds=3600,
        expected_outputs=["reconstruction_metrics.json"],
    )
    return execute_remote_job(worker_id, job, runner=runner, ingest=ingest)


def execute_remote_job(
    worker_id: str,
    job: RemoteJobSpec,
    runner: Any | None = None,
    ingest: bool = True,
    workspace_id: str = "default",
) -> dict[str, Any]:
    worker = RemoteWorkerRegistry().get_worker(worker_id)
    active_runner = runner or SSHRemoteRunner(worker=worker)
    result: RemoteJobResult = active_runner.run_remote_job(worker, job)
    ingestion = ingest_remote_job_result(result, workspace_id=workspace_id) if ingest else None
    return {"result": result, "ingestion": ingestion}


def check_remote_worker(worker_id: str, runner: Any | None = None) -> dict[str, Any]:
    worker = RemoteWorkerRegistry().get_worker(worker_id)
    active_runner = runner or SSHRemoteRunner(worker=worker)
    return active_runner.check_connection()


def export_remote_job_outputs(
    remote_job_id: str,
    job_type: str,
    result: dict[str, Any],
    source_dirs: list[Path],
    metrics_summary: dict[str, Any] | None = None,
) -> Path:
    """Copy command outputs into workspace/remote_jobs/<job_id> on the worker."""

    output_dir = Path("workspace/remote_jobs") / remote_job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    copied_root = output_dir / "outputs"
    copied_root.mkdir(parents=True, exist_ok=True)
    for source_dir in source_dirs:
        if source_dir.exists():
            dest = copied_root / source_dir.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(source_dir, dest)

    fallback_used = bool(result.get("fallback_used_any") or result.get("fallback_used"))
    status = "failed" if result.get("error") or result.get("status") == "failed" else "succeeded"
    metrics = {
        "job_type": job_type,
        "remote_job_id": remote_job_id,
        "remote_run_id": result.get("loop_id") or result.get("run_id") or remote_job_id,
        "status": status,
        "error_code": result.get("error") or result.get("error_code"),
        "fallback_used": fallback_used,
        "backend": result.get("backend", "deeplens"),
        "objective": result.get("objective"),
        "evidence_domain": "codesign" if job_type == "codesign_loop" else job_type,
        "backend_capability_level": "source" if job_type == "deeplens_source_smoke" else "adapter_proxy",
        "selected_realization_level": "adapter_proxy" if job_type == "codesign_loop" else None,
        "claim_scope": "DeepLens-backed black-box execution, not native differentiable optimization",
    }
    metrics.update(metrics_summary or {})
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    manifest = _build_artifact_manifest(output_dir, copied_root)
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    remote_job_result = {
        "job_id": remote_job_id,
        "status": status,
        "remote_run_id": metrics["remote_run_id"],
        "artifact_manifest": manifest,
        "metrics_summary": metrics,
        "error_code": metrics.get("error_code"),
        "caveats": result.get("caveats", []),
        "local_output_dir": str(output_dir),
    }
    (output_dir / "remote_job_result.json").write_text(
        json.dumps(remote_job_result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return output_dir


def _build_artifact_manifest(output_dir: Path, copied_root: Path) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(copied_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(output_dir).as_posix()
        metrics = _extract_metrics(path)
        artifacts.append(
            {
                "path": rel,
                "artifact_type": _artifact_type(path.name),
                "metrics": metrics,
            }
        )
    for name in ["command_result.json", "metrics_summary.json"]:
        path = output_dir / name
        if path.exists():
            artifacts.append({"path": name, "artifact_type": _artifact_type(name), "metrics": _extract_metrics(path)})
    return {"artifacts": artifacts}


def _extract_metrics(path: Path) -> dict[str, Any]:
    if path.suffix != ".json":
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {k: v for k, v in payload.items() if isinstance(v, (int, float, bool))}


def _artifact_type(filename: str) -> str:
    lower = filename.lower()
    if "metrics" in lower:
        return "metrics"
    if "manifest" in lower or "result" in lower or "summary" in lower:
        return "manifest"
    if lower.endswith(".npz"):
        return "psf_cube"
    if lower.endswith((".png", ".jpg", ".jpeg")):
        return "figure"
    return "unknown"


def _job(
    job_type: str,
    objective: str,
    cli_args: dict[str, Any],
    timeout_seconds: int,
    expected_outputs: list[str],
) -> RemoteJobSpec:
    job_id = make_deterministic_id("remote_job", job_type, objective, time.time())
    return RemoteJobSpec(
        job_id=job_id,
        job_type=job_type,
        objective=objective,
        cli_args=cli_args,
        input_artifacts=[],
        expected_outputs=expected_outputs,
        timeout_seconds=timeout_seconds,
        evidence_policy={"fallback_allowed": False, "claim_policy": "conservative"},
    )
