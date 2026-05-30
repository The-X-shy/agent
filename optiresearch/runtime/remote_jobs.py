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


def run_remote_native_optimization_probe(
    worker_id: str,
    lens_class: str,
    objective: str,
    max_steps: int = 2,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    strict_native: bool = True,
    allow_adapter_proxy: bool = False,
    runner: Any | None = None,
    ingest: bool = True,
) -> dict[str, Any]:
    job = _job(
        "native_optimization_probe",
        objective=f"Native optimization probe: {lens_class} / {objective}",
        cli_args={
            "lens_class": lens_class,
            "objective": objective,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "device": device,
            "strict_native": strict_native,
            "allow_adapter_proxy": allow_adapter_proxy,
        },
        timeout_seconds=1800,
        expected_outputs=["probe_result.json", "psf_before.npz", "psf_after.npz", "loss_trace.json"],
    )
    return execute_remote_job(worker_id, job, runner=runner, ingest=ingest)


def run_remote_deeplens_surface_optimization_probe(
    worker_id: str,
    surface: str,
    objective: str,
    max_steps: int = 3,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    runner: Any | None = None,
    ingest: bool = True,
) -> dict[str, Any]:
    job = _job(
        "deeplens_surface_optimization_probe",
        objective=f"DeepLens surface optimization probe: {surface} / {objective}",
        cli_args={
            "surface": surface,
            "objective": objective,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "device": device,
        },
        timeout_seconds=1800,
        expected_outputs=[
            "probe_result.json",
            "loss_trace.json",
            "parameter_before.json",
            "parameter_after.json",
            "phase_before.npz",
            "phase_after.npz",
        ],
    )
    return execute_remote_job(worker_id, job, runner=runner, ingest=ingest)


def run_remote_deeplens_lensfile_optimization_probe(
    worker_id: str,
    lens_class: str,
    max_files: int = 5,
    max_steps: int = 2,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    runner: Any | None = None,
    ingest: bool = True,
) -> dict[str, Any]:
    job = _job(
        "deeplens_lensfile_optimization_probe",
        objective=f"DeepLens lens-file optimization probe: {lens_class}",
        cli_args={
            "lens_class": lens_class,
            "max_files": max_files,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "device": device,
        },
        timeout_seconds=1800,
        expected_outputs=["probe_result.json", "loss_trace.json", "parameter_before.json", "parameter_after.json"],
    )
    return execute_remote_job(worker_id, job, runner=runner, ingest=ingest)


def run_remote_native_hsi_codesign(
    worker_id: str,
    optical_component: str,
    objective: str,
    max_steps: int = 3,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    bands: int = 31,
    image_size: int = 32,
    psf_size: int = 16,
    runner: Any | None = None,
    ingest: bool = True,
) -> dict[str, Any]:
    job = _job(
        "native_hsi_codesign",
        objective=f"Native optical-HSI co-design: {optical_component} / {objective}",
        cli_args={
            "optical_component": optical_component,
            "objective": objective,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "device": device,
            "bands": bands,
            "image_size": image_size,
            "psf_size": psf_size,
        },
        timeout_seconds=1800,
        expected_outputs=[
            "result.json", "loss_trace.json", "parameter_before.json",
            "parameter_after.json", "psf_before.npz", "psf_after.npz",
            "hsi_proxy_metrics.json",
        ],
    )
    return execute_remote_job(worker_id, job, runner=runner, ingest=ingest)


def run_remote_native_hsi_reconstruction_codesign(
    worker_id: str,
    optical_component: str,
    reconstructor: str,
    max_steps: int = 5,
    optical_lr: float = 1e-3,
    recon_lr: float = 1e-3,
    device: str = "cpu",
    bands: int = 31,
    image_size: int = 32,
    psf_size: int = 16,
    runner: Any | None = None,
    ingest: bool = True,
) -> dict[str, Any]:
    job = _job(
        "native_hsi_reconstruction_codesign",
        objective=f"Native HSI reconstruction co-design: {optical_component} / {reconstructor}",
        cli_args={
            "optical_component": optical_component,
            "reconstructor": reconstructor,
            "max_steps": max_steps,
            "optical_lr": optical_lr,
            "recon_lr": recon_lr,
            "device": device,
            "bands": bands,
            "image_size": image_size,
            "psf_size": psf_size,
        },
        timeout_seconds=1800,
        expected_outputs=["result.json", "loss_trace.json", "metrics.json"],
    )
    return execute_remote_job(worker_id, job, runner=runner, ingest=ingest)


def run_remote_deeplens_waveoptics_probe(
    worker_id: str, candidate: str, objective: str,
    psf_size: int = 32, max_steps: int = 3, learning_rate: float = 1e-3,
    device: str = "cpu",
    runner: Any | None = None, ingest: bool = True,
) -> dict[str, Any]:
    job = _job(
        "deeplens_waveoptics_probe",
        objective=f"DeepLens wave-optics probe: {candidate} / {objective}",
        cli_args={"candidate": candidate, "objective": objective,
                  "psf_size": psf_size, "max_steps": max_steps,
                  "learning_rate": learning_rate, "device": device},
        timeout_seconds=3600,
        expected_outputs=["result.json", "report.md"],
    )
    return execute_remote_job(worker_id, job, runner=runner, ingest=ingest)


def run_remote_native_waveoptics_hsi_codesign(
    worker_id: str, candidate: str, reconstructor: str,
    dataset: str = "synthetic",
    max_steps: int = 3, optical_lr: float = 1e-3, recon_lr: float = 1e-3,
    device: str = "cpu", bands: int = 4, image_size: int = 16, psf_size: int = 32,
    runner: Any | None = None, ingest: bool = True,
) -> dict[str, Any]:
    job = _job(
        "native_waveoptics_hsi_codesign",
        objective=f"Native wave-optics HSI co-design: {candidate} / {reconstructor}",
        cli_args={"candidate": candidate, "reconstructor": reconstructor,
                  "dataset": dataset,
                  "max_steps": max_steps, "optical_lr": optical_lr, "recon_lr": recon_lr,
                  "device": device, "bands": bands, "image_size": image_size,
                  "psf_size": psf_size},
        timeout_seconds=3600,
        expected_outputs=["result.json", "report.md"],
    )
    return execute_remote_job(worker_id, job, runner=runner, ingest=ingest)


def run_remote_stable_native_lens_hsi_codesign(
    worker_id: str, candidate: str, reconstructor: str,
    max_steps: int = 10, optical_lr: float = 1e-6, recon_lr: float = 1e-3,
    optical_grad_clip: float = 1.0, device: str = "cpu",
    runner: Any | None = None, ingest: bool = True,
) -> dict[str, Any]:
    job = _job(
        "stable_native_lens_hsi_codesign",
        objective=f"Stable native lens HSI co-design: {candidate} / {reconstructor}",
        cli_args={"candidate": candidate, "reconstructor": reconstructor,
                  "max_steps": max_steps, "optical_lr": optical_lr, "recon_lr": recon_lr,
                  "optical_grad_clip": optical_grad_clip, "device": device, "rollback_on_loss_increase": True},
        timeout_seconds=3600,
        expected_outputs=["result.json", "report.md"],
    )
    return execute_remote_job(worker_id, job, runner=runner, ingest=ingest)


def run_remote_stable_native_lens_hsi_ablation(
    worker_id: str, candidate: str, reconstructor: str, device: str = "cpu",
    runner: Any | None = None, ingest: bool = True,
) -> dict[str, Any]:
    job = _job(
        "stable_native_lens_hsi_ablation",
        objective=f"Stable native lens HSI ablation: {candidate} / {reconstructor}",
        cli_args={"candidate": candidate, "reconstructor": reconstructor, "device": device},
        timeout_seconds=7200,
        expected_outputs=["ablation_results.json", "best_config.json"],
    )
    return execute_remote_job(worker_id, job, runner=runner, ingest=ingest)


def run_remote_deeplens_native_geolens_hsi_codesign(
    worker_id: str,
    lens_file: str = "auto:cooke",
    dataset: str = "synthetic",
    reconstructor: str = "differentiable_linear",
    max_steps: int = 5,
    optical_lr: float = 1e-6,
    device: str = "cpu",
    runner: Any | None = None,
    ingest: bool = True,
) -> dict[str, Any]:
    job = _job(
        "deeplens_native_geolens_hsi_codesign",
        objective=f"DeepLens native GeoLens geometric HSI co-design: {lens_file} / {reconstructor}",
        cli_args={
            "lens_file": lens_file,
            "dataset": dataset,
            "reconstructor": reconstructor,
            "max_steps": max_steps,
            "optical_lr": optical_lr,
            "rollback_on_loss_increase": True,
            "device": device,
        },
        timeout_seconds=3600,
        expected_outputs=["result.json", "spec.json"],
    )
    return execute_remote_job(worker_id, job, runner=runner, ingest=ingest)


def run_remote_native_geolens_stabilization_sweep(
    worker_id: str,
    lens_file: str = "auto:cooke",
    dataset: str = "synthetic",
    reconstructor: str = "differentiable_linear",
    device: str = "cpu",
    runner: Any | None = None,
    ingest: bool = True,
) -> dict[str, Any]:
    job = _job(
        "native_geolens_stabilization_sweep",
        objective=f"Native GeoLens stabilization sweep: {lens_file} / {reconstructor}",
        cli_args={
            "lens_file": lens_file,
            "dataset": dataset,
            "reconstructor": reconstructor,
            "device": device,
        },
        timeout_seconds=7200,
        expected_outputs=["sweep_results.json", "best_config.json", "sweep_table.md"],
    )
    return execute_remote_job(worker_id, job, runner=runner, ingest=ingest)


def run_remote_stabilized_native_geolens_hsi(
    worker_id: str,
    lens_file: str = "auto:cooke",
    dataset: str = "synthetic",
    reconstructor: str = "differentiable_linear",
    max_steps: int = 10,
    spectral_angle_weight: float = 0.2,
    optical_lr: float = 1e-6,
    device: str = "cpu",
    runner: Any | None = None,
    ingest: bool = True,
) -> dict[str, Any]:
    job = _job(
        "stabilized_native_geolens_hsi",
        objective=f"Stabilized native GeoLens HSI co-design: {lens_file} / {reconstructor} "
                  f"(spectral_weight={spectral_angle_weight})",
        cli_args={
            "lens_file": lens_file,
            "dataset": dataset,
            "reconstructor": reconstructor,
            "max_steps": max_steps,
            "spectral_angle_weight": spectral_angle_weight,
            "optical_lr": optical_lr,
            "device": device,
        },
        timeout_seconds=3600,
        expected_outputs=["result.json", "spec.json"],
    )
    return execute_remote_job(worker_id, job, runner=runner, ingest=ingest)


def run_remote_deeplens_trainable_parameter_inspection(
    worker_id: str,
    lens_file: str = "auto:cooke",
    device: str = "cpu",
    runner: Any | None = None,
    ingest: bool = True,
) -> dict[str, Any]:
    return execute_remote_job(worker_id, _job(
        "deeplens_trainable_parameter_inspection",
        "Inspect GeoLens trainable parameters",
        {"lens_file": lens_file, "device": device},
        600, ["inspection_result.json"],
    ), runner, ingest)


def run_remote_deeplens_autograd_audit(
    worker_id: str,
    lens_file: str = "auto:cooke",
    device: str = "cpu",
    runner: Any | None = None,
    ingest: bool = True,
) -> dict[str, Any]:
    return execute_remote_job(worker_id, _job(
        "deeplens_autograd_audit",
        "Audit GeoLens autograd graph",
        {"lens_file": lens_file, "device": device},
        600, ["audit_result.json"],
    ), runner, ingest)


def run_remote_deeplens_curriculum_probe(
    worker_id: str,
    max_steps: int = 3,
    device: str = "cpu",
    runner: Any | None = None,
    ingest: bool = True,
) -> dict[str, Any]:
    return execute_remote_job(worker_id, _job(
        "deeplens_curriculum_probe",
        "Run DeepLens curriculum probe",
        {"max_steps": max_steps, "device": device},
        300, ["curriculum_result.json"],
    ), runner, ingest)


def run_remote_deeplens_regularized_probe(
    worker_id: str,
    max_steps: int = 3,
    device: str = "cpu",
    runner: Any | None = None,
    ingest: bool = True,
) -> dict[str, Any]:
    return execute_remote_job(worker_id, _job(
        "deeplens_regularized_probe",
        "Run DeepLens regularized probe",
        {"max_steps": max_steps, "device": device},
        300, ["reg_probe_result.json"],
    ), runner, ingest)


def run_remote_resolve_lens_file(
    worker_id: str,
    lens_file: str = "auto:cooke",
    backend_id: str | None = None,
    runner: Any | None = None,
    ingest: bool = True,
) -> dict[str, Any]:
    return execute_remote_job(worker_id, _job(
        "resolve_lens_file",
        "Resolve lens file path on remote worker",
        {"lens_file": lens_file, "backend_id": backend_id},
        120, ["resolution_result.json"],
    ), runner, ingest)


def run_remote_deeplens_component_probe(
    worker_id: str,
    component: str = "fresnel",
    objective: str = "parameter_sanity_check",
    max_steps: int = 5,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    runner: Any | None = None,
    ingest: bool = True,
) -> dict[str, Any]:
    return execute_remote_job(worker_id, _job(
        "deeplens_component_probe",
        f"DeepLens component probe: {component} / {objective}",
        {"component": component, "objective": objective,
         "max_steps": max_steps, "learning_rate": learning_rate, "device": device},
        600, ["result.json", "component_probe_metrics.json"],
    ), runner, ingest)


def run_remote_deeplens_component_discovery(
    worker_id: str,
    components: str = "fresnel,binary2phase,diffractive",
    device: str = "cpu",
    runner: Any | None = None,
    ingest: bool = True,
) -> dict[str, Any]:
    return execute_remote_job(worker_id, _job(
        "deeplens_component_discovery",
        "Discover DeepLens component backends",
        {"components": components, "device": device},
        300, ["discovery_result.json"],
    ), runner, ingest)


def run_remote_component_surrogate_hsi_codesign(
    worker_id: str,
    component: str = "fresnel",
    dataset: str = "synthetic",
    steps: int = 3,
    device: str = "cpu",
    runner: Any | None = None,
    ingest: bool = True,
) -> dict[str, Any]:
    return execute_remote_job(worker_id, _job(
        "component_surrogate_hsi_codesign",
        f"Component surrogate HSI co-design: {component} / {dataset}",
        {"component": component, "dataset": dataset, "steps": steps, "device": device},
        600, ["result.json", "metrics.json", "psf_artifact.npz", "artifact_manifest.json", "report.md"],
    ), runner, ingest)


def run_remote_native_optimization_inspection(
    worker_id: str,
    runner: Any | None = None,
    ingest: bool = True,
) -> dict[str, Any]:
    job = _job(
        "native_optimization_inspection",
        objective="Inspect DeepLens native optimization capabilities on remote worker",
        cli_args={},
        timeout_seconds=600,
        expected_outputs=["deeplens_native_optimization_inspection.json"],
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
    output_resolved = output_dir.resolve()
    for source_dir in source_dirs:
        if source_dir.exists():
            source_resolved = source_dir.resolve()
            if source_resolved == output_resolved:
                continue
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
        "backend_capability_level": _remote_backend_capability_level(job_type),
        "selected_realization_level": _remote_realization_level(job_type),
        "claim_scope": _remote_claim_scope(job_type),
    }
    metrics.update(metrics_summary or {})
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    manifest = _build_artifact_manifest(
        output_dir, copied_root,
        job_id=remote_job_id,
        run_id=result.get("run_id", ""),
    )
    # Already written inside _build_artifact_manifest
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


def _remote_backend_capability_level(job_type: str) -> str:
    if job_type == "deeplens_source_smoke":
        return "source"
    if job_type in {"deeplens_surface_optimization_probe", "deeplens_component_probe", "deeplens_component_discovery"}:
        return "native_component"
    if job_type == "component_surrogate_hsi_codesign":
        return "component_surrogate"
    if job_type == "deeplens_lensfile_optimization_probe":
        return "native_lens"
    if job_type == "native_hsi_codesign":
        return "native_component"
    if job_type == "native_hsi_reconstruction_codesign":
        return "native_component"
    if job_type == "deeplens_waveoptics_probe":
        return "native_lens"
    if job_type == "native_waveoptics_hsi_codesign":
        return "native_lens"
    if job_type == "stable_native_lens_hsi_codesign":
        return "native_lens"
    if job_type == "stable_native_lens_hsi_ablation":
        return "native_lens"
    if job_type == "deeplens_native_geolens_hsi_codesign":
        return "native_lens"
    if job_type == "native_geolens_stabilization_sweep":
        return "native_lens"
    if job_type == "stabilized_native_geolens_hsi":
        return "native_lens"
    return "adapter_proxy"


def _remote_realization_level(job_type: str) -> str | None:
    if job_type == "codesign_loop":
        return "adapter_proxy"
    if job_type in {"deeplens_surface_optimization_probe", "deeplens_lensfile_optimization_probe", "deeplens_native_geolens_hsi_codesign", "deeplens_component_probe", "stabilized_native_geolens_hsi"}:
        return "native"
    if job_type == "component_surrogate_hsi_codesign":
        return "component_surrogate"
    return None


def _remote_claim_scope(job_type: str) -> str:
    if job_type in {"deeplens_surface_optimization_probe", "deeplens_component_probe"}:
        return "DeepLens native differentiable component optimization"
    if job_type == "deeplens_component_discovery":
        return "DeepLens component backend discovery"
    if job_type == "component_surrogate_hsi_codesign":
        return "Component-level surrogate PSF HSI co-design"
    if job_type == "deeplens_lensfile_optimization_probe":
        return "DeepLens native differentiable lens optimization"
    if job_type == "native_hsi_codesign":
        return "DeepLens native differentiable optical-HSI proxy co-design"
    if job_type == "native_hsi_reconstruction_codesign":
        return "DeepLens native differentiable optical-HSI reconstruction co-design"
    if job_type == "deeplens_waveoptics_probe":
        return "DeepLens native differentiable lens simulation"
    if job_type == "native_waveoptics_hsi_codesign":
        return "DeepLens native differentiable lens simulation HSI co-design"
    if job_type == "deeplens_native_geolens_hsi_codesign":
        return "DeepLens native GeoLens geometric HSI co-design"
    if job_type == "native_geolens_stabilization_sweep":
        return "DeepLens native GeoLens geometric HSI stabilization sweep"
    return "DeepLens-backed black-box execution, not native differentiable optimization"


def _build_artifact_manifest(
    output_dir: Path, copied_root: Path, job_id: str = "", run_id: str = "", worker_id: str = ""
) -> dict[str, Any]:
    import hashlib
    from datetime import datetime, timezone

    entries: list[dict[str, Any]] = []
    required_outputs = {"result.json", "spec.json"}

    for path in sorted(copied_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(output_dir).as_posix()
        sha256_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        size = path.stat().st_size
        metrics = _extract_metrics(path)
        name = path.name
        entries.append({
            "artifact_name": name,
            "relative_path": rel,
            "artifact_type": _map_artifact_type(name, rel),
            "required": name in required_outputs,
            "sha256": sha256_hash,
            "size_bytes": size,
            "producer": "remote_worker",
            "source_run_id": run_id,
            "source_job_id": job_id,
            "evidence_role": _map_evidence_role(name, rel),
            "metadata": {"metrics": metrics} if metrics else {},
        })

    for name in ["command_result.json", "metrics_summary.json"]:
        path = output_dir / name
        if path.exists():
            sha256_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            size = path.stat().st_size
            entries.append({
                "artifact_name": name,
                "relative_path": name,
                "artifact_type": _map_artifact_type(name, name),
                "required": False,
                "sha256": sha256_hash,
                "size_bytes": size,
                "producer": "remote_cli",
                "source_run_id": run_id,
                "source_job_id": job_id,
                "evidence_role": "auxiliary",
                "metadata": {},
            })

    # Check completeness
    found_names = {e["artifact_name"] for e in entries}
    missing = [r for r in required_outputs if r not in found_names]
    completeness = "complete" if not missing else "partial"

    manifest = {
        "schema_version": "0.1",
        "remote_job_id": job_id,
        "run_id": run_id,
        "worker_id": worker_id,
        "execution_target": "remote_wsl",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": entries,
        "completeness": completeness,
        "missing_required_artifacts": missing,
        "warnings": [],
    }

    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    return manifest


def _map_artifact_type(filename: str, rel_path: str = "") -> str:
    lower = filename.lower()
    if lower.endswith(".npz"):
        return "psf_npz" if "psf" in lower else "reconstruction_npz"
    if lower.endswith(".csv"):
        return "mtf_csv"
    if lower.endswith((".png", ".jpg", ".jpeg")):
        return "figure_png" if lower.endswith(".png") else "figure_jpg"
    if lower.endswith(".json"):
        if "metrics" in lower:
            return "metrics_json"
        if "result" in lower or "spec" in lower:
            return "result_json"
        if "manifest" in lower:
            return "manifest_json"
        if "trace" in lower:
            return "trace_json"
        return "result_json" if "result" in rel_path.lower() else "other"
    if lower.endswith(".md"):
        return "report_md"
    return "other"


def _map_evidence_role(filename: str, rel_path: str = "") -> str:
    lower = filename.lower()
    if "metrics" in lower:
        return "primary_metric"
    if "result" in lower or "spec" in lower:
        return "execution_result"
    if lower.endswith(".npz"):
        if "psf" in lower:
            return "optical_artifact"
        return "reconstruction_artifact"
    if "trace" in lower:
        return "trace"
    if lower.endswith(".md"):
        return "report"
    return "auxiliary"


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
