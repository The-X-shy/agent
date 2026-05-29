"""SSH subprocess runner for remote workers."""

from __future__ import annotations

import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from optiresearch.remote.command_allowlist import CommandValidationError, validate_remote_command
from optiresearch.schemas.remote import RemoteJobResult, RemoteJobSpec, RemoteWorkerSpec


ProcessRunner = Callable[..., subprocess.CompletedProcess]


class SSHRemoteRunner:
    def __init__(
        self,
        worker: RemoteWorkerSpec | None = None,
        workspace_root: str | Path = "workspace",
        process_runner: ProcessRunner | None = None,
    ) -> None:
        self.worker = worker
        self.workspace_root = Path(workspace_root)
        self._process_runner = process_runner or subprocess.run

    def check_connection(self) -> dict[str, Any]:
        worker = self._require_worker()
        return self.run_command(
            [worker.python_executable, "-m", "optiresearch.cli", "check-deeplens"],
            cwd=worker.remote_project_dir,
            timeout=min(worker.max_runtime_seconds, 60),
            job_id="check_remote_worker",
        )

    def run_command(
        self,
        command: list[str],
        cwd: str,
        timeout: int,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            validation = validate_remote_command(command)
        except CommandValidationError as exc:
            return self._structured_command_error(command, job_id, "REMOTE_COMMAND_NOT_ALLOWLISTED", str(exc))

        ssh_command = self.build_ssh_command(command, cwd)
        log_dir = self._log_dir(job_id or "adhoc")
        stdout_path = log_dir / "stdout.txt"
        stderr_path = log_dir / "stderr.txt"
        try:
            completed = self._process_runner(
                ssh_command,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout_path.write_text(exc.stdout or "", encoding="utf-8")
            stderr_path.write_text(exc.stderr or "", encoding="utf-8")
            return {
                "status": "failed",
                "error_code": "REMOTE_COMMAND_TIMEOUT",
                "returncode": None,
                "command": command,
                "ssh_command": ssh_command,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "validation": validation,
            }

        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        status = "succeeded" if completed.returncode == 0 else "failed"
        return {
            "status": status,
            "error_code": None if status == "succeeded" else "REMOTE_COMMAND_FAILED",
            "returncode": completed.returncode,
            "command": command,
            "ssh_command": ssh_command,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "validation": validation,
        }

    def upload_file(self, local_path: Path, remote_path: str) -> dict[str, Any]:
        worker = self._require_worker()
        scp_command = [*self._scp_base(worker), str(local_path), f"{self._target(worker)}:{remote_path}"]
        return self._run_transfer(scp_command, "upload")

    def download_dir(self, remote_path: str, local_path: Path) -> dict[str, Any]:
        worker = self._require_worker()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        scp_command = [*self._scp_base(worker), "-r", f"{self._target(worker)}:{remote_path}", str(local_path)]
        return self._run_transfer(scp_command, "download")

    def run_remote_job(self, worker: RemoteWorkerSpec, job: RemoteJobSpec) -> RemoteJobResult:
        self.worker = worker
        local_job_dir = self.workspace_root / "remote_jobs" / job.job_id
        local_job_dir.mkdir(parents=True, exist_ok=True)
        started = _now()
        remote_output_dir = f"{worker.remote_workspace_dir.rstrip('/')}/remote_jobs/{job.job_id}"
        command = build_job_command(worker, job)
        job_config = {
            "worker": worker.model_dump(mode="json"),
            "job": job.model_dump(mode="json"),
            "remote_output_dir": remote_output_dir,
            "command": command,
        }
        (local_job_dir / "job_config.json").write_text(
            json.dumps(job_config, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        command_result = self.run_command(
            command,
            cwd=worker.remote_project_dir,
            timeout=job.timeout_seconds,
            job_id=job.job_id,
        )
        download_result = self.download_dir(remote_output_dir, local_job_dir)
        remote_payload = self._read_remote_job_payload(local_job_dir)
        finished = _now()

        status = "succeeded" if command_result["status"] == "succeeded" else "failed"
        error_code = command_result.get("error_code")
        caveats = list(remote_payload.get("caveats", []))
        if remote_payload.get("status") == "failed":
            status = "failed"
            error_code = remote_payload.get("error_code") or error_code or "REMOTE_JOB_FAILED"
        if download_result["status"] == "failed" and status == "succeeded":
            status = "failed"
            error_code = "REMOTE_OUTPUT_DOWNLOAD_FAILED"

        result = RemoteJobResult(
            job_id=job.job_id,
            status=status,
            remote_run_id=remote_payload.get("remote_run_id"),
            started_at=started,
            finished_at=finished,
            command=command,
            stdout_path=command_result["stdout_path"],
            stderr_path=command_result["stderr_path"],
            remote_output_dir=remote_output_dir,
            local_output_dir=str(local_job_dir),
            artifact_manifest=remote_payload.get("artifact_manifest", {}),
            metrics_summary=remote_payload.get("metrics_summary", {}),
            error_code=error_code,
            caveats=caveats,
        )
        (local_job_dir / "remote_job_result.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        return result

    def build_ssh_command(self, command: list[str], cwd: str) -> list[str]:
        worker = self._require_worker()
        remote_command = "cd {cwd} && {command}".format(
            cwd=shlex.quote(cwd),
            command=shlex.join(command),
        )
        env_prefix = _build_env_prefix()
        if env_prefix:
            remote_command = f"{env_prefix} && {remote_command}"
        return [*self._ssh_base(worker), self._target(worker), remote_command]

    def _read_remote_job_payload(self, local_job_dir: Path) -> dict[str, Any]:
        candidates = [
            local_job_dir / local_job_dir.name / "remote_job_result.json",
            local_job_dir / "remote_job_result.json",
        ]
        for path in candidates:
            if path.exists():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    return {}
        manifest = local_job_dir / local_job_dir.name / "artifact_manifest.json"
        metrics = local_job_dir / local_job_dir.name / "metrics_summary.json"
        return {
            "artifact_manifest": json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {},
            "metrics_summary": json.loads(metrics.read_text(encoding="utf-8")) if metrics.exists() else {},
        }

    def _run_transfer(self, command: list[str], operation: str) -> dict[str, Any]:
        try:
            completed = self._process_runner(command, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired as exc:
            return {"status": "failed", "error_code": f"REMOTE_{operation.upper()}_TIMEOUT", "stderr": exc.stderr or ""}
        return {
            "status": "succeeded" if completed.returncode == 0 else "failed",
            "error_code": None if completed.returncode == 0 else f"REMOTE_{operation.upper()}_FAILED",
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "command": command,
        }

    def _structured_command_error(
        self,
        command: list[str],
        job_id: str | None,
        error_code: str,
        message: str,
    ) -> dict[str, Any]:
        log_dir = self._log_dir(job_id or "adhoc")
        stdout_path = log_dir / "stdout.txt"
        stderr_path = log_dir / "stderr.txt"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(message, encoding="utf-8")
        return {
            "status": "failed",
            "error_code": error_code,
            "returncode": None,
            "command": command,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }

    def _log_dir(self, job_id: str) -> Path:
        path = self.workspace_root / "remote_jobs" / job_id / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _require_worker(self) -> RemoteWorkerSpec:
        if self.worker is None:
            raise ValueError("SSHRemoteRunner requires a worker")
        return self.worker

    def _ssh_base(self, worker: RemoteWorkerSpec) -> list[str]:
        base = ["ssh", "-p", str(worker.port)]
        if worker.ssh_key_path:
            base.extend(["-i", worker.ssh_key_path])
        return base

    def _scp_base(self, worker: RemoteWorkerSpec) -> list[str]:
        base = ["scp", "-P", str(worker.port)]
        if worker.ssh_key_path:
            base.extend(["-i", worker.ssh_key_path])
        return base

    def _target(self, worker: RemoteWorkerSpec) -> str:
        return f"{worker.username}@{worker.host}"


def _build_env_prefix() -> str:
    """Build an env export prefix for lens resolution on remote workers."""
    import os
    parts: list[str] = []
    for var in ("DEEPLENS_REPO_PATH", "OPTIRESEARCH_COOKE_LENS_FILE"):
        val = os.getenv(var)
        if val:
            parts.append(f"export {var}={shlex.quote(val)}")
    return " && ".join(parts) if parts else ""


def build_job_command(worker: RemoteWorkerSpec, job: RemoteJobSpec) -> list[str]:
    cli_command = {
        "deeplens_source_smoke": "run-deeplens-source-smoke",
        "codesign_loop": "run-codesign-loop",
        "hsi_reconstruction": "run-hsi-reconstruction",
        "hsi_matrix": "run-hsi-matrix",
        "autonomous_loop": "run-autonomous-loop",
        "native_optimization_probe": "run-native-optimization-probe",
        "deeplens_surface_optimization_probe": "run-deeplens-surface-optimization-probe",
        "deeplens_lensfile_optimization_probe": "run-deeplens-lensfile-optimization-probe",
        "native_hsi_codesign": "run-native-hsi-codesign",
        "native_hsi_reconstruction_codesign": "run-native-hsi-reconstruction-codesign",
        "deeplens_waveoptics_probe": "run-deeplens-waveoptics-probe",
        "native_waveoptics_hsi_codesign": "run-native-waveoptics-hsi-codesign",
        "stable_native_lens_hsi_codesign": "run-stable-native-lens-hsi-codesign",
        "stable_native_lens_hsi_ablation": "run-stable-native-lens-hsi-ablation",
        "deeplens_native_geolens_hsi_codesign": "run-deeplens-native-geolens-hsi-codesign",
        "native_geolens_stabilization_sweep": "run-native-geolens-stabilization-sweep",
        # Phase 58: remote diagnostic jobs
        "deeplens_trainable_parameter_inspection": "run-deeplens-trainable-parameter-inspection",
        "deeplens_autograd_audit": "run-deeplens-autograd-audit",
        "deeplens_curriculum_probe": "run-deeplens-curriculum-probe",
        "deeplens_regularized_probe": "run-deeplens-regularized-probe",
        # Phase 59: lens file resolver
        "resolve_lens_file": "resolve-lens-file",
        # Phase 62: component backend validation
        "deeplens_component_probe": "run-deeplens-component-probe",
        "deeplens_component_discovery": "discover-deeplens-components",
        "component_surrogate_hsi_codesign": "run-component-surrogate-hsi-codesign",
    }[job.job_type]
    # Build env prefix for lens resolution on remote workers
    env_prefix = _build_env_prefix()
    command = [worker.python_executable, "-m", "optiresearch.cli", cli_command]
    args = dict(job.cli_args)
    if job.objective and cli_command in {"run-codesign-loop", "run-hsi-reconstruction", "run-hsi-matrix", "run-autonomous-loop"}:
        args.setdefault("objective", job.objective)
    args["remote_job_id"] = job.job_id
    for key, value in args.items():
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(option)
            continue
        if value is None:
            continue
        command.extend([option, str(value)])
    validate_remote_command(command)
    return command


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
