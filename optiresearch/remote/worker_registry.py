"""Remote worker registry backed by workspace/remote_workers/workers.json."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from optiresearch.remote.command_allowlist import (
    CommandValidationError,
    validate_remote_command,
)
from optiresearch.schemas.remote import RemoteWorkerSpec


class RemoteWorkerRegistry:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or os.getenv("OPTIRESEARCH_REMOTE_WORKER_ROOT", "workspace/remote_workers"))
        self.config_path = self.root / "workers.json"

    def list_workers(self) -> list[RemoteWorkerSpec]:
        payload = self._read()
        return [RemoteWorkerSpec(**item) for item in payload.get("workers", [])]

    def get_worker(self, worker_id: str) -> RemoteWorkerSpec:
        for worker in self.list_workers():
            if worker.worker_id == worker_id:
                return worker
        raise KeyError(f"Unknown remote worker: {worker_id}")

    def add_worker(self, worker: RemoteWorkerSpec) -> RemoteWorkerSpec:
        workers = [item for item in self.list_workers() if item.worker_id != worker.worker_id]
        workers.append(worker)
        self._write({"workers": [item.model_dump(mode="json") for item in workers]})
        return worker

    def _read(self) -> dict:
        if not self.config_path.exists():
            return {"workers": []}
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    def _write(self, payload: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_remote_worker_requirements(
    handler_capability: Any,
    worker_id: str,
) -> dict[str, Any]:
    """Validate that a worker can run a remote handler through the safe path."""

    required = list(getattr(handler_capability, "remote_worker_requirements", []) or [])
    handler_id = getattr(handler_capability, "handler_id", "")
    result: dict[str, Any] = {
        "handler_id": handler_id,
        "worker_id": worker_id,
        "requirements_met": False,
        "missing_requirements": [],
        "worker_tags": [],
        "command_allowlisted": False,
        "allowed_command_valid": False,
        "max_runtime_valid": False,
        "artifact_return_path_valid": False,
        "remote_command": [],
        "cli_command": "run-deeplens-native-geolens-hsi-codesign",
        "stop_reason": "",
        "final_claim_ceiling": "needs_followup",
        "errors": [],
        "warnings": [],
    }

    try:
        worker = RemoteWorkerRegistry().get_worker(worker_id)
    except KeyError:
        result["missing_requirements"] = ["worker_exists"]
        result["stop_reason"] = "remote_worker_requirements_not_met"
        result["errors"].append({"type": "REMOTE_WORKER_NOT_FOUND", "message": f"Unknown remote worker: {worker_id}"})
        return result

    tags = _effective_worker_tags(worker)
    result["worker_tags"] = sorted(tags)
    missing = [tag for tag in required if tag not in tags]
    result["missing_requirements"] = missing

    command = _remote_handler_probe_command(worker)
    result["remote_command"] = command
    try:
        validation = validate_remote_command(command)
        result["command_allowlisted"] = bool(validation.get("allowed"))
        result["cli_command"] = validation.get("cli_command", result["cli_command"])
    except CommandValidationError as exc:
        result["errors"].append({"type": "REMOTE_COMMAND_NOT_ALLOWLISTED", "message": str(exc)})

    allowed_commands = worker.capabilities.get("allowed_commands")
    if isinstance(allowed_commands, list) and allowed_commands:
        result["allowed_command_valid"] = result["cli_command"] in allowed_commands
        if not result["allowed_command_valid"]:
            result["errors"].append({
                "type": "REMOTE_COMMAND_NOT_ALLOWED_BY_WORKER",
                "message": f"{result['cli_command']} is not enabled for worker {worker.worker_id}",
            })
    else:
        result["allowed_command_valid"] = True
        result["warnings"].append({
            "type": "REMOTE_WORKER_ALLOWED_COMMANDS_UNSPECIFIED",
            "message": "Worker has no allowed_commands list; global remote allowlist was used.",
        })

    default_timeout = int(getattr(handler_capability, "default_timeout_sec", 0) or 0)
    result["max_runtime_valid"] = worker.max_runtime_seconds > 0 and (
        default_timeout <= 0 or worker.max_runtime_seconds >= default_timeout
    )
    if not result["max_runtime_valid"]:
        result["errors"].append({
            "type": "REMOTE_WORKER_RUNTIME_TOO_LOW",
            "message": f"Worker max_runtime_seconds={worker.max_runtime_seconds} is lower than handler timeout {default_timeout}",
        })

    artifact_root = worker.capabilities.get("artifact_return_path") or f"{worker.remote_workspace_dir.rstrip('/')}/remote_jobs"
    result["artifact_return_path"] = artifact_root
    result["artifact_return_path_valid"] = (
        isinstance(artifact_root, str)
        and artifact_root.startswith("/")
        and artifact_root.startswith(worker.remote_workspace_dir.rstrip("/") + "/")
    )
    if not result["artifact_return_path_valid"]:
        result["errors"].append({
            "type": "REMOTE_ARTIFACT_RETURN_PATH_INVALID",
            "message": "artifact_return_path must be an absolute path below remote_workspace_dir",
        })

    result["requirements_met"] = (
        not missing
        and result["command_allowlisted"]
        and result["allowed_command_valid"]
        and result["max_runtime_valid"]
        and result["artifact_return_path_valid"]
    )
    if not result["requirements_met"]:
        result["stop_reason"] = "remote_worker_requirements_not_met"
    else:
        result["final_claim_ceiling"] = getattr(handler_capability, "remote_evidence_ceiling", "") or "native_lens_simulation"
    return result


def _effective_worker_tags(worker: RemoteWorkerSpec) -> set[str]:
    tags = set(worker.backend_tags)
    tags.add(worker.worker_id)
    capability_tags = worker.capabilities.get("tags", [])
    if isinstance(capability_tags, list):
        tags.update(str(tag) for tag in capability_tags)
    for key, value in worker.capabilities.items():
        if value is True:
            tags.add(str(key))
    if worker.worker_id == "windows_wsl" or "wsl" in tags:
        tags.add("windows_wsl")
    if "deeplens" in tags:
        tags.add("deeplens_available")
    if "deeplens_geolens_geometric" in tags:
        tags.add("geolens_psf_geometric")
    return tags


def _remote_handler_probe_command(worker: RemoteWorkerSpec) -> list[str]:
    return [
        worker.python_executable,
        "-m",
        "optiresearch.cli",
        "run-deeplens-native-geolens-hsi-codesign",
        "--lens-file",
        "auto:cooke",
        "--dataset",
        "synthetic",
        "--reconstructor",
        "differentiable_linear",
        "--max-steps",
        "5",
        "--optical-lr",
        "1e-06",
        "--rollback-on-loss-increase",
        "--device",
        "cpu",
        "--remote-job-id",
        "remote_job_1436c05c2c4d6359",
    ]
