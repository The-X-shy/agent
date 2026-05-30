"""Remote execution contract schema for Phase 68."""
from __future__ import annotations

from typing import Any

from optiresearch.memory.schemas import StrictModel


class RemoteExecutionContract(StrictModel):
    """Contract for a remote CLI command execution."""

    remote_contract_id: str
    command_name: str
    handler_id: str = ""
    allowed_args: list[str] = []
    forbidden_args: list[str] = []
    required_worker_capabilities: list[str] = []
    required_env_vars: list[str] = []
    propagated_env_vars: list[str] = []
    timeout_sec: int = 600
    output_dir_policy: str = "required"
    artifact_return_policy: str = "required"
    allowlist_entry_required: bool = True
    workspace_write_policy: str = "restricted"
    remote_job_id_required: bool = True
    result_parser: str = ""
    failure_parser: str = ""
    retry_policy: dict[str, Any] = {}
