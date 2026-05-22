"""Planner trace and audit recording.

Records the full lifecycle of an LLM planner run for audit,
debugging, and reproducibility. All outputs are sanitized to
prevent credential leakage.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from optiresearch.llm.audit import SECRET_ENV_KEYS


def redact_api_keys(data: Any) -> Any:
    """Recursively replace API key values with [REDACTED]."""
    if isinstance(data, str):
        result = data
        for key in SECRET_ENV_KEYS:
            value = os.getenv(key)
            if value and value in result:
                result = result.replace(value, "[REDACTED]")
        return result
    if isinstance(data, dict):
        return {k: redact_api_keys(v) for k, v in data.items()}
    if isinstance(data, list):
        return [redact_api_keys(v) for v in data]
    return data


def redact_authorization_headers(data: dict[str, Any]) -> dict[str, Any]:
    """Redact Authorization and API key headers from a dict."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if key.lower() in ("authorization", "x-api-key", "api-key"):
            result[key] = "[REDACTED]"
        elif isinstance(value, dict):
            result[key] = redact_authorization_headers(value)
        elif isinstance(value, list):
            result[key] = [
                redact_authorization_headers(v) if isinstance(v, dict) else v
                for v in value
            ]
        else:
            result[key] = value
    return result


def redact_env_values(data: Any) -> Any:
    """Redact known environment variable values from data."""
    return redact_api_keys(data)


class PlannerTrace:
    """In-memory trace of a single planner run."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.output_dir = Path("workspace/planner_traces") / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict[str, Any]] = []

    def add(self, filename: str, data: Any) -> Path:
        path = self.output_dir / filename
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        self._entries.append({"file": filename, "time": time.time()})
        return path


def start_planner_trace(run_id: str) -> PlannerTrace:
    """Initialize a new planner trace."""
    return PlannerTrace(run_id)


def record_context(trace: PlannerTrace, context: dict[str, Any]) -> Path:
    sanitized = dict(context)
    sanitized.pop("api_key", None)
    sanitized.pop("DEEPSEEK_API_KEY", None)
    sanitized = redact_api_keys(sanitized)
    return trace.add("context_summary.json", sanitized)


def record_response(
    trace: PlannerTrace, raw_response: list[dict[str, Any]]
) -> Path:
    sanitized = redact_api_keys(raw_response)
    return trace.add("raw_response.json", sanitized)


def record_validation(
    trace: PlannerTrace,
    validated: list[tuple[Any, dict[str, Any]]],
) -> Path:
    report = []
    for proposal, result in validated:
        report.append({
            "proposal_id": proposal.proposal_id,
            "valid": result.get("valid", False),
            "errors": result.get("errors", []),
        })
    return trace.add("validation_report.json", report)


def record_selection(
    trace: PlannerTrace, selected: dict[str, Any]
) -> Path:
    return trace.add("selected_proposal.json", selected)


def finalize_trace(trace: PlannerTrace) -> str:
    trace.add("_trace_index.json", {
        "run_id": trace.run_id,
        "entries": trace._entries,
        "finalized_at": time.time(),
    })
    return str(trace.output_dir)


def list_planner_traces() -> list[dict[str, Any]]:
    """List all saved planner traces."""
    traces_dir = Path("workspace/planner_traces")
    if not traces_dir.exists():
        return []
    results: list[dict[str, Any]] = []
    for d in sorted(traces_dir.iterdir(), reverse=True):
        if d.is_dir():
            index = d / "_trace_index.json"
            results.append({
                "run_id": d.name,
                "has_index": index.exists(),
                "path": str(d),
            })
    return results


def inspect_planner_trace(run_id: str) -> Optional[dict[str, Any]]:
    """Inspect a saved planner trace."""
    trace_dir = Path("workspace/planner_traces") / run_id
    if not trace_dir.exists():
        return None

    info: dict[str, Any] = {"run_id": run_id, "files": {}}
    for f in sorted(trace_dir.iterdir()):
        if f.suffix == ".json":
            try:
                info["files"][f.name] = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                info["files"][f.name] = f"<unreadable: {f.stat().st_size} bytes>"
    return info
