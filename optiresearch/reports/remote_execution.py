"""Remote execution report export."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def export_remote_execution_report(
    job_id: str,
    remote_jobs_root: str | Path | None = None,
) -> Path:
    root = Path(remote_jobs_root or os.getenv("OPTIRESEARCH_REMOTE_JOBS_ROOT", "workspace/remote_jobs"))
    job_dir = root / job_id
    result = _read_json(job_dir / "remote_job_result.json", {})
    ingestion = _read_json(job_dir / "ingestion_summary.json", {})
    path = job_dir / "remote_execution_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_markdown(job_id, result, ingestion), encoding="utf-8")
    return path


def _markdown(job_id: str, result: dict[str, Any], ingestion: dict[str, Any]) -> str:
    metrics = result.get("metrics_summary", {})
    lines = [
        "# Remote Execution Report",
        "",
        f"**Job ID:** `{job_id}`",
        f"**Status:** {result.get('status', 'unknown')}",
        f"**Remote run ID:** `{result.get('remote_run_id')}`",
        f"**Error code:** {result.get('error_code')}",
        f"**Fallback used:** {metrics.get('fallback_used')}",
        f"**Local output dir:** `{result.get('local_output_dir', '')}`",
        "",
        "## Metrics",
        "",
        "| Key | Value |",
        "|---|---|",
    ]
    for key, value in sorted(metrics.items()):
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Artifacts", ""])
    for artifact_id in ingestion.get("artifact_ids", []):
        lines.append(f"- `{artifact_id}`")
    if not ingestion.get("artifact_ids"):
        lines.append("(none)")

    lines.extend(["", "## Claims", ""])
    for claim in ingestion.get("claims", []):
        lines.append(f"- `{claim.get('status')}` {claim.get('claim_text') or claim.get('claim_id')}")
    if not ingestion.get("claims"):
        lines.append("(none)")

    lines.extend(["", "## Caveats", ""])
    caveats = result.get("caveats", [])
    if metrics.get("claim_scope"):
        caveats.append(metrics["claim_scope"])
    if not caveats:
        caveats.append("No caveats recorded.")
    for caveat in caveats:
        lines.append(f"- {caveat}")
    return "\n".join(lines) + "\n"


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
