"""Benchmark report writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_reports(report: dict[str, Any], report_root: Path) -> dict[str, str]:
    report_root.mkdir(parents=True, exist_ok=True)
    json_path = report_root / "opti_memory_bench_report.json"
    md_path = report_root / "opti_memory_bench_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# OptiMemoryBench Report", "", f"Task count: {report['summary']['task_count']}", ""]
    for task in report["tasks"]:
        lines.append(f"## {task['task_type']}")
        for key, value in task["metrics"].items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    return "\n".join(lines)
