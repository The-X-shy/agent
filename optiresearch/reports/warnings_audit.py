"""Pytest warnings audit and classification.

Classifies warnings into categories:
  - deprecation
  - optional_dependency
  - test_skip
  - numerical
  - file/path
  - unknown
"""

from __future__ import annotations

import os
from pathlib import Path


def classify_warnings(lines: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {
        "deprecation": 0,
        "optional_dependency": 0,
        "test_skip": 0,
        "numerical": 0,
        "file_path": 0,
        "unknown": 0,
    }
    for line in lines:
        lower = line.lower()
        if any(kw in lower for kw in ["deprecationwarning", "pytestremovedin8warning", "pendingdeprecationwarning"]):
            counts["deprecation"] += 1
        elif any(kw in lower for kw in ["optional", "torch not available", "tensorflow not", "import error", "modulenotfound"]):
            counts["optional_dependency"] += 1
        elif any(kw in lower for kw in ["skipped", "skipif", "skip"]):
            counts["test_skip"] += 1
        elif any(kw in lower for kw in ["runtimewarning", "invalid value", "divide by zero", "overflow", "underflow"]):
            counts["numerical"] += 1
        elif any(kw in lower for kw in ["filenotfound", "nosuchfile", "path", "directory not found"]):
            counts["file_path"] += 1
        else:
            counts["unknown"] += 1
    return counts


class WarningsAudit:
    """Parse and classify pytest warnings, generate audit report."""

    def generate_report(self, lines: list[str]) -> str:
        counts = classify_warnings(lines)
        total = sum(counts.values())
        md_lines = [
            "# Warnings Audit",
            "",
            f"**Total warnings:** {total}",
            "",
            "## Classification",
            "",
            "| Category | Count |",
            "|---|---|",
        ]
        for cat, count in counts.items():
            md_lines.append(f"| {cat.replace('_', ' ').title()} | {count} |")
        md_lines.extend([
            "",
            "## Categories",
            "",
            "- **Deprecation**: Warnings about deprecated APIs — review and update to current APIs.",
            "- **Optional Dependency**: Missing optional packages (torch, tensorflow, etc.) — expected when optional deps not installed.",
            "- **Test Skip**: Tests skipped due to missing conditions (DeepLens, datasets, etc.) — expected behavior.",
            "- **Numerical**: RuntimeWarnings about numerical operations — review for numerical stability.",
            "- **File/Path**: File or path related warnings — check workspace paths.",
            "- **Unknown**: Warnings not matching known categories — review individually.",
            "",
            "## Recommendation",
            "",
            "1. Deprecation warnings should be addressed by updating to current APIs.",
            "2. Optional dependency warnings are expected and can be ignored if the dependency is not needed.",
            "3. Test skip warnings confirm structured skip behavior is working correctly.",
            "4. Numerical warnings should be reviewed for potential precision issues.",
        ])
        return "\n".join(md_lines)

    def export_report(self, lines: list[str] | None = None, output_dir: Path | None = None) -> Path:
        root = output_dir or Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
        root.mkdir(parents=True, exist_ok=True)
        path = root / "warnings_audit.md"
        path.write_text(self.generate_report(lines or []), encoding="utf-8")
        return path
