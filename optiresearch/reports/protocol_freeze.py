"""Freeze the paper experiment protocol v0.1 into the workspace report folder."""

from __future__ import annotations

import os
from pathlib import Path


def freeze_paper_protocol() -> Path:
    source = Path("docs/paper_experiment_protocol_v0.1.md")
    root = Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
    root.mkdir(parents=True, exist_ok=True)
    target = root / "paper_experiment_protocol_v0.1_freeze.md"
    text = source.read_text(encoding="utf-8")
    target.write_text(text, encoding="utf-8")
    return target

