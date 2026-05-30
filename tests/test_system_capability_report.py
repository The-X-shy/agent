"""Tests for system capability report."""
from __future__ import annotations

from pathlib import Path

from optiresearch.reports.system_capability_report import export_system_capability_report


def test_export_system_capability_report(tmp_path):
    md_path = export_system_capability_report(str(tmp_path))
    assert md_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert "System Capability Report" in content
    assert "## 1. System Overview" in content
    assert "## 13. What Not to Claim" in content


def test_report_contains_handler_table():
    out = Path("workspace/system_capability")
    out.mkdir(parents=True, exist_ok=True)
    md_path = export_system_capability_report()
    content = md_path.read_text(encoding="utf-8")
    assert "Handler Capability Table" in content


def test_report_contains_known_gaps():
    out = Path("workspace/system_capability")
    out.mkdir(parents=True, exist_ok=True)
    md_path = export_system_capability_report()
    content = md_path.read_text(encoding="utf-8")
    assert "Known Gaps" in content
