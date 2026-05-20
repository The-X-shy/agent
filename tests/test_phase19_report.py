"""Tests for Phase 19 report export."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_phase19_report_module_importable():
    """Phase 19 report module should be importable."""
    from optiresearch.reports.phase19 import export_phase19_report
    assert callable(export_phase19_report)


def test_phase19_report_returns_path(tmp_path, monkeypatch):
    """export_phase19_report should return a Path."""
    from optiresearch.reports.phase19 import export_phase19_report
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path))
    path = export_phase19_report()
    assert isinstance(path, Path)
    assert path.exists()


def test_phase19_report_content(tmp_path, monkeypatch):
    """Report should contain expected sections."""
    from optiresearch.reports.phase19 import export_phase19_report
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path))
    path = export_phase19_report()
    content = path.read_text(encoding="utf-8")
    assert "Phase 19" in content
    assert "Native" in content
    assert "Optimization" in content
