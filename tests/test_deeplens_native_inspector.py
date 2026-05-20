"""Tests for DeepLens native optimization inspector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from optiresearch.adapters.deeplens_native_inspector import (
    DeepLensNativeOptimizationInspector,
    export_native_optimization_inspection,
    LENS_CLASSES,
)


class MockDeepLensAdapter:
    """Mock adapter that simulates DeepLens being unavailable."""

    def __init__(self, available: bool = False):
        self._available = available
        self._deeplens = "mock" if available else None

    def validate_environment(self) -> dict:
        if not self._available:
            return {
                "available": False,
                "error": {"code": "DEEPLENS_NOT_INSTALLED", "detail": "Mock unavailable"},
                "deeplens_version": None,
                "import_path": None,
                "is_source_checkout": False,
            }
        return {
            "available": True,
            "deeplens_version": "1.5.2-mock",
            "import_path": "/mock/deeplens/__init__.py",
            "is_source_checkout": False,
        }


def test_inspector_unavailable_when_deeplens_missing():
    inspector = DeepLensNativeOptimizationInspector(adapter=MockDeepLensAdapter(available=False))
    assert inspector.available is False
    result = inspector.scan()
    assert result["available"] is False
    assert "error" in result
    for cls_name in LENS_CLASSES:
        assert result["lens_classes"][cls_name]["class_available"] is False
        assert result["lens_classes"][cls_name]["unsupported_reason"] is not None


def test_inspector_scans_all_five_lens_classes():
    inspector = DeepLensNativeOptimizationInspector(adapter=MockDeepLensAdapter(available=False))
    result = inspector.scan()
    for cls_name in LENS_CLASSES:
        assert cls_name in result["lens_classes"], f"Missing {cls_name} in scan results"
    assert len(result["lens_classes"]) == 5


def test_unavailable_result_structure():
    inspector = DeepLensNativeOptimizationInspector(adapter=MockDeepLensAdapter(available=False))
    result = inspector.scan()
    info = result["lens_classes"]["ParaxialLens"]
    assert info["class_available"] is False
    assert info["has_activate_grad"] is False
    assert info["has_get_optimizer"] is False
    assert info["likely_differentiable"] is False
    assert info["unsupported_reason"] is not None


def test_export_writes_files(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path))
    mock = MockDeepLensAdapter(available=False)
    result = export_native_optimization_inspection(adapter=mock)
    assert result["available"] is False
    json_path = tmp_path / "deeplens_native_optimization_inspection.json"
    md_path = tmp_path / "deeplens_native_optimization_inspection.md"
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["available"] is False


def test_inspection_result_is_json_serializable():
    inspector = DeepLensNativeOptimizationInspector(adapter=MockDeepLensAdapter(available=False))
    result = inspector.scan()
    json_str = json.dumps(result, default=str)
    assert len(json_str) > 0
    restored = json.loads(json_str)
    assert "lens_classes" in restored


def test_inspector_accepts_no_adapter():
    inspector = DeepLensNativeOptimizationInspector()
    assert isinstance(inspector.available, bool)


def test_markdown_output_not_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path))
    mock = MockDeepLensAdapter(available=False)
    export_native_optimization_inspection(adapter=mock)
    md_path = tmp_path / "deeplens_native_optimization_inspection.md"
    content = md_path.read_text(encoding="utf-8")
    assert "DeepLens" in content
    assert len(content) > 100
