"""Test DeepLens source inspector."""
import os
from optiresearch.adapters.deeplens_source_inspector import DeepLensSourceInspector


def test_inspector_detects_repo_path():
    inspector = DeepLensSourceInspector("/Users/lilin/Desktop/external/DeepLens")
    assert inspector.available is True


def test_inspector_scan_returns_modules():
    inspector = DeepLensSourceInspector("/Users/lilin/Desktop/external/DeepLens")
    result = inspector.scan()
    assert result["available"] is True
    assert "modules" in result
    assert "paraxiallens" in result["modules"] or "geolens" in result["modules"]


def test_inspector_finds_classes():
    inspector = DeepLensSourceInspector("/Users/lilin/Desktop/external/DeepLens")
    result = inspector.scan()
    assert "classes" in result
    total_classes = sum(len(v) for v in result["classes"].values())
    assert total_classes > 0


def test_inspector_finds_likely_methods():
    inspector = DeepLensSourceInspector("/Users/lilin/Desktop/external/DeepLens")
    result = inspector.scan()

    assert "likely_psf_methods" in result
    assert "likely_optimization_methods" in result
    assert "likely_surface_classes" in result
    assert "likely_phase_classes" in result
    assert "likely_doe_classes" in result


def test_inspector_without_repo_is_unavailable():
    inspector = DeepLensSourceInspector("/nonexistent/path")
    assert inspector.available is False
    result = inspector.scan()
    assert result["available"] is False
