"""Real DeepLens native optimization probe tests.

Opt-in tests that require actual DeepLens installation.
Skip unless OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS=1.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from optiresearch.schemas.native_optimization import (
    NativeOptimizationProbeSpec,
    make_probe_id,
)

pytestmark = pytest.mark.skipif(
    os.getenv("OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS") != "1",
    reason="Real DeepLens test requires explicit opt-in.",
)


def _deep_lens_available() -> bool:
    """Check if DeepLens can actually be imported."""
    try:
        import deeplens
        return True
    except ImportError:
        return False


requires_deeplens = pytest.mark.skipif(
    not _deep_lens_available(),
    reason="DeepLens package not importable.",
)


@requires_deeplens
def test_native_optimization_inspection_runs():
    """Inspector should produce structured output when DeepLens is available."""
    from optiresearch.adapters.deeplens_native_inspector import (
        DeepLensNativeOptimizationInspector,
    )
    inspector = DeepLensNativeOptimizationInspector()
    assert inspector.available is True
    result = inspector.scan()
    assert result["available"] is True
    assert "lens_classes" in result
    paraxial = result["lens_classes"]["ParaxialLens"]
    assert paraxial["class_available"] is True
    # ParaxialLens should have these methods
    assert paraxial["has_activate_grad"] is True, f"ParaxialLens missing activate_grad: {paraxial}"
    assert paraxial["has_get_optimizer"] is True, f"ParaxialLens missing get_optimizer: {paraxial}"


@requires_deeplens
def test_paraxial_lens_minimize_psf_width():
    """Full native probe on ParaxialLens with minimize_psf_width."""
    from optiresearch.runtime.native_optimization_probe import run_native_optimization_probe

    spec = NativeOptimizationProbeSpec(
        probe_id=make_probe_id("ParaxialLens", "minimize_psf_width"),
        lens_class="ParaxialLens",
        objective="minimize_psf_width",
        max_steps=2,
        learning_rate=1e-3,
        device="cpu",
        strict_native=False,  # Allow partial results for real lens
        save_artifacts=True,
    )
    result = run_native_optimization_probe(spec)
    assert result is not None
    assert result.probe_id == spec.probe_id
    assert result.status in ("succeeded", "unsupported", "failed")
    # If it succeeded, check native optimization markers
    if result.status == "succeeded":
        assert result.differentiable is True
        assert result.native_parameter_update is True
        assert result.gradient_norm is not None
        assert result.gradient_norm > 0
        assert result.parameters_changed is True
    else:
        # Structured unsupported result
        assert result.error_code is not None
        assert result.error_message is not None or len(result.caveats) > 0


@requires_deeplens
def test_all_lens_classes_inspection():
    """All five lens classes should have entries in the inspection."""
    from optiresearch.adapters.deeplens_native_inspector import (
        DeepLensNativeOptimizationInspector,
        LENS_CLASSES,
    )
    inspector = DeepLensNativeOptimizationInspector()
    result = inspector.scan()
    for cls_name in LENS_CLASSES:
        assert cls_name in result["lens_classes"], f"Missing {cls_name}"
        info = result["lens_classes"][cls_name]
        # Every class should at least report whether it's available
        assert "class_available" in info
        assert "has_activate_grad" in info


@requires_deeplens
def test_export_inspection_files(tmp_path, monkeypatch):
    """Export should write JSON and markdown files."""
    from optiresearch.adapters.deeplens_native_inspector import (
        export_native_optimization_inspection,
    )
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path))
    export_native_optimization_inspection()
    assert (tmp_path / "deeplens_native_optimization_inspection.json").exists()
    assert (tmp_path / "deeplens_native_optimization_inspection.md").exists()


@requires_deeplens
def test_probe_with_paraxial_lens_maximize_center():
    """ParaxialLens probe with maximize_center_intensity objective."""
    from optiresearch.runtime.native_optimization_probe import run_native_optimization_probe

    spec = NativeOptimizationProbeSpec(
        probe_id=make_probe_id("ParaxialLens", "maximize_center_intensity"),
        lens_class="ParaxialLens",
        objective="maximize_center_intensity",
        max_steps=2,
        device="cpu",
        strict_native=False,
        save_artifacts=False,
    )
    result = run_native_optimization_probe(spec)
    assert result is not None
    assert result.objective == "maximize_center_intensity"
    assert result.status in ("succeeded", "unsupported", "failed")
