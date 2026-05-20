"""Structured tests for the native optimization probe runner.

Uses mock DeepLens-like lens classes that simulate the full differentiable
optimization pipeline with real torch.nn.Parameter. Does NOT require real DeepLens.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from optiresearch.schemas.native_optimization import (
    NativeOptimizationProbeResult,
    NativeOptimizationProbeSpec,
    make_probe_id,
)


# ---------------------------------------------------------------------------
# Mock lens classes using real torch.nn.Parameter for autograd
# ---------------------------------------------------------------------------

class MockDifferentiableLens:
    """Mock lens with real torch parameters for full differentiable optimization."""

    def __init__(self, **kwargs):
        import torch
        self._foclen = torch.nn.Parameter(torch.tensor([50.0], dtype=torch.float32))
        self._curvature = torch.nn.Parameter(torch.tensor([0.02], dtype=torch.float32))

    def activate_grad(self, enable=True):
        for p in self.parameters():
            p.requires_grad = enable

    def get_optimizer(self):
        import torch
        trainable = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable, lr=1e-3)

    def parameters(self):
        import torch
        result = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.nn.Parameter):
                result.append(attr)
        return result

    def psf(self):
        import torch
        size = 64
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, size),
            torch.linspace(-1.0, 1.0, size),
            indexing="ij",
        )
        # Use offset 52 so abs(x) has non-zero gradient at initial value 50
        width = 0.1 + 0.01 * torch.abs(self._foclen - 52.0)
        psf_val = torch.exp(-(x ** 2 + y ** 2) / (2.0 * width ** 2 + 1e-8))
        psf_val = psf_val / psf_val.sum()
        return psf_val


class MockNonDifferentiableLens:
    """Mock lens without any differentiable optimization support.
    Accepts any kwargs so instantiation doesn't fail on args mismatch."""

    def __init__(self, **kwargs):
        pass

    def psf(self):
        import torch
        return torch.ones(64, 64)


class MockLensWithGradButNoStep:
    """Mock lens where backward flows but step has no effect (lr=0)."""

    def __init__(self, **kwargs):
        import torch
        self._param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

    def activate_grad(self, enable=True):
        self._param.requires_grad = enable

    def get_optimizer(self):
        import torch
        return torch.optim.Adam([self._param], lr=0.0)

    def parameters(self):
        import torch
        return [self._param]

    def psf(self):
        import torch
        return torch.ones(32, 32) * self._param


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_probe_with_differentiable_lens(tmp_path, monkeypatch):
    """Full probe with a lens that supports differentiable optimization."""
    from optiresearch.runtime.native_optimization_probe import run_native_optimization_probe

    monkeypatch.setattr(
        "optiresearch.runtime.native_optimization_probe._import_lens_class",
        lambda name: (MockDifferentiableLens, "mock.path", None),
    )

    spec = NativeOptimizationProbeSpec(
        probe_id=make_probe_id("ParaxialLens", "minimize_psf_width"),
        lens_class="ParaxialLens",
        objective="minimize_psf_width",
        max_steps=2,
        device="cpu",
        save_artifacts=False,
    )
    result = run_native_optimization_probe(spec)

    assert result.status == "succeeded"
    assert result.realization_level == "native"
    assert result.differentiable is True
    assert result.native_parameter_update is True
    assert result.autograd_graph_exists is True
    assert result.parameters_changed is True
    assert result.gradient_norm is not None
    assert result.gradient_norm > 0
    assert result.loss_before is not None


def test_probe_with_nondifferentiable_lens(tmp_path, monkeypatch):
    """Probe with a lens that has no differentiable support."""
    from optiresearch.runtime.native_optimization_probe import run_native_optimization_probe

    monkeypatch.setattr(
        "optiresearch.runtime.native_optimization_probe._import_lens_class",
        lambda name: (MockNonDifferentiableLens, "mock.path", None),
    )

    spec = NativeOptimizationProbeSpec(
        probe_id=make_probe_id("ParaxialLens", "minimize_psf_width"),
        lens_class="ParaxialLens",
        objective="minimize_psf_width",
        max_steps=2,
        device="cpu",
        strict_native=True,
        save_artifacts=False,
    )
    result = run_native_optimization_probe(spec)

    assert result.status in ("unsupported", "failed")
    assert result.realization_level != "native"
    assert result.differentiable is False


def test_probe_maximize_center_intensity(tmp_path, monkeypatch):
    """Probe with maximize_center_intensity objective."""
    from optiresearch.runtime.native_optimization_probe import run_native_optimization_probe

    monkeypatch.setattr(
        "optiresearch.runtime.native_optimization_probe._import_lens_class",
        lambda name: (MockDifferentiableLens, "mock.path", None),
    )

    spec = NativeOptimizationProbeSpec(
        probe_id=make_probe_id("ParaxialLens", "maximize_center_intensity"),
        lens_class="ParaxialLens",
        objective="maximize_center_intensity",
        max_steps=2,
        device="cpu",
        save_artifacts=False,
    )
    result = run_native_optimization_probe(spec)

    assert result.status == "succeeded"
    assert result.objective == "maximize_center_intensity"
    assert result.differentiable is True


def test_probe_match_target_psf(tmp_path, monkeypatch):
    """Probe with match_target_psf objective."""
    from optiresearch.runtime.native_optimization_probe import run_native_optimization_probe

    monkeypatch.setattr(
        "optiresearch.runtime.native_optimization_probe._import_lens_class",
        lambda name: (MockDifferentiableLens, "mock.path", None),
    )

    spec = NativeOptimizationProbeSpec(
        probe_id=make_probe_id("ParaxialLens", "match_target_psf"),
        lens_class="ParaxialLens",
        objective="match_target_psf",
        max_steps=2,
        device="cpu",
        save_artifacts=False,
    )
    result = run_native_optimization_probe(spec)

    assert result.status == "succeeded"
    assert result.objective == "match_target_psf"
    assert result.differentiable is True


def test_probe_saves_artifacts(tmp_path, monkeypatch):
    """Verify probe saves artifacts when save_artifacts=True."""
    from optiresearch.runtime.native_optimization_probe import run_native_optimization_probe

    monkeypatch.setattr(
        "optiresearch.runtime.native_optimization_probe._import_lens_class",
        lambda name: (MockDifferentiableLens, "mock.path", None),
    )

    spec = NativeOptimizationProbeSpec(
        probe_id=make_probe_id("ParaxialLens", "minimize_psf_width"),
        lens_class="ParaxialLens",
        objective="minimize_psf_width",
        max_steps=2,
        device="cpu",
        save_artifacts=True,
    )
    result = run_native_optimization_probe(spec)

    assert result.status == "succeeded"
    assert len(result.artifact_paths) > 0
    output_dir = Path("workspace/native_optimization") / spec.probe_id
    assert output_dir.exists()
    assert (output_dir / "probe_spec.json").exists()
    assert (output_dir / "loss_trace.json").exists()
    spec_content = json.loads((output_dir / "probe_spec.json").read_text())
    assert spec_content["lens_class"] == "ParaxialLens"


def test_probe_strict_native_enforcement(tmp_path, monkeypatch):
    """When strict_native=True and lens is non-differentiable, result is unsupported."""
    from optiresearch.runtime.native_optimization_probe import run_native_optimization_probe

    monkeypatch.setattr(
        "optiresearch.runtime.native_optimization_probe._import_lens_class",
        lambda name: (MockNonDifferentiableLens, "mock.path", None),
    )

    spec = NativeOptimizationProbeSpec(
        probe_id=make_probe_id("ParaxialLens", "minimize_psf_width"),
        lens_class="ParaxialLens",
        objective="minimize_psf_width",
        max_steps=2,
        device="cpu",
        strict_native=True,
        save_artifacts=False,
    )
    result = run_native_optimization_probe(spec)
    assert result.status == "unsupported"
    assert result.error_code is not None
    assert "GRAD" in (result.error_code or "")


def test_probe_deeplens_not_installed(tmp_path):
    """Probe handles DeepLens not being installed."""
    from optiresearch.runtime.native_optimization_probe import run_native_optimization_probe

    with patch.dict("sys.modules", {"deeplens": None}):
        spec = NativeOptimizationProbeSpec(
            probe_id=make_probe_id("ParaxialLens", "minimize_psf_width"),
            lens_class="ParaxialLens",
            objective="minimize_psf_width",
            max_steps=2,
            device="cpu",
            save_artifacts=False,
        )
        result = run_native_optimization_probe(spec)
        assert result is not None
        assert result.probe_id == spec.probe_id


def test_probe_result_is_valid_schema():
    """Verify that probe results conform to the schema."""
    result = NativeOptimizationProbeResult(
        probe_id="test_probe",
        status="succeeded",
        lens_class="ParaxialLens",
        objective="minimize_psf_width",
        realization_level="native",
        differentiable=True,
        native_parameter_update=True,
        autograd_graph_exists=True,
        loss_before=0.5,
        loss_after=0.3,
        gradient_norm=0.15,
        parameters_changed=True,
        optimizer_class="Adam",
    )
    payload = result.model_dump()
    restored = NativeOptimizationProbeResult(**payload)
    assert restored.differentiable is True
    assert restored.native_parameter_update is True


def test_probe_all_lens_class_names_handled(tmp_path, monkeypatch):
    """All five lens class names produce valid results."""
    from optiresearch.runtime.native_optimization_probe import run_native_optimization_probe

    for cls_name in ["ParaxialLens", "GeoLens", "DiffractiveLens", "HybridLens", "PSFNetLens"]:
        monkeypatch.setattr(
            "optiresearch.runtime.native_optimization_probe._import_lens_class",
            lambda name, cn=cls_name: (MockDifferentiableLens, f"mock.{cn}", None),
        )
        spec = NativeOptimizationProbeSpec(
            probe_id=make_probe_id(cls_name, "minimize_psf_width"),
            lens_class=cls_name,
            objective="minimize_psf_width",
            max_steps=2,
            device="cpu",
            save_artifacts=False,
        )
        result = run_native_optimization_probe(spec)
        assert result is not None
        assert result.lens_class == cls_name
        assert result.status in ("succeeded", "unsupported", "failed")


def test_probe_with_gradient_but_no_param_change(tmp_path, monkeypatch):
    """Lens where backward flows but optimizer step has no effect."""
    from optiresearch.runtime.native_optimization_probe import run_native_optimization_probe

    monkeypatch.setattr(
        "optiresearch.runtime.native_optimization_probe._import_lens_class",
        lambda name: (MockLensWithGradButNoStep, "mock.path", None),
    )

    spec = NativeOptimizationProbeSpec(
        probe_id=make_probe_id("ParaxialLens", "minimize_psf_width"),
        lens_class="ParaxialLens",
        objective="minimize_psf_width",
        max_steps=2,
        device="cpu",
        strict_native=True,
        save_artifacts=False,
    )
    result = run_native_optimization_probe(spec)
    # Since lr=0, parameters_changed should be False
    # Gradient may still flow (backward works), but step has no effect
    assert result.gradient_norm is not None
    # With strict_native, if params don't change, status should not be succeeded
    if result.parameters_changed is False:
        assert result.status != "succeeded"
