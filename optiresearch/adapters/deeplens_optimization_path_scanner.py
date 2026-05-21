"""Compatibility wrapper for the Phase 19B DeepLens optimization path scanner."""

from __future__ import annotations

from optiresearch.adapters.deeplens_native_inspector import (
    DeepLensOptimizationPathScanner,
    export_optimization_path_scan,
)

__all__ = ["DeepLensOptimizationPathScanner", "export_optimization_path_scan"]
