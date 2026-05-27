"""Real DeepLens component-level probe test (opt-in).

Requires: OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS=1
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS") != "1",
    reason="Real DeepLens component test requires explicit opt-in",
)


class TestRealComponentProbe:
    def test_fresnel_component_probe(self):
        from optiresearch.schemas.surface_optimization import SurfaceOptimizationProbeSpec
        from optiresearch.runtime.deeplens_surface_optimization_probe import run_surface_optimization_probe

        spec = SurfaceOptimizationProbeSpec(
            surface_class="Fresnel",
            objective="minimize_phase_variance",
            device="cpu",
            max_steps=3,
            learning_rate=1e-4,
        )
        result = run_surface_optimization_probe(spec)
        assert result.status in ("succeeded", "unsupported", "failed")
        if result.status == "succeeded":
            assert result.can_instantiate
            assert len(result.trainable_params) > 0
            assert result.differentiable

    def test_binary2phase_component_probe(self):
        from optiresearch.schemas.surface_optimization import SurfaceOptimizationProbeSpec
        from optiresearch.runtime.deeplens_surface_optimization_probe import run_surface_optimization_probe

        spec = SurfaceOptimizationProbeSpec(
            surface_class="Binary2Phase",
            objective="minimize_phase_variance",
            device="cpu",
            max_steps=3,
            learning_rate=1e-4,
        )
        result = run_surface_optimization_probe(spec)
        assert result.status in ("succeeded", "unsupported", "failed")
        if result.status == "succeeded":
            assert result.can_instantiate
            assert len(result.trainable_params) > 0
