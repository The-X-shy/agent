"""Real DeepLens Fresnel component probe test.

Requires DeepLens to be installed locally.  Uses the local runtime directly.
"""

import os

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS") != "1",
    reason="Real DeepLens test requires OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS=1",
)


class TestRealDeepLensFresnelComponentProbe:
    def test_fresnel_component_probe_real(self):
        from optiresearch.schemas.component_probe import ComponentProbeSpec, make_component_probe_id
        from optiresearch.runtime.deeplens_component_first_probe import run_deeplens_component_probe

        spec = ComponentProbeSpec(
            probe_id=make_component_probe_id("fresnel"),
            component="fresnel",
            objective="parameter_sanity_check",
            max_steps=3,
            device="cpu",
        )
        result = run_deeplens_component_probe(spec)
        assert result.component == "fresnel"
        assert result.status in (
            "succeeded", "needs_followup", "structured_unavailable", "failed",
        )
        assert result.evidence_level == "diagnostic_evidence"
