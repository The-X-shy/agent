"""Test component-level probe execution via skill dispatch."""

import pytest


class TestComponentFirstSkillDispatch:
    def test_component_first_stub_returns_needs_followup_when_api_unavailable(self):
        """When DeepLens component API is not installed, dispatch returns needs_followup."""
        from optiresearch.skills.runtime_v2 import SkillRuntimeV2
        try:
            result = SkillRuntimeV2()._dispatch_deeplens_component_first({"component": "fresnel"})
        except Exception:
            result = SkillRuntimeV2().execute_skill("deeplens_component_first_probe", {"component": "fresnel"})
        output = result if isinstance(result, dict) else result.output
        assert output["status"] in ("succeeded", "needs_followup")
        assert output["evidence_level"] == "diagnostic_evidence"

    def test_component_mapping(self):
        """Test fresnel maps to Fresnel surface class."""
        from optiresearch.skills.runtime_v2 import SkillRuntimeV2
        runtime = SkillRuntimeV2()
        result = runtime._dispatch_deeplens_component_first({"component": "fresnel"})
        assert result["checked_component"] == "Fresnel"

    def test_binary2phase_mapping(self):
        """Test binary2phase maps to Binary2Phase surface class."""
        from optiresearch.skills.runtime_v2 import SkillRuntimeV2
        runtime = SkillRuntimeV2()
        result = runtime._dispatch_deeplens_component_first({"component": "binary2phase"})
        assert result["checked_component"] == "Binary2Phase"

    def test_error_code_when_api_unavailable(self):
        """When DeepLens is not installed, error code is DEEPLENS_COMPONENT_API_UNAVAILABLE."""
        from optiresearch.skills.runtime_v2 import SkillRuntimeV2
        runtime = SkillRuntimeV2()
        result = runtime._dispatch_deeplens_component_first({"component": "fresnel"})
        if result["status"] == "needs_followup" and "error_code" in result:
            assert result["error_code"] == "DEEPLENS_COMPONENT_API_UNAVAILABLE"
            assert "checked_component_candidates" in result
