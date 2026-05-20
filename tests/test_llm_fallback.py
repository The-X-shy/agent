from optiresearch.agents.lead_investigator import LeadInvestigator
from optiresearch.agents.method_builder import MethodBuilder


class UnavailableProvider:
    provider_name = "unavailable"
    model = "none"

    def available(self):
        return False


def test_agents_fallback_when_llm_unavailable():
    provider = UnavailableProvider()

    plan = LeadInvestigator().plan("Design fallback optics", use_llm=True, provider=provider)
    spec = MethodBuilder().build_experiment_spec_with_llm("Design fallback optics", {}, provider=provider)

    assert plan["llm_metadata"]["llm_used"] is False
    assert plan["llm_metadata"]["fallback_used"] is True
    assert spec.metadata.get("llm_used") is False
