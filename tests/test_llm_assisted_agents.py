from optiresearch.agents.lead_investigator import LeadInvestigator
from optiresearch.agents.method_builder import MethodBuilder
from optiresearch.agents.paper_writing_assistant import PaperWritingAssistant
from optiresearch.llm.mock_provider import MockLLMProvider
from optiresearch.schemas.experiment import ExperimentSpec


def test_lead_and_method_builder_support_llm_assisted_mode():
    provider = MockLLMProvider()

    plan = LeadInvestigator().plan("Design a mock EDOF-HSI encoder", use_llm=True, provider=provider)
    spec = MethodBuilder().build_experiment_spec_with_llm("Design a mock EDOF-HSI encoder", {}, provider=provider)

    assert plan["llm_metadata"]["llm_used"] is True
    assert plan["steps"]
    assert isinstance(spec, ExperimentSpec)
    assert spec.metadata["llm_used"] is True


def test_paper_writing_assistant_only_cites_existing_ids():
    draft = PaperWritingAssistant().draft_experiment_section(
        {
            "artifact_ids": ["artifact_1"],
            "claim_ids": ["claim_1"],
            "summary": "Depth stability was measured.",
        },
        provider=MockLLMProvider(),
    )

    assert draft.cited_artifact_ids == ["artifact_1"]
    assert draft.cited_claim_ids == ["claim_1"]
