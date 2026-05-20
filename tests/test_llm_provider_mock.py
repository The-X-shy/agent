from optiresearch.llm.mock_provider import MockLLMProvider
from optiresearch.llm.structured_output import ResearchPlanDraft


def test_mock_llm_provider_structured_response_is_deterministic():
    provider = MockLLMProvider()

    response = provider.complete([{"role": "user", "content": "Design optics"}])
    draft = provider.structured_complete([{"role": "user", "content": "Design optics"}], ResearchPlanDraft)

    assert provider.available() is True
    assert response.provider == "mock"
    assert "Design optics" in response.content
    assert draft.objective
    assert draft.next_action == "run first simulation"
