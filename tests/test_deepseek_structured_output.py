from optiresearch.llm.base import LLMResponse
from optiresearch.llm.deepseek_provider import DeepSeekProvider
from optiresearch.llm.structured_output import ResearchPlanDraft


def test_deepseek_structured_output_parses_json(monkeypatch):
    provider = DeepSeekProvider(api_key="test-key")

    def fake_complete(messages, **kwargs):
        return LLMResponse(
            content='{"objective":"x","hypotheses":["h"],"candidate_experiments":["e"],"required_skills":["s"],"risks":[],"next_action":"run"}',
            provider="deepseek",
            model="deepseek-v4-pro",
        )

    monkeypatch.setattr(provider, "complete", fake_complete)

    draft = provider.structured_complete([{"role": "user", "content": "x"}], ResearchPlanDraft)

    assert draft.objective == "x"
    assert draft.next_action == "run"


def test_deepseek_structured_output_parses_markdown_fence(monkeypatch):
    provider = DeepSeekProvider(api_key="test-key")

    def fake_complete(messages, **kwargs):
        return LLMResponse(
            content='```json\n{"objective":"x","hypotheses":[],"candidate_experiments":[],"required_skills":[],"risks":[],"next_action":"run"}\n```',
            provider="deepseek",
            model="deepseek-v4-pro",
        )

    monkeypatch.setattr(provider, "complete", fake_complete)

    assert provider.structured_complete([{"role": "user", "content": "x"}], ResearchPlanDraft).objective == "x"


def test_deepseek_structured_output_repairs_invalid_json(monkeypatch):
    provider = DeepSeekProvider(api_key="test-key")
    calls = {"count": 0}

    def fake_complete(messages, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return LLMResponse(content="not json", provider="deepseek", model="deepseek-v4-pro")
        return LLMResponse(
            content='{"objective":"fixed","hypotheses":[],"candidate_experiments":[],"required_skills":[],"risks":[],"next_action":"run"}',
            provider="deepseek",
            model="deepseek-v4-pro",
        )

    monkeypatch.setattr(provider, "complete", fake_complete)

    draft = provider.structured_complete([{"role": "user", "content": "x"}], ResearchPlanDraft)

    assert draft.objective == "fixed"
    assert calls["count"] == 2
