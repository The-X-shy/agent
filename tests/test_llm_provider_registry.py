from optiresearch.llm.deepseek_provider import DeepSeekProvider
from optiresearch.llm.mock_provider import MockLLMProvider
from optiresearch.llm.registry import get_llm_provider, list_llm_providers


def test_llm_registry_defaults_to_mock_without_keys(monkeypatch):
    monkeypatch.delenv("OPTIRESEARCH_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert isinstance(get_llm_provider(), MockLLMProvider)
    assert any(item["provider"] == "mock" for item in list_llm_providers())


def test_llm_registry_prefers_deepseek_when_key_exists(monkeypatch):
    monkeypatch.delenv("OPTIRESEARCH_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")

    assert isinstance(get_llm_provider(), DeepSeekProvider)
    assert isinstance(get_llm_provider("mock"), MockLLMProvider)
